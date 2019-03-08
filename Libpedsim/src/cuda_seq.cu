// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cuda.h>
//#include "cude_runtime.h"
//#include <cuda_runtime_api>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

using namespace std;

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

// Sets up the heatmap
void Ped::Model::setupHeatmapCuda()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE * sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE * sizeof(int));

	heatmap = (int**)malloc(SIZE * sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));

	x_arr_host = (int *)malloc(agents.size() * sizeof(int));
	y_arr_host = (int *)malloc(agents.size() * sizeof(int));

	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE * i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE * i;
		blurred_heatmap[i] = bhm + SCALED_SIZE * i;
	}

	
	cudaMalloc(&heatmap_cuda, (SIZE*SIZE) * sizeof(int));
	cudaMalloc(&scaled_cuda, (SCALED_SIZE*SCALED_SIZE) * sizeof(int));
	cudaMalloc(&blurred_cuda, (SCALED_SIZE*SCALED_SIZE) * sizeof(int));
	cudaMalloc(&x_arr, agents.size() * sizeof(int));
	cudaMalloc(&y_arr, agents.size() * sizeof(int));

	cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_cuda, *scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(blurred_cuda, *blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
}


__global__ 
void kernel_func_1(int *heatmap_cuda) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	heatmap_cuda[thread_id] = (int)round(heatmap_cuda[thread_id] * 0.80);
}

__global__
void kernel_func_2(int *heatmap_cuda, int *x_arr, int *y_arr) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int x = x_arr[thread_id];
	int y = y_arr[thread_id];


	if ((x < 0 || x >= SIZE || y < 0 || y >= SIZE) == false) {
		// Gånger size för välja vilken rad, dvs rad * antal element
		// x värdet för vilket x värde i raden, dvs kolumnen 
#ifdef __CUDACC__
		atomicAdd(&heatmap_cuda[x + y * SIZE], 40);
#endif
	}

}

__global__ 
void kernel_func_3(int *heatmap_cuda) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (heatmap_cuda[thread_id] < 255) {
		heatmap_cuda[thread_id] = heatmap_cuda[thread_id];
	}
	else {
		heatmap_cuda[thread_id] = 255; 
	}
}

__global__ void kernel_func_4(int *heatmap_cuda, int *scaled_cuda) {
	// Blockdim är strokleken på blocket, hur bred. Hur bred den är i x led detta fall
	// dvs hur många x den har

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int value = heatmap_cuda[thread_id];

	int row = thread_id / SIZE;
	int col = thread_id % SIZE;

	for (int cellY = 0; cellY < CELLSIZE; cellY++)
	{
		for (int cellX = 0; cellX < CELLSIZE; cellX++)
		{	
			
			// Tråden gånger cellsize och lägg på vilken cell vi befinner oss i
			// FÖR HOPPA I RADER ANVÄNDER MAN SCALED SIZE
			scaled_cuda[(row * CELLSIZE + cellY)*SCALED_SIZE + (col * CELLSIZE + cellX)] = value;
		}
	}
}

__global__ void kernel_func_5(int *scaled_cuda, int *blurred_cuda) {

	__shared__ int s[32][32];
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};
//	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	// threadIdx.x är var tråden ligger i x led i sitt block
	// threadIdy.y är var den ligger i y led i sitt block
	int x = threadIdx.x;
	int y = threadIdx.y;
	// blockidx.y är vart blocket ligger i y led i gridden. gånger blocksize(32), + y för hitta ar tråden i är blocket
	int row = blockIdx.y * 32 + y;
	// blockidx.x är vart blocket ligger i x led i gridden 
	int col = blockIdx.x * 32 + x;

	// Alla trådar i ett block kommer att dela
	// minne, det är vad s matrisen de
	// Alla trådar i ett block delar s-matrisen,
	// där varje position (y,x) används av en tråd
	s[y][x] = scaled_cuda[row * SCALED_SIZE + col];
	__syncthreads();
	//int row = thread_id / SIZE;
	//int col = thread_id % SIZE;
	int sum = 0; 

#define WEIGHTSUM 273
	for (int k = -2; k < 3; k++)
	{
		for (int l = -2; l < 3; l++)
		{
			// Kollar i princip så att den inte är out of bound
			if (y + k < 0 || y + k > 31 || x + l < 0 || x + l > 31) {
				sum += w[2 + k][2 + l] * scaled_cuda[(row + k + 2)*SCALED_SIZE + (col + l + 2)];
			}
			// if it's out of bound, that is we are checking outside of the cell
			else {
				sum += w[2 + k][2 + l] * s[y + k][x + l];
			}
			// +2 because we want to take in 2 extra 
			
				
				//scaled_cuda[(row * CELLSIZE + k)*SCALED_SIZE + (col * CELLSIZE + l)];//scaled_heatmap[SCALED_SIZE - 2 + k][SCALED_SIZE - 2 + l];
		}
	}
	int value = sum / WEIGHTSUM;
	// Vi vill ta raden * scaledSize för få rätt rad + kolumner för rätt cell
	blurred_cuda[row * SCALED_SIZE + col] = 0x00FF0000 | value << 24;
	/*

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}*/
}


// Updates the heatmap according to the agent positions
// TODO: Parallilize this using CUDA

void Ped::Model::updateHeatmapCuda()
{
	// for kernel func 1, fade
	cudaEvent_t start1, stop1; 
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);


	// For kernel func 2, agents 
	// Allocate arrs for taking care of this

	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++) {
		Ped::Tagent* agent = agents[i];
		x_arr_host[i] = agent->getDesiredX();
		y_arr_host[i] = agent->getDesiredY();
	}

	cudaMemcpy(x_arr, x_arr_host, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(y_arr, y_arr_host, agents.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start1);
	kernel_func_1<<<SIZE, SIZE >>>(heatmap_cuda);
	


	//cudaDeviceSynchronize();

	// Ett block med agents.size() antal trådar
	kernel_func_2<<<1, agents.size()>>>(heatmap_cuda, x_arr, y_arr);


	// For kernel func 3
	//cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	kernel_func_3<<<SIZE, SIZE >>>(heatmap_cuda);
	cudaEventRecord(stop1);



	//cudaDeviceSynchronize();
	//cudaMemcpy(*heatmap, heatmap_cuda, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	// For kernel func 4
	//cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	
	dim3 dimBlock4(512, 2);
	dim3 dimGrid4(512, 2);
	cudaEventRecord(start2);
	kernel_func_4<<<dimGrid4, dimBlock4 >>>(heatmap_cuda, scaled_cuda);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop2);

	// For kernel func 5
	//cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(scaled_cuda, *scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	/*cudaMemcpy(blurred_cuda, *blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);*/
	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	dim3 dimBlock(32, 32);
	dim3 dimGrid(160, 160);

	cudaEventRecord(start3);
	kernel_func_5 <<<dimGrid, dimBlock >>>(scaled_cuda, blurred_cuda);
	cudaEventRecord(stop3);

	//cudaMemcpy(*scaled_heatmap, scaled_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*blurred_heatmap, blurred_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	

	float timeMS = 0;
	cudaEventElapsedTime(&timeMS, start1, stop1);
	cout << "Time is: " << timeMS << endl;
	float timeMS2 = 0;
	cudaEventElapsedTime(&timeMS2, start2, stop2);
	cout << "Time for kernelfunc4 is ...: " << timeMS2 << endl;

	float timeMS3 = 0;
	cudaEventElapsedTime(&timeMS3, start3, stop3);
	cout << "Time for kernelfunc5 is ...: " << timeMS3 << endl;
	//cudaMemcpy(*heatmap, heatmap_cuda, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	// Scale the data for visual representation
	//for (int y = 0; y < SIZE; y++)
	//{
	//	for (int x = 0; x < SIZE; x++)
	//	{
	//		int value = heatmap[y][x];
	//		for (int cellY = 0; cellY < CELLSIZE; cellY++)
	//		{
	//			for (int cellX = 0; cellX < CELLSIZE; cellX++)
	//			{
	//				scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
	//			}
	//		}
	//	}
	//}
	
	// Weights for blur filter
//	const int w[5][5] = {
//		{ 1, 4, 7, 4, 1 },
//	{ 4, 16, 26, 16, 4 },
//	{ 7, 26, 41, 26, 7 },
//	{ 4, 16, 26, 16, 4 },
//	{ 1, 4, 7, 4, 1 }
//	};
//
//#define WEIGHTSUM 273
//	// Apply gaussian blurfilter		       
//	for (int i = 2; i < SCALED_SIZE - 2; i++)
//	{
//		for (int j = 2; j < SCALED_SIZE - 2; j++)
//		{
//			int sum = 0;
//			for (int k = -2; k < 3; k++)
//			{
//				for (int l = -2; l < 3; l++)
//				{
//					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
//				}
//			}
//			int value = sum / WEIGHTSUM;
//			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
//		}
//	}
}

