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
	cudaMalloc(&scaled_cuda, (SCALED_SIZE*SCALED_SIZE) * sizeof(int));
	cudaMalloc(&x_arr, agents.size() * sizeof(int));
	cudaMalloc(&y_arr, agents.size() * sizeof(int));
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
		heatmap_cuda[x + y * SIZE] += 40;
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

__global__ void kernel_func_5(int *heatmap_cuda, int *scaled_cuda, int *blurred_cuda) {
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int row = thread_id / SIZE;
	int col = thread_id % SIZE;
	int sum = 0; 
#define WEIGHTSUM 273
	for (int k = -2; k < 3; k++)
	{
		for (int l = -2; l < 3; l++)
		{
			sum += w[2 + k][2 + l] * scaled_cuda[(row * CELLSIZE + k)*SCALED_SIZE + (col * CELLSIZE + l)];//scaled_heatmap[SCALED_SIZE - 2 + k][SCALED_SIZE - 2 + l];
		}
		int value = sum / WEIGHTSUM;
		blurred_cuda[SCALED_SIZE * SCALED_SIZE] = 0x00FF0000 | value << 24;
	}
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
	cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	kernel_func_1<<<SIZE, SIZE >>>(heatmap_cuda);
	cudaDeviceSynchronize();


	// For kernel func 2, agents 
	// Allocate arrs for taking care of this
	int *x_arr_host = (int *)malloc(agents.size() * sizeof(int));
	int *y_arr_host = (int *)malloc(agents.size() * sizeof(int));
	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++) {
		Ped::Tagent* agent = agents[i];
		x_arr_host[i] = agent->getDesiredX();
		y_arr_host[i] = agent->getDesiredY();
	}

	cudaMemcpy(x_arr, x_arr_host, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(y_arr, y_arr_host, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
	// Ett block med agents.size() antal trådar
	kernel_func_2<<<1, agents.size()>>>(heatmap_cuda, x_arr, y_arr);
	cudaDeviceSynchronize();
	cudaMemcpy(*heatmap, heatmap_cuda, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	// End

	// For kernel func 3
	cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	kernel_func_3<<<SIZE, SIZE >>>(heatmap_cuda);
	cudaDeviceSynchronize();

	// For kernel func 4
	cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_cuda, *scaled_heatmap, CELLSIZE * CELLSIZE * sizeof(int), cudaMemcpyHostToDevice);
	kernel_func_4<<<SIZE, SIZE >>>(heatmap_cuda, scaled_cuda);
	cudaDeviceSynchronize();
	cudaMemcpy(*scaled_heatmap, scaled_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// For kernel func 5
	cudaMemcpy(heatmap_cuda, *heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_cuda, *scaled_heatmap, CELLSIZE * CELLSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(blurred_cuda, *blurred_heatmap, CELLSIZE * CELLSIZE * sizeof(int), cudaMemcpyHostToDevice);
	kernel_func_5 <<<SIZE, SIZE >>>(heatmap_cuda, scaled_cuda, blurred_cuda);
	cudaMemcpy(*scaled_heatmap, scaled_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*blurred_heatmap, blurred_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Scale the data for visual representation
	/*for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}*/
	/*
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};

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

