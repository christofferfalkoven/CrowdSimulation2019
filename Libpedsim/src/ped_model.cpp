//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_agent.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <mmintrin.h>
//#include pedagent
#include <xmmintrin.h>
#include <emmintrin.h>
//#include "ped_model.h"
//#include <ia64intrin.h>
//#include <ia64regs.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <string>
#include "region.h"
// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	size_t agent_size = agents.size();
	
	
	
	
	//float *agent_x = (float*)malloc((agents.size() * sizeof(float) + 3) / 4);
	//float *agent_y = (float*)malloc((agents.size() * sizeof(float) + 3) / 4);

	//Ped::cool_agent *a = new Ped::cool_agent(agents);
	if (implementation == VECTOR) { 
		std::vector<Tagent *> b = getAgents();
		a = new Ped::cool_agent(b); 
	}
	
	if (implementation == OMP) {
		//std::vector<Tagent*> reg = new Ped::region(agents);
		// get all the agents and save i vector
		std::vector<Tagent *> b = getAgents(); 
		// initialize all the regions
		reg = new Ped::region(b);

		//cout << reg << endl;
	}
	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}



/*
#pragma omp section
{
if (!reg->leftinbox2.empty()) {
/*for (int i = 0; i < reg->leftinbox2.size(); i++) {
//->region1.push_back(reg->rightinbox1[i]);
//cout << "yo" << endl;
reg->region2.push_back(reg->leftinbox2[i]);
}
reg->region2.insert(reg->region2.end(), reg->leftinbox2.begin(), reg->leftinbox2.end());
}

if (!reg->rightinbox2.empty()) {
/*for (int i = 0; i < reg->rightinbox2.size(); i++) {
//->region1.push_back(reg->rightinbox1[i]);
//cout << "yo" << endl;
reg->region2.push_back(reg->rightinbox2[i]);
}

reg->region2.insert(reg->region2.end(), reg->rightinbox2.begin(), reg->rightinbox2.end());
}

if (!reg->leftoutbox2.empty()) {
for (int i = 0; i < reg->leftoutbox2.size(); i++) {
//std::vector<int>::iterator position = find(reg->region1.begin(), reg->region1.end(), reg->rightoutbox1[i]);
reg->region2.erase(remove(reg->region2.begin(), reg->region2.end(), reg->leftoutbox2[i]), reg->region2.end());
}
}

if (!reg->rightoutbox2.empty()) {
for (int i = 0; i < reg->rightoutbox2.size(); i++) {
//std::vector<int>::iterator position = find(reg->region1.begin(), reg->region1.end(), reg->rightoutbox1[i]);
reg->region2.erase(remove(reg->region2.begin(), reg->region2.end(), reg->rightoutbox2[i]), reg->region2.end());
}
}


}


#pragma omp section
{
if (!reg->leftinbox3.empty()) {
/*for (int i = 0; i < reg->leftinbox3.size(); i++) {
//->region1.push_back(reg->rightinbox1[i]);
//cout << "yo" << endl;
reg->region3.push_back(reg->leftinbox3[i]);
}
reg->region3.insert(reg->region3.end(), reg->leftinbox3.begin(), reg->leftinbox3.end());
}

if (!reg->rightinbox3.empty()) {
/*for (int i = 0; i < reg->rightinbox3.size(); i++) {
//->region1.push_back(reg->rightinbox1[i]);
//cout << "yo" << endl;
reg->region3.push_back(reg->rightinbox3[i]);
}
reg->region3.insert(reg->region3.end(), reg->rightinbox3.begin(), reg->rightinbox3.end());
}

if (!reg->leftoutbox3.empty()) {
for (int i = 0; i < reg->leftoutbox3.size(); i++) {
//std::vector<int>::iterator position = find(reg->region1.begin(), reg->region1.end(), reg->rightoutbox1[i]);
reg->region3.erase(remove(reg->region3.begin(), reg->region3.end(), reg->leftoutbox3[i]), reg->region3.end());
}
}

if (!reg->rightoutbox3.empty()) {
for (int i = 0; i < reg->rightoutbox3.size(); i++) {
//std::vector<int>::iterator position = find(reg->region1.begin(), reg->region1.end(), reg->rightoutbox1[i]);
reg->region3.erase(remove(reg->region3.begin(), reg->region3.end(), reg->rightoutbox3[i]), reg->region3.end());
}
}


}

#pragma omp section
{
if (!reg->leftinbox4.empty()) {
/*for (int i = 0; i < reg->leftinbox4.size(); i++) {
//->region1.push_back(reg->rightinbox1[i]);
//cout << "yo" << endl;
reg->region4.push_back(reg->leftinbox4[i]);
}
reg->region4.insert(reg->region4.end(), reg->leftinbox4.begin(), reg->leftinbox4.end());
}

if (!reg->leftoutbox4.empty()) {
for (int i = 0; i < reg->leftoutbox4.size(); i++) {
//std::vector<int>::iterator position = find(reg->region1.begin(), reg->region1.end(), reg->rightoutbox1[i]);
reg->region4.erase(remove(reg->region4.begin(), reg->region4.end(), reg->leftoutbox4[i]), reg->region4.end());
}
}
}
}*/


//tick_move_func(agents[i]);


// OR 
// Different implementation using only omp parallel command
/*
#pragma omp parallel
{
omp_set_num_threads(4);
int num_thread = omp_get_num_threads();
size_t size = agents.size();
int thread_id = omp_get_thread_num();

// Starts at diff places, but indexes i based on num of threads
for (int i = thread_id; i < size; i += num_thread) {
tick_one(agents[i]);
}
}
*/




void Ped::Model::tick()
{

	// OMP 
	if (implementation == Ped::OMP) {

		// Set number of threads 
		omp_set_num_threads(4);
		size_t size = agents.size();
		cout << "yo" << endl;
		for (int i = 0; i < size; i++) {
			#pragma omp parallel sections
			{

				// Commented this function out since we are going to be using the movefunction tick for lab3
				//tick_one(agents[i]);

				#pragma omp section 
				{
					
					if (!reg->rightinbox1.empty()) {
						//reg->region1.insert(reg->region1.end(), reg->rightinbox1.begin(), reg->rightinbox1.end());
						//cout << "size is: " << reg->rightinbox1.size() << endl;

						for (int i = 0; i < reg->rightinbox1.size(); i++) {
							reg->region1.push_back(reg->rightinbox1[i]);
							//reg->reg1size++;
						}
						//reg->reg1size = reg->reg1size+ ;
						reg->rightinbox1.clear();
						
					}

				}
				
				#pragma omp section 
				{
					if (!reg->leftinbox2.empty()) {
						//reg->region2.insert(reg->region2.end(), reg->leftinbox2.begin(), reg->leftinbox2.end());
						//reg->reg2size++;
						for (int i = 0; i < reg->leftinbox2.size(); i++) {
							reg->region2.push_back(reg->leftinbox2[i]);
							//reg->reg1size++;
						}
						reg->leftinbox2.clear();
					}
					if (!reg->rightinbox2.empty()) {
						//reg->region2.insert(reg->region2.end(), reg->rightinbox2.begin(), reg->rightinbox2.end());
						//reg->reg2size++;
						for (int i = 0; i < reg->rightinbox2.size(); i++) {
							reg->region2.push_back(reg->rightinbox2[i]);
							//reg->reg1size++;
						}
						reg->rightinbox2.clear();
					}
				}
				
				#pragma omp section 
				{
					if (!reg->leftinbox3.empty()) {
						//reg->region3.insert(reg->region3.end(), reg->leftinbox3.begin(), reg->leftinbox3.end());
						//reg->reg3size++;
						for (int i = 0; i < reg->leftinbox3.size(); i++) {
							reg->region3.push_back(reg->leftinbox3[i]);
							//reg->reg1size++;
						}
						reg->leftinbox3.clear();
					}
					if (!reg->rightinbox3.empty()) {
						//reg->region3.insert(reg->region3.end(), reg->rightinbox3.begin(), reg->rightinbox3.end());
						//reg->reg3size++;
						for (int i = 0; i < reg->rightinbox3.size(); i++) {
							reg->region3.push_back(reg->rightinbox3[i]);
							//reg->reg1size++;
						}
						reg->rightinbox3.clear();
					}
				}

				
				#pragma omp section 
				{
					if (!reg->leftinbox4.empty()) {
						//reg->region4.insert(reg->region4.end(), reg->leftinbox4.begin(), reg->leftinbox4.end());
						//reg->reg4size++;
						for (int i = 0; i < reg->leftinbox4.size(); i++) {
							reg->region4.push_back(reg->leftinbox4[i]);
							//reg->reg1size++;
						}
						reg->leftinbox4.clear();
					}
				}
			}

		tick_move_func(agents[i], i);
		}
	}


	// PTHREAD 
	else if (implementation == Ped::PTHREAD) {
		
		const int num_of_threads = 6;
		std::thread t[num_of_threads];
		int agents_size = agents.size(); 
		int thread_size = (agents_size / num_of_threads);
		//int start = 0; 
		//int end = thread_size; 
		for (int id = 0; id < num_of_threads; id++) {
			t[id] = std::thread(&Model::tick_many, this, id, num_of_threads);
		}
		for (int i = 0; i < num_of_threads; i++) {
			t[i].join();
		}
		
	}
	
	// SEQ
	else if (implementation == Ped::SEQ) {
		for (Ped::Tagent* agent : agents) {
			
			tick_one(agent);
		}
		
	}

	else if (implementation == Ped::VECTOR) {
		
		tick_vec();
		
	}
	else {
		std::cout << "No implementation chosen" << endl; 
	}
}



void Ped::Model::tick_many(int id, int thread_size)
{
	size_t agent_size = agents.size();
	for (int i = id; i < agent_size; i += thread_size) {
		tick_one(agents[i]);
	}
}

//Ped::cool_agent* agent
void Ped::Model::tick_vec()
{
	
	//agent->computeNextDesiredPosition();
	
	size_t agent_size = agents.size();
	//Ped::cool_agent *cool_agent = Ped::cool_agent::cool_agent(x, y);

	//cout << agent_size << endl;
	//__m128 X, Y, DSTX, DSTY, DIFFX, DIFFY, MULY, MULX, ADDOFMULDIFFXY, SQRTXY, DIFFXLEN, DIFFYLEN, XDIFFXLEN, YDIFFYLEN; 
	__m128 X, Y, DSTX, DSTY, DIFFX, DIFFY, LST;
	for (int i = 0; i < agent_size; i += 4) {
		if ((agent_size - i) < 4 == false) {
			for (int j = i; j < i + 4; j++) {
				bool agentReachedDestination = false;
				if (agents[j]->destination != NULL) {
					// compute if agent reached its current destination
					// NEW NEW NEW POINTERS 
					double diffX = agents[j]->destination->getx() - a->x[j];
					double diffY = agents[j]->destination->gety() - a->y[j];
					double length = sqrt(diffX * diffX + diffY * diffY);
					agentReachedDestination = length < agents[j]->destination->getr();
				}

				if ((agentReachedDestination || agents[j]->destination == NULL) && !agents[j]->waypoints.empty()) {
					// Case 1: agent has reached destination (or has no current destination);
					// get next destination if available
					agents[j]->waypoints.push_back(agents[j]->destination);
					Twaypoint *nextDestination = agents[j]->waypoints.front();
					agents[j]->waypoints.pop_front();

					if (nextDestination) {
						//int desX = agent->getDesiredX();

						a->desiredPositionX[j] = (float)nextDestination->getx();
						a->desiredPositionY[j] = (float)nextDestination->gety();
					}

					agents[j]->destination = nextDestination;
				}
			}



			X = _mm_load_ps(&a->x[i]);
			DSTX = _mm_load_ps(&a->desiredPositionX[i]);

			Y = _mm_load_ps(&a->y[i]);
			DSTY = _mm_load_ps(&a->desiredPositionY[i]);

			// Subtract of dstx and x
			DIFFX = _mm_sub_ps(DSTX, X);
			DIFFY = _mm_sub_ps(DSTY, Y);

			// DiffX
			DSTX = _mm_mul_ps(DIFFX, DIFFX);
			// DiffY
			DSTY = _mm_mul_ps(DIFFY, DIFFY);

			// add the DIFFX*DIFFX + DIFFY*DIFFY
			LST = _mm_add_ps(DSTX, DSTY);

			// Take square root of add of DIFFX*DIFFX + DIFFY*DIFFY
			// the len var
			LST = _mm_sqrt_ps(LST);
			//diffX / len
			DIFFX = _mm_div_ps(DIFFX, LST);
			// diffY / len
			DIFFY = _mm_div_ps(DIFFY, LST);

			// now we add the x and y with diffx + len osv   x + diffX / len
			DSTX = _mm_add_ps(X, DIFFX);
			DSTY = _mm_add_ps(Y, DIFFY);

			// Now we round
			DSTX = _mm_round_ps(DSTX, _MM_FROUND_TO_NEAREST_INT);
			DSTY = _mm_round_ps(DSTY, _MM_FROUND_TO_NEAREST_INT);

			_mm_store_ps(&a->x[i], DSTX);
			_mm_store_ps(&a->y[i], DSTY);
		}
		else {
			// gör sekventiellt
			for (int g = i; g < agent_size; g++) {
				agents[g]->computeNextDesiredPosition();
				int x = agents[g]->getDesiredX();
				int y = agents[g]->getDesiredY();
				a->x[g] = x;
				a->y[g] = y;
			}
			
		}
	}
	
}

// This function will be using only the move function logic, we will choose to apply the lab3 using OPENMP
void Ped::Model::tick_move_func(Ped::Tagent* agent, int i)
{
	if (implementation == OMP) {

		/*size_t agent_size = agent.size();
		for (int i = 0; i < agent.size(); i++) {
			agent[i]->computeNextDesiredPosition();
			int x = agent[i]->getX();
			int dstX = agent[i]->getDesiredX();
			//cout << x << endl;
			// going from region1 to region2
			if (x == 199 && dstX > 200) {
				cout << "hej1" << endl;

				// Push the current agent to region1's right outbox, and then after send it to region2 inbox
				reg->rightoutbox1.push_back(agent[i]);
				// Push current agent to region2's leftinbox
				reg->leftinbox2.push_back(agent[i]);
				// If region2 has mail 
				//if (!reg->leftinbox2.empty()) {
				//	reg->region2.push_back(reg->leftinbox2);
				//}
			}
			// going from region2 to region1
			else if (x == 201 && dstX < 200) {
				cout << "hej2" << endl;

				reg->leftoutbox2.push_back(agent[i]);
				reg->rightinbox1.push_back(agent[i]);

			}
			// going from region2 to region3
			else if (x == 399 && dstX > 400) {
				cout << "hej3" << endl;

				reg->rightoutbox2.push_back(agent[i]);
				reg->leftinbox3.push_back(agent[i]);

			}
			// going from region3 to region2
			else if (x == 401 && dstX < 400) {
				cout << "hej4" << endl;

				reg->leftoutbox3.push_back(agent[i]);
				reg->rightinbox2.push_back(agent[i]);

			}
			// going from region3 to region4
			else if (x == 599 && dstX > 600) {
				cout << "hej5" << endl;

				reg->rightoutbox3.push_back(agent[i]);
				reg->leftinbox4.push_back(agent[i]);

			}
			// going from region4 to region 3
			else if (x == 601 && dstX < 600) {
				cout << "hej6" << endl;

				reg->leftoutbox4.push_back(agent[i]);
				reg->rightinbox3.push_back(agent[i]);

			}


			move(agent[i]);
			//cout << reg->rightinbox1[i]->getX() << endl; 
			//printf(" %p\n", reg->leftinbox2[i].getX());
			//printf(reg->leftinbox2[agent[i]]);
			//cout << reg->leftinbox2 << endl; 

		} */

		int one = reg->region1.size();
		int two = reg->region2.size();
		int three = reg->region3.size();
		int four = reg->region4.size();
		int tot = one + two + three + four;
		cout << "size reg1: " <<  one << "::::::::size reg2: " << two << "::::::::size reg3: " <<three << "::::::::size reg4: " << four<< "::::::::Total: " << tot  << endl;
		
		agent->computeNextDesiredPosition();
		int x = agent->getX();
		int dstX = agent->getDesiredX();
		if (tot != 452) {
			cout << "get x: " << x << " :: dest: " <<dstX << endl;
			cout << "len before: " << reg->region1.size() << endl;
			reg->region2.erase(remove(reg->region1.begin(), reg->region1.end(), agent), reg->region2.end());
			cout << "len after: " << reg->region1.size() << endl;

			exit(0);
		}
		// going from region1 to region2
		if (x == 39 && dstX == 40) {
			// Push the current agent to region1's right outb		ox, and then after send it to region2 inbox
			//reg->rightoutbox1.push_back(agent);

			// Push current agent to region2's leftinbox
			reg->leftinbox2.push_back(agent);

			//remove it from region 1
			//här va felet
			//reg->region1.erase(reg->region1.begin() + i);
			reg->region1.erase(remove(reg->region1.begin(), reg->region1.end(), agent), reg->region1.end());
			//reg->reg1size--;
			//cout << reg->reg1size << endl;

			//cout << "the distance is: " << i<<endl;
			
		}
		// going from region2 to region1
		else if (x == 40 && dstX == 39) {
			//cout << "hej2" << endl;

			//reg->leftoutbox2.push_back(agent);
			reg->rightinbox1.push_back(agent);

			reg->region2.erase(remove(reg->region2.begin(), reg->region2.end(), agent), reg->region2.end());


			//reg->region2.erase(reg->region2.begin() + i);

		}
		// going from region2 to region3. from 80 to 81
		else if (x == 79 && dstX == 80) {
			//cout << "hej3" << endl;

			//reg->rightoutbox2.push_back(agent);
			reg->leftinbox3.push_back(agent); //
			reg->region2.erase(remove(reg->region2.begin(), reg->region2.end(), agent), reg->region2.end());

			//reg->region2.erase(reg->region2.begin() + i);

		}
		// going from region3 to region2
		else if (x == 80 && dstX == 79) {
			//cout << "hej4" << endl;

			//reg->leftoutbox3.push_back(agent);
			reg->rightinbox2.push_back(agent);
			reg->region3.erase(remove(reg->region3.begin(), reg->region3.end(), agent), reg->region3.end());

			//reg->region3.erase(reg->region3.begin() + i);

		}
		// going from region3 to region4
		else if (x == 119 && dstX == 120) {
			//cout << "hej5" << endl;

			//reg->rightoutbox3.push_back(agent);
			reg->leftinbox4.push_back(agent);

			reg->region3.erase(remove(reg->region3.begin(), reg->region3.end(), agent), reg->region3.end());

			//reg->region3.erase(reg->region3.begin() + i);

		}
		// going from region4 to region 3
		else if (x == 120 && dstX == 119) {
			//cout << "hej6" << endl;

			//reg->leftoutbox4.push_back(agent);
			reg->rightinbox3.push_back(agent);

			reg->region4.erase(remove(reg->region4.begin(), reg->region4.end(), agent), reg->region4.end());

			//reg->region4.erase(reg->region4.begin() + i);

		}


		move(agent);
	}
	
}


// Potential source of parallellism
void Ped::Model::tick_one(Ped::Tagent* agent)
{
	//agent->computeNextDesiredPosition();
	//move(agent);
	// get the desired X coord
	int x = agent->getDesiredX();
	
	// get the desired Y coord
	int y = agent->getDesiredY();

	// set the agent x-value to the desired x-value
	agent->setX(x);

	//int x = agent->getDesiredX(); 
	//agent->setPointerX(x)
	//agent->setX(&x);
	/*
		x[i] = agents[i]->getX();
		y[i] = agents[i]->getY();
		agents[i]->setPointerX(&x[i]);
		agents[i]->setPointerY(&y[i]);
	*/
	

	
	// set the agent y-value to the desired y-value
	agent->setY(y);
	
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list 
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
