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
#include <tuple>
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
	
	//if (implementation == OMP) {
		//std::vector<Tagent*> reg = new Ped::region(agents);
		// get all the agents and save i vector
		std::vector<Tagent *> b = getAgents(); 
		// initialize all the regions
		reg = new Ped::region(b);

		//cout << reg << endl;
	//}
	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapCuda();
	omp_set_num_threads(4);
}

void Ped::Model::reg_func() {
	size_t size = agents.size();
	int i = 0;

	#pragma omp parallel sections 
	{
		#pragma omp section 
		{
			tick_move_func_reg1();
			if (!reg->region1ToRemove.empty()) {
				for (int n = 0; n < reg->region1ToRemove.size(); n++) {
					reg->region1.erase(reg->region1.begin() + (get<1>(reg->region1ToRemove[n]) - n));
				}
				reg->region1ToRemove.clear();
			}


		}

		#pragma omp section 
		{
			tick_move_func_reg2();
			if (!reg->region2ToRemove.empty()) {
				for (int n = 0; n < reg->region2ToRemove.size(); n++) {
					reg->region2.erase(reg->region2.begin() + (get<1>(reg->region2ToRemove[n]) - n));
				}
				reg->region2ToRemove.clear();
			}
		}

		#pragma omp section 
		{
			tick_move_func_reg3();
			if (!reg->region3ToRemove.empty()) {

				for (int n = 0; n < reg->region3ToRemove.size(); n++) {
					reg->region3.erase(reg->region3.begin() + (get<1>(reg->region3ToRemove[n]) - n));
				}
				reg->region3ToRemove.clear();
			}
		}

		#pragma omp section 
		{
			tick_move_func_reg4();
			if (!reg->region4ToRemove.empty()) {
				for (int n = 0; n < reg->region4ToRemove.size(); n++) {
					//cout << get<1>(reg->region1ToRemove[n]) << endl;
					reg->region4.erase(reg->region4.begin() + (get<1>(reg->region4ToRemove[n]) - n));
				}
				reg->region4ToRemove.clear();
			}
		}
	}
}



void Ped::Model::tick()
{
	
	// OMP 
	if (implementation == Ped::OMP) {
		#pragma omp parallel
		{
			// parallell block, en tråd kör reg func och en tr¨åd kör update
			#pragma omp single nowait
			{
				reg_func();
			}
			#pragma omp single nowait
			{
				updateHeatmapCuda();
			}
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
		reg_func();
		updateHeatmapSeq();
		
		
		//tick_heat(); 
	/*	for (Ped::Tagent* agent : agents) {
			tick_one(agent);
		}*/
		
	}

	else if (implementation == Ped::VECTOR) {
		
		tick_vec();
		
	}
	else {
		std::cout << "No implementation chosen" << endl; 
	}
}



void Ped::Model::tick_heat()
{	

	for (Ped::Tagent* agent : agents) {

		agent->computeNextDesiredPosition();

		move(agent);
	}
	
	updateHeatmapCuda();
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

void Ped::Model::tick_move_func_reg1() {
	#pragma omp parallel for 
	for (int p = 0; p < reg->region1.size(); p++) {

		Tagent* agent = reg->region1[p];
		agent->computeNextDesiredPosition();
		int x = agent->getX();
		int dstX = agent->getDesiredX();

		// going from region 1 to 2
		if (x == 40 && dstX == 41) {
			reg->region2.push_back(agent);
			//reg->region1.erase(reg->region1.begin() + p);
			reg->region1ToRemove.push_back(std::make_tuple(agent, p));
			

		} 
		move(agent);
	}

	//exit(0);
}
void Ped::Model::tick_move_func_reg2() {
	#pragma omp parallel for 
	for (int p = 0; p < reg->region2.size(); p++) {

		Tagent* agent = reg->region2[p];
		agent->computeNextDesiredPosition();
		int x = agent->getX();
		int dstX = agent->getDesiredX();

		// going from region 2 to 1
		if (x == 41 && dstX == 40) {
			reg->region1.push_back(agent);
			reg->region2ToRemove.push_back(std::make_tuple(agent, p));

		}
		// going from region2 to region3
		else if (x == 80 && dstX == 81) {
			reg->region3.push_back(agent);
			reg->region2ToRemove.push_back(std::make_tuple(agent, p));
		}
		move(agent);
	}
}

void Ped::Model::tick_move_func_reg3() {
	#pragma omp parallel for 
	for (int p = 0; p < reg->region3.size(); p++) {

		Tagent* agent = reg->region3[p];
		agent->computeNextDesiredPosition();
		int x = agent->getX();
		int dstX = agent->getDesiredX();

		// going from region 3 to 2
		if (x == 81 && dstX == 80) {
			reg->region2.push_back(agent);
			reg->region3ToRemove.push_back(std::make_tuple(agent, p));

		}
		// going from region3 to region4
		else if (x == 120 && dstX == 121) {
			reg->region4.push_back(agent);
			reg->region3ToRemove.push_back(std::make_tuple(agent, p));
		}
		move(agent);
	}
}
 
void Ped::Model::tick_move_func_reg4() {
	#pragma omp parallel for 
	for (int p = 0; p < reg->region4.size(); p++) {

		Tagent* agent = reg->region4[p];
		agent->computeNextDesiredPosition();
		int x = agent->getX();
		int dstX = agent->getDesiredX();
		// going from region 4 to 3
		if (x == 121 && dstX == 120) {
			reg->region3.push_back(agent);
			reg->region4ToRemove.push_back(std::make_tuple(agent, p));
		}
		move(agent);
	}

}

// This function will be using only the move function logic, we will choose to apply the lab3 using OPENMP

void Ped::Model::tick_move_func(std::vector<Tagent*> spec_region, int i) {
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
	// if neg = go left, else go right
	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2, p3, p4, p5, p6, p7;
	//om den ej vill gå diagonalt
	//cout << reg->region1.size << endl;
	int getX = agent->getX();
	int getY = agent->getY();

	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);

		//cout << "x value: " << agent->getX() << " :: y value: " << agent->getY()<< " :: pDesired.first: " << (int)pDesired.first << " :: pDesired.second: " << (int)pDesired.second << " :: diffX: " << diffX << " :: diffY: " << diffY << " :: p1.first: " << p1.first << " :: p1.second: " << p1.second << endl;
		//cout << "x value: " << agent->getX() << " :: y value: " << agent->getY() << " :: p1.first: " << p2.first << " :: p2.second: " << p2.second << endl;

		//exit(0);
		// desX-getX()
		//  151-150 = 
		if (diffX < 0 ) {
			p3 = std::make_pair(getX, getY + 1);
			p4 = std::make_pair(getX, getY - 1);
			p5 = std::make_pair(getX + 1, getY + 1);
			p6 = std::make_pair(getX + 1, getY - 1);
			p7 = std::make_pair(getX + 1, getY);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		else if (diffX > 0 ) {
			p3 = std::make_pair(getX, getY + 1);
			p4 = std::make_pair(getX, getY - 1);
			p5 = std::make_pair(getX - 1, getY + 1);
			p6 = std::make_pair(getX - 1, getY - 1);
			p7 = std::make_pair(getX - 1, getY);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		else if (diffY < 0 ) {
			p3 = std::make_pair(getX - 1, getY);
			p4 = std::make_pair(getX - 1, getY + 1);
			p5 = std::make_pair(getX, getY + 1);
			p6 = std::make_pair(getX + 1, getY);
			p7 = std::make_pair(getX + 1, getY + 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);

		}
		else if (diffY > 0 ) {
			p3 = std::make_pair(getX + 1, getY);
			p4 = std::make_pair(getX + 1, getY - 1);
			p5 = std::make_pair(getX - 1, getY);
			p6 = std::make_pair(getX - 1, getY - 1);
			p7 = std::make_pair(getX, getY - 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
	
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
		
		//cout << "x value: " << agent->getX() << " :: y value: " << agent->getY() << " :: p1.first: " << p1.first << " :: p1.second: " << p1.second << endl;
		//cout << "x value: " << agent->getX() << " :: y value: " << agent->getY() << " :: p2.first: " << p2.first << " :: p2.second: " << p1.second << endl;
		//exit(0);
		if (diffX < 0 && diffY < 0) {
			p3 = std::make_pair(getX + 1, getY - 1);
			p4 = std::make_pair(getX + 1, getY + 1);
			p5 = std::make_pair(getX + 1, getY);
			p6 = std::make_pair(getX, getY + 1);
			p7 = std::make_pair(getX - 1, getY + 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		else if (diffX < 0 && diffY > 0) {
			p3 = std::make_pair(getX + 1, getY + 1);
			p4 = std::make_pair(getX + 1, getY - 1);
			p5 = std::make_pair(getX + 1, getY);
			p6 = std::make_pair(getX - 1, getY - 1);
			p7 = std::make_pair(getX, getY - 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		else if (diffX > 0 && diffY < 0) {
			p3 = std::make_pair(getX - 1, getY - 1);
			p4 = std::make_pair(getX - 1, getY + 1);
			p5 = std::make_pair(getX - 1, getY);
			p6 = std::make_pair(getX + 1, getY + 1);
			p7 = std::make_pair(getX, getY + 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		else if (diffX > 0 && diffY > 0) {
			p3 = std::make_pair(getX - 1, getY + 1);
			p4 = std::make_pair(getX - 1, getY - 1);
			p5 = std::make_pair(getX - 1, getY);
			p6 = std::make_pair(getX - 1, getY - 1);
			p7 = std::make_pair(getX, getY - 1);

			prioritizedAlternatives.push_back(p3);
			prioritizedAlternatives.push_back(p4);
			prioritizedAlternatives.push_back(p5);
			prioritizedAlternatives.push_back(p6);
			prioritizedAlternatives.push_back(p7);
		}
		prioritizedAlternatives.push_back(p1);
		prioritizedAlternatives.push_back(p2);
	}
	

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
