#include "cool_agent.h"
#include "ped_waypoint.h"
#include "ped_agent.h"
#include <math.h>
#include <mmintrin.h>
//#include pedagent
#include <xmmintrin.h>
#include <emmintrin.h>
//#include "ped_model.h"
//#include <ia64intrin.h>
//#include <ia64regs.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <vector>
// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

Ped::cool_agent::cool_agent(std::vector<Tagent*> agents) {
	Ped::cool_agent::init(agents);
}

void Ped::cool_agent::init(std::vector<Tagent*> agents)
{
	
	
	x = (float*)_mm_malloc(agents.size() * sizeof(float), 16);
	y = (float*)_mm_malloc(agents.size() * sizeof(float), 16);
	desiredPositionX = (float*)_mm_malloc(agents.size() * sizeof(float), 16);
	desiredPositionY = (float*)_mm_malloc(agents.size() * sizeof(float), 16);


	for (int i = 0; i < agents.size(); i++) {
		
		x[i] = agents[i]->getX();
		//cout << x[i] << endl;
		y[i] = agents[i]->getY();
		//cout << y[i] << endl;
		agents[i]->setPointerX(&x[i]);
		//cout << agents[i]->getX() << endl;
		agents[i]->setPointerY(&y[i]);
		//cout << agents[i]->getY() << endl;
		//agents[i]->xd = true; 
	}

	destination = NULL; 
	lastDestination = NULL;
}




/*

void Ped::cool_agent::init(float *posX, float *posY) {
	// NEW NEW NEW; CHANGED THESE TO POINTERS
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;
}
*/
/*
void Ped::cool_agent::computeNextDesiredPosition() {
	Ped::Twaypoint *destination = getNextDestination();
	if (destination == NULL) return;


	/*
		double diffX = destination->getx() - *x;
	double diffY = destination->gety() - *y;
	double len = sqrt(diffX * diffX + diffY * diffY);
	// NEW NEW NEW, all desired and x and y osv that are pointers now, were not before
	desiredPositionX = (int)round(x + diffX / len);
	desiredPositionY = (int)round(y + diffY / len);




	for i to whatever, i+4
	ladda posvect->x[i] i _mm_register 1
	ladda pos->nxtx{i} i mm_register
	plussa mm_rgegstetrna
	spara mm_reg 1 i posvect->x[i]
	

	__m128 A, B, C, D, E, F, G; 
	size_t agent_size = agents.size(); 
	//Ped::cool_agent *cool_agent = Ped::cool_agent::cool_agent(x, y);

	__m128 diffX = _mm_sub_ps(_mm_cvtepi32_ps(_mm_cvtsi32_si128(destination->getx())), _mm_cvtepi32_ps(_mm_cvtsi32_si128(*x)));

	__m128 diffY = _mm_sub_ps(_mm_cvtepi32_ps(_mm_cvtsi32_si128(destination->gety())), _mm_cvtepi32_ps(_mm_cvtsi32_si128(*y)));

	__m128 diffXMul = _mm_mul_ss(diffX, diffX);
	__m128 diffYMul = _mm_mul_ss(diffY, diffY);

	__m128 diffAdd = _mm_add_ps(diffXMul, diffYMul);

	__m128 len = _mm_cvtpd_ps(_mm_sqrt_pd(_mm_cvtps_pd(diffAdd)));

	// Not rounded
	__m128 desiredPositionX = (_mm_add_ss(_mm_cvtepi32_ps(_mm_cvtsi32_si128(*x)), (_mm_div_ss(diffX, len))));
	__m128 desiredPositionY = (_mm_add_ss(_mm_cvtepi32_ps(_mm_cvtsi32_si128(*y)), (_mm_div_ss(diffY, len))));

}
*/
void Ped::cool_agent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::cool_agent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		// NEW NEW NEW POINTERS 
		double diffX = destination->getx() - *x;
		double diffY = destination->gety() - *y;
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < destination->getr();
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	return nextDestination;
}
