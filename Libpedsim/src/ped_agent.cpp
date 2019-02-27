//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
//#include <ia64intrin.h>
//#include <ia64regs.h>
#include <pmmintrin.h>
#include <smmintrin.h>
// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

Ped::Tagent::Tagent(float posX, float posY) {
	Ped::Tagent::init(posX, posY);
}



void Ped::Tagent::init(float posX, float posY) {
	// NEW NEW NEW; CHANGED THESE TO POINTERS
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;
	pointerX = NULL;
	pointerY = NULL;
}


void Ped::Tagent::computeNextDesiredPosition() {
	
	destination = getNextDestination();
	if (destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination->getx() - getX();
	double diffY = destination->gety() - getY();
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int)round(x + diffX / len);
	desiredPositionY = (int)round(y + diffY / len);
}
void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		double diffX = destination->getx() - getX();
		double diffY = destination->gety() - getY();
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

	
	/*Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;
	/*
	if (usesimd) {
		nvänd pointe rx;
	} anars
		användx
		
	if (destination != NULL) {
		// compute if agent reached its current destination
		// NEW NEW NEW POINTERS 
		double diffX = destination->getx() - *pointerX;
		double diffY = destination->gety() - *pointerY;
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

	return nextDestination;*/
}
