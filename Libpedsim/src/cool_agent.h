#pragma once
#include "ped_agent.h"

#ifndef _cool_agent_h_
#define _cool_agent_h_ 1

#include <vector>
#include <deque>


using namespace std;

namespace Ped {
	class Twaypoint;

	class cool_agent {
	public:
		//cool_agent(float *posX, float *posY);
		cool_agent(std::vector<Tagent*> agents);
		//int getDesiredX() const { return *desiredPositionX; }
		// pointer to mem where we have our y values
		//int getDesiredY() const { return *desiredPositionY; }


		//void setX(int newX) { *x = (float)newX; }
		//void setY(int newY) { *y = (float)newY; }


		//void computeNextDesiredPosition();


		//int getX() const { return *x; };
		//int getY() const { return *y; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);


		
		// The current destination (may require several steps to reach)
		Twaypoint* destination;

		float *x;
		float *y;

		float* desiredPositionX;
		float* desiredPositionY;


	private:
		cool_agent() {};



		std::vector<cool_agent*> agents;

	
		

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void Ped::cool_agent::init(std::vector<Tagent*> agents);


		// Returns the next destination to visit
		Ped::Twaypoint* getNextDestination();
	};

}
#endif