//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <iostream>

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(float posX, float posY);
		//Tagent(double *posX, double *posY);

		// Returns the coordinates of the desired position

		// NEW NEW NEW
		// change these to pointers that point to memory where we have our x values
		int getDesiredX() const { 
			//cout << "Get des X" << endl;
			return desiredPositionX; 
		}
		// pointer to mem where we have our y values
		int getDesiredY() const { 
			//cout << "get des Y" << endl;
			return desiredPositionY; 
		}

		// Sets the agent's position

		// NEW: Change this to pointers as well, we repoint our x pointer to the new computed one
		void setX(int newX) { x = (float)newX; }
		void setY(int newY) { y = (float)newY; }

		void setPointerX(float *newX) { 

			pointerX = newX; 
		}
		void setPointerY(float *newY) { 
			pointerY = newY; 
		}
		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();


		//bool xd = false; 
		//void Ped::Tagent::computeNextDesiredPositionVec();
		// NEW NEW NEW
		// Position of agent defined by x and y
		// RETURN POINTER TO X AND Y 
		int getX() const { 
			if (pointerX) {
				//cout << "HEHEH" << endl;
				//return *pointerX;
				return (int)round(*pointerX);
			//	return (int)*pointerX;
				
			}
			//if (!xd) { return x; }
			//return *pointerX;
			return x;
			//return x;
		};
		int getY() const { 
			if (pointerY) {
			//	return (int)*pointerY;
				//cout << "POINTER Y " << endl;
				//cout << (int)round(*pointerY) << endl;
				//return *pointerY;
				return (int)round(*pointerY);
			}
			//cout << "Y  " << endl;
			//cout << (int)round(y) << endl;
			return y;
			//return y;
			//if (!xd) { return y; }
			//return *pointerY;
		};

		float *pointerX;
		float *pointerY;
		Twaypoint* destination;
		deque<Twaypoint*> waypoints;
		int desiredPositionX;
		int desiredPositionY;
		
		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);
	private:
		Tagent() {};

		// The agent's current position
		// NEW NEW NEW
		// we need to use pointers to use the mm functions since they expect to take in pointers to memory where the x and y values are stored
		// this is the whole structure of arrays tänk
		float x;
		float y;

		// The agent's desired next position
		// NEW NEW NEW NEW 
		// USE POINTERS TO VEC
		

		// The current destination (may require several steps to reach)
		
		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		

		// Internal init function 
		void init(float posX, float posY);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();
	};
}

#endif