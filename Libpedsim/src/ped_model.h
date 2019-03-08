//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>

#include "ped_agent.h"
#include "region.h"
#include "cool_agent.h"
#include <stdio.h>
namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		void tick_one(Ped::Tagent* agent);

		void Ped::Model::tick_many(int start, int end);
		

		// This is the function that our new lab3 implementation of OMP will use
		//void Ped::Model::tick_move_func(std::vector<Tagent*> agent);
		//void Ped::Model::tick_move_func(Ped::Tagent* agent, int i);
		void Ped::Model::tick_move_func(std::vector<Tagent*> region, int i);
		void Ped::Model::tick_move_func_reg1();
		void Ped::Model::tick_move_func_reg2();
		void Ped::Model::tick_move_func_reg3();
		void Ped::Model::tick_move_func_reg4();


		// Tick func for heatmap
		void Ped::Model::tick_heat();


		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *ag);

		void Ped::Model::tick_vec();
		//Ped::cool_agent* agent 
		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;
	
	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The pointer to the regions
		Ped::region *reg;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		void Ped::Model::reg_func();
		Ped::cool_agent *a;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view

		int ** blurred_heatmap;

		int *heatmap_cuda;
		int *scaled_cuda;
		int *blurred_cuda;
		int *x_arr;
		int *y_arr;

		int *x_arr_host;
		int *y_arr_host;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
		void setupHeatmapCuda();
		void updateHeatmapCuda();
	};
}
#endif
