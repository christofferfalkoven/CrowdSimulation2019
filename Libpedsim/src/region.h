#pragma once
#include "ped_agent.h"

#ifndef _region_h_
#define _region_h_ 1

#include <vector>
#include <deque>
#include <list>
#include <tuple>

using namespace std;

namespace Ped {


	class region {

	public:

		// Creating a region vector containnig of agents

		// Might change to containing lists
		// Should it be vectors of lists? 
		//std::vector<std::list<Tagent*>> reg; 
		int reg1size = 0;
		int reg2size = 0;
		int reg3size = 0;
		int reg4size = 0;
		
		std::vector<Tagent*> region1;
		std::vector<Tagent*> region2;
		std::vector<Tagent*> region3;
		std::vector<Tagent*> region4;
		
		std::vector<tuple<Tagent*, int>> region1ToRemove;
		std::vector<tuple<Tagent*, int>> region2ToRemove;
		std::vector<tuple<Tagent*, int>> region3ToRemove;
		std::vector<tuple<Tagent*, int>> region4ToRemove;

		/*
		std::vector<Tagent*> leftinbox2;
		std::vector<Tagent*> rightinbox2;

		std::vector<Tagent*> leftinbox3;
		std::vector<Tagent*> rightinbox3;

		std::vector<Tagent*> leftinbox4;

		std::vector<Tagent*> rightoutbox1;
		std::vector<Tagent*> leftoutbox2;
		std::vector<Tagent*> rightoutbox2;
		std::vector<Tagent*> leftoutbox3;
		std::vector<Tagent*> rightoutbox3;
		std::vector<Tagent*> leftoutbox4;
		*/
		//int *reg1;
		//int *reg2;
		//int *reg3;
		//int *reg3;
		//int *reg4;
		// Region1's left neighbor will be NULL, its right will be region2
		// Region2's left neighbor will be region1, it right will be region3
		// Region3's left neighbor will be region2, it right will be region4
		// Region4's left neighbor will be region3, its right will be NULL 
		//Ped::Tagent* leftNeighbor; 
		//Ped::Tagent* rightNeighbor; 

		// Should implment a xMax for each region
		//int xMin; 
		//int xMax; 


		region(std::vector<Tagent*> agents);

	private:
		void Ped::region::init(std::vector<Tagent*> agents);
	};
}



#endif