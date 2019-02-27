#include "region.h"
#include "ped_model.h"
#include "ped_agent.h"
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <list>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif



Ped::region::region(std::vector<Tagent*> agents) {
	Ped::region::init(agents);
}

void Ped::region::init(std::vector<Tagent*> agents) {

	// Get all the agents
	// Check their x values

	// iterate over all x values

	// if their x value is less than 40 then push those agents into region 1
	// if their x value is more than 40 but less than 80 push those agents into region 2 and so on

	//int agent_size = 1;
	

	for (int i = 0; i < agents.size(); i++) {
		int x = agents[i]->getX();
		/*
		agents[i]->setX(39);
		int x = agents[i]->getX();
		agents[i]->desiredPositionX = 40;
		*/

		if (x < 40) {
			region1.push_back(agents[i]);
		}
		else if (40 < x && x < 80) {
			region2.push_back(agents[i]);
		}
		else if (80 < x && x < 120) {
			region3.push_back(agents[i]);
		}
		else {
			region4.push_back(agents[i]);
		}

	}
	
	

	/*for (int i = 0; i < agents.size(); i++) {
		region1->leftNeighbor = NULL; 
		region1->rightNeighbor = region2; 
		region1->xMax = 200; 
		region1->xMin = 0; 
	}*/


}
