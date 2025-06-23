#ifndef SHORTEST_PATH_H_
#define SHORTEST_PATH_H_

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "../adt/ALGraph.h"
#include "../adt/CircQueue.h"
#include <math.h>
#include "../adt/MinHeap.h"
#include "../adt/MGraph.h"

int* ShortestPathTopeDownDivideDFSWithoutRepetition(ALGraph *G, int src);
int* ShortestPathDownTopDivideBFSDynamicProgramWithouRepetition(ALGraph *G, int src);
int* ShortestPathDownTopDivideTaskDependencyTopologicalSort(ALGraph *G, int src);
int* ShortestPathBellmanFord(ALGraph* G, int src);
int* ShortestPathHeuristicGreedyDijkstraRecord(ALGraph* G, int src);
int* ShortestPathHeuristicGreedyDijkstraMinHeap(ALGraph* G, int src);
int ShortestPathFloydDynamicProgram(MGraph* G);

#endif