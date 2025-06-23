#ifndef CRITICAL_PATH_H_
#define CRITICAL_PATH_H_

#include <stdio.h>
#include <stdio.h>
#include "../adt/ALGraph.h"
#include "../adt/CircQueue.h"
#include "../adt/MinHeap.h"
#include "../adt/MaxHeap.h"

int* CriticalPathTopeDownDivideDFSWithoutRepetition(ALGraph *G, int src);
int* CriticalPathDownTopDivideBFSDynamicProgramWithouRepetition(ALGraph *G, int src);
int* CriticalPathDownTopDivideTaskDependencyTopologicalSort(ALGraph *G, int src);
int* CriticalPathHeuristicGreedyDijkstraRecord(ALGraph* G, int src);
int* CriticalPathHeuristicGreedyDijkstraMinHeap(ALGraph* G, int src);

#endif
