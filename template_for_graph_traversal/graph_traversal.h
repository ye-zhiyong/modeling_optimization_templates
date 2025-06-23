#ifndef GRAPH_TRAVERSAL_H_
#define GRAPH_TRAVERSAL_H_

#include <stdio.h>
#include <stdlib.h>
#include "../adt/ALGraph.h"
#include "../adt/CircQueue.h"

void PrintGraphPath(int prev[], int src, int dest);
int* SearchPathALGraphCircQueueBFS(ALGraph* G, int src, int dest);
int* SearchPathALGraphDFSRecursiveWithoutRepetition(ALGraph* G, int src, int dest);

#endif