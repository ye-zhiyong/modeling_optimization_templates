#ifndef MIN_SPANNING_TREE_H__
#define MIN_SPANNING_TREE_H_

#include <stdio.h>
#include <stdlib.h>
#include "../adt/ALGraph.h"
#include "../adt/MinHeap.h"
#include "../adt/UFSet.h"
#include "../adt/CircQueue.h"

typedef struct EdgeNodeMST{
    int begin;
    int end;
    EdgeType weight;
}EdgeNodeMST;

int HeuristicSearchGreedyKruskal(ALGraph* G);

#endif