#ifndef TOPO_SORT_H_
#define TOPO_SORT_H_

#include <stdio.h>
#include <stdlib.h>
#include "../adt/ALGraph.h"

void TopoSortIndegree(ALGraph *G);
void TopoSortInverseALGraph(ALGraph *G);

#endif