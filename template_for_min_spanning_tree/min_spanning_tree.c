#include "min-spanning-tree.h"

/**
 * 1. 启发式搜索，基于贪心算法，使用小根堆MinHeap获取最小边，使用并查集Unio-Find Set查询顶点是否连通/成环。（Kruskal's Algorithm）
 *    时间复杂度O(max{|V|+|E|, |E|log|E|})。执行时间测试40us。   
 * 
 * 2. 启发式搜索，基于贪心算法，使用小根堆MinHeap获取最小边，使用并查集Union-Find Set查询顶点是否连通/成环。(Primer‘s Algorithm)
 *    时间复杂度O(max{|V|+|E|, |E|log|E|})
 */

int HeuristicSearchGreedyKruskal(ALGraph *G){
    EdgeNodeMST edgeNode[maxEdges];
    for(int i = 0; i < G->numVertices; i++){
        EdgeNode* p = G->vertexList[i].next;
        while(p != NULL){
            edgeNode[i].begin = i;
            edgeNode[i].end = p->destVertex;
            edgeNode[i].weight = p->edge;
            p = p->next;
        }
    }
    int set[maxVertices];
    InitUFSet(set, maxVertices);
    MinHeap minH;
    InitMinHeap(&minH);
    for(int j = 0; j < G->numEdges; j++){
        EnMinHeap(&minH, edgeNode[j].weight, j);
    }
    int index;
    while(minH.count != 0){
        index = DeMinHeap(&minH);
        if(Find(set, edgeNode[index].begin) != Find(set, edgeNode[index].end)){
            Union(set, edgeNode[index].begin, edgeNode[index].end);
            printf("%d —— %d\n", edgeNode[index].begin, edgeNode[index].end);
        }
    }
    return 0;
}
