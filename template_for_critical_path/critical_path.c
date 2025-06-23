#include "critical-path.h"

/**
 * 1. top-down分治，递归程序，DFS，单源最长距离。适用负权图。
 *    最坏情况下执行效率O(|V||E|)。执行时间测试1us。
 * 
 * 2. down-top分治，队列，BFS，单源最长距离，迭代程序。适用负权图。(SPFA Algorithm's Modification)
 *    最坏情况下执行效率O(|V||E|)。执行时间测试1us。
 * 
 * 3. down-top分治，根据计算节点依赖关系，每次选取入度为0节点进行松弛/更新，使用inDegree数组和队列，拓扑排序思想。适用负权图。
 *    执行效率O(|V|+|E|)。执行时间测试1us。
 * 
 * 4. top-down分治，每次选取未被访问的最长距离节点进行松弛/更新，使用visited[]和dist[]。适用有环图。(Dijkstra's Algorithm's Modification)
 *    执行效率O(|V|^2)。执行时间测试1us。
 * 
 * 5. top-down分治，每次选取未被访问的最长距离节点进行松弛/更新，使用小根堆MinHeap-优先级队列。适用有环图。(Dijkstra's Algorithm's Modification)
 *    执行效率O(ElogV)。执行时间测试1us。
 */

int* CriticalPathTopeDownDivideDFSWithoutRepetition(ALGraph *G, int src){
    static int dist[maxVertices] = {0};
    static int sign = 1;
    static int* prev;
    static int ps = 1;
    if(ps == 1){
        prev = (int*)malloc(maxVertices * sizeof(int));
        ps++;
    }
    static int flag = 1;
    if(flag == 1){
        for(int i = 0; i < maxVertices; i++){
            prev[i] = -1;
        }
        flag++;
    }
    EdgeNode* p = G->vertexList[src].next;
    while(p != NULL){  
        if(dist[src] + p->edge > dist[p->destVertex]){
            dist[p->destVertex] = dist[src] + p->edge;
            prev[p->destVertex] = src;
            CriticalPathTopeDownDivideDFSWithoutRepetition(G, p->destVertex);
        } 
        p = p->next;
    }
    return prev;
}

int* CriticalPathDownTopDivideBFSDynamicProgramWithouRepetition(ALGraph *G, int src){
    CircQueue Q;
    InitCircQueue(&Q, maxVertices);
    int dist[maxVertices] = {0};
    int* prev = (int*)malloc(G->numVertices * sizeof(int));
    for(int i = 0; i < maxVertices; i++){
        prev[i] = -1;
    }
    EnCircQueue(&Q, src);
    while(Q.head != Q.tail){
        //1.dequeue
        int vertex = DeCircQueue(&Q);
        //2.access：self-definition
        ;
        //3.enqueue including modify visited and prev 
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            if(dist[vertex] + p->edge > dist[p->destVertex]){
                EnCircQueue(&Q, p->destVertex);
                prev[p->destVertex] = vertex;
                dist[p->destVertex] = dist[vertex] + p->edge;
            }
            p = p->next;
        }
    }
    return prev;
}

int* CriticalPathDownTopDivideTaskDependencyTopologicalSort(ALGraph *G, int src){
    CircQueue Q;
    InitCircQueue(&Q, maxVertices);
    int inDegree[maxVertices] = {0};
    for(int i = 0; i < G->numVertices; i++){
        EdgeNode* p = G->vertexList[i].next;
        while(p != NULL){
            inDegree[p->destVertex]++;
            p = p->next;
        }
    }
    inDegree[src] = 0;
    int dist[maxVertices] = {0};
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int k = 0; k < maxVertices; k++)
        prev[k] = -1;
    EnCircQueue(&Q, src);
    while(Q.head != Q.tail){
        //1.DeQueue
        int vertex = DeCircQueue(&Q);
        //2. access
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            inDegree[p->destVertex]--;
            if(inDegree[p->destVertex] == 0)
                EnCircQueue(&Q, p->destVertex);
            if(dist[vertex] + p->edge > dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            p = p->next;
        }
    }
    return prev;
}


int* CriticalPathHeuristicGreedyDijkstraRecord(ALGraph* G, int src){
    int dist[maxVertices] = {0};
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int j = 0; j < maxVertices; j++){
        prev[j] = -1;
    }
    int visited[maxVertices] = {0};
    for(int i = 0; i < maxVertices; i++){
        //1. search the longest distance vertex without access
        int maxDist = 0;
        int vertex = 0;
        for(int j = 0; j < maxVertices; j++){
            if (visited[j] == 0 && dist[j] >= maxDist)
            {   
                maxDist = dist[j];
                vertex = j;
            }
        }
        //2. access
        visited[vertex] = 1;
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            if(dist[vertex] + p->edge > dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            p = p->next;
        }
    }
    return prev;
}

int* CriticalPathHeuristicGreedyDijkstraMinHeap(ALGraph* G, int src){
    int dist[maxVertices] = {0};
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int j = 0; j < maxVertices; j++){
        prev[j] = -1;
    }
    MaxHeap maxH;
    InitMaxHeap(&maxH);
    EnMaxHeap(&maxH, dist[src], src);
    printf("H");for(int k = 0; k < maxVertices; k++){
        //1. DeMinHeap
        int vertex = DeMaxHeap(&maxH);
        //2. access
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            if(dist[vertex] + p->edge > dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            EnMaxHeap(&maxH, dist[p->destVertex], p->destVertex);
            p = p->next;
        }
    }
    return prev;
}