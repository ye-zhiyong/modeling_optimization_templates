#include "shortest-path.h"

/**
 * 1. top-down分治，递归程序，DFS，单源最短距离。适用负权图。
 *    最坏情况下执行效率O(|V||E|)。执行时间测试1us。
 * 
 * 2. down-top分治，队列，BFS，单源最短距离，迭代程序。适用负权图。(Shortest Path Faster Algorithm, SPFA Algorithm)
 *    最坏情况下执行效率O(|V||E|)。执行时间测试1us。
 * 
 * 3. down-top分治，根据计算节点依赖关系，每次选取入度为0节点进行松弛/更新，使用inDegree数组和队列，拓扑排序思想。适用负权图。
 *    执行效率O(|V|+|E|)。执行时间测试1us。
 * 
 * 4. 每次扫描所有边并对相邻节点进行松弛/更新，直到某次扫描没有产生任何更新为止，最坏情况下需要扫描|V|-1次，除非存在负权环图。适用负权图。（Bellman-Ford's Algorithm）
 *    最坏情况下执行效率O(|V||E|)。执行时间测试1us。
 * 
 * 5. top-down分治，每次选取未被访问的最短距离节点进行松弛/更新，使用visited[]和dist[]，贪心策略。适用有环图。(Dijkstra's Algorithm)
 *    执行效率O(|V|^2)。执行时间测试1us。
 * 
 * 6. top-down分治，每次选取未被访问的最短距离节点进行松弛/更新，使用小根堆MinHeap-优先级队列，贪心策略。适用有环图。(Dijkstra's Algorithm)
 *    执行效率O(ElogV)。执行时间测试1us。
 * 
 * 7. 多源最短路径，每次选取中间节点进行松弛/更新，使用dist[][]和prev[][]，动态规划思想。(Floyd's Algorithm)
 *    执行效率O(|V|^3)。执行时间测试26us。
 */

int* ShortestPathTopeDownDivideDFSWithoutRepetition(ALGraph *G, int src){
    static int dist[maxVertices];
    static int sign = 1;
    if(sign == 1){
        for(int j = 0; j < maxVertices; j++)
            dist[j] = maxWeight;
        dist[src] = 0;
        sign++;
    }
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
        if(dist[src] + p->edge < dist[p->destVertex]){
            dist[p->destVertex] = dist[src] + p->edge;
            prev[p->destVertex] = src;
            ShortestPathTopeDownDivideDFSWithoutRepetition(G, p->destVertex);
        } 
        p = p->next;
    }
    return prev;
}

int* ShortestPathDownTopDivideBFSDynamicProgramWithouRepetition(ALGraph *G, int src){
    CircQueue Q;
    InitCircQueue(&Q, maxVertices);
    int dist[maxVertices];
    for(int i = 1; i < maxVertices; i++)
        dist[i] = maxWeight;
    dist[0] = 0;
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
            if(dist[vertex] + p->edge < dist[p->destVertex]){
                EnCircQueue(&Q, p->destVertex);
                prev[p->destVertex] = vertex;
                dist[p->destVertex] = dist[vertex] + p->edge;
            }
            p = p->next;
        }
    }
    return prev;
}

int* ShortestPathDownTopDivideTaskDependencyTopologicalSort(ALGraph *G, int src){
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
    for(int j = 1; j < maxVertices; j++)
        dist[j] = maxWeight;
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
            if(dist[vertex] + p->edge < dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            p = p->next;
        }
    }
    return prev;
}

int* ShortestPathBellmanFord(ALGraph* G, int src){
    int dist[maxVertices];
    for(int i = 0; i < maxVertices; i++){
        dist[i] = maxWeight;
    }
    dist[src] = 0;
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int j = 0; j < maxVertices; j++){
        prev[j] = -1;
    }
    int update = 1;  //success if there is no update each scans
    int scans = 0;  //numbers of scans
    //start the |V|-1 scans at most 
    do{
        scans++;
        if(scans > G->numVertices - 1){
            printf("Error: this is a negative weight ring graph.\n");
            return prev;    
        }
        update = 0;
        for(int k = 0; k < G->numVertices; k++){
            
            EdgeNode* p = G->vertexList[k].next;
            while(p != NULL){
                if(dist[k] + p->edge < dist[p->destVertex]){
                    dist[p->destVertex] = dist[k] + p->edge;
                    prev[p->destVertex] = k;
                    update = 1;
                }
                p = p->next;
            }
        }
    }while(update);
    return prev;
}

int* ShortestPathHeuristicGreedyDijkstraRecord(ALGraph* G, int src){
    int dist[maxVertices];
    for(int i = 0; i < maxVertices; i++){
        dist[i] = maxWeight;
    }
    dist[src] = 0;
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int j = 0; j < maxVertices; j++){
        prev[j] = -1;
    }
    int visited[maxVertices] = {0};
    for(int i = 0; i < maxVertices; i++){
        //1. search the nearest distance vertex without access
        int minDist = maxWeight;
        int vertex = 0;
        for(int j = 0; j < maxVertices; j++){
            if (visited[j] == 0 && dist[j] < minDist)
            {   
                minDist = dist[j];
                vertex = j;
            }
        }
        //2. access
        visited[vertex] = 1;
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            if(dist[vertex] + p->edge < dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            p = p->next;
        }
    }
    return prev;
}

int* ShortestPathHeuristicGreedyDijkstraMinHeap(ALGraph* G, int src){
    int dist[maxVertices];
    for(int i = 0; i < maxVertices; i++){
        dist[i] = maxWeight;
    }
    dist[src] = 0;
    int* prev = (int*)malloc(maxVertices * sizeof(int));
    for(int j = 0; j < maxVertices; j++){
        prev[j] = -1;
    }
    MinHeap minH;
    InitMinHeap(&minH);
    EnMinHeap(&minH, dist[src], src);
    for(int k = 0; k < maxVertices; k++){
        //1. DeMinHeap
        int vertex = DeMinHeap(&minH);
        //2. access
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            if(dist[vertex] + p->edge < dist[p->destVertex]){
                dist[p->destVertex] = dist[vertex] + p->edge;
                prev[p->destVertex] = vertex;
            }
            EnMinHeap(&minH, dist[p->destVertex], p->destVertex);
            p = p->next;
        }
    }
    return prev;
}

int ShortestPathFloydDynamicProgram(MGraph* G){
    //1. create dist[][]、prev[][]
    EdgeType dist[maxVertices][maxVertices];
    int prev[maxVertices][maxVertices];
    for(int i = 0; i < G->numVertices; i++){
        for(int j = 0; j < G->numVertices; j++){
            dist[i][j] = G->edge[i][j];
            if( dist[i][j] == 0 || dist[i][j] == maxWeight){
                prev[i][j] = -1;
            }
            else{
                prev[i][j] = i;
            }
        }
    }
    //2. update
    for(int k = 0; k < G->numVertices; k++){
        for(int i = 0; i < G->numVertices; i++){
            for(int j = 0; j < G->numVertices; j++){
                if(dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] =  dist[i][k] + dist[k][j];
                    prev[i][j] =  prev[k][j];
            }
        }   
    }
    //3. print tes（略）
    printf("%d\n", dist[0][5]);
    printf("%d—>%d\n", 0, prev[0][5]);
}