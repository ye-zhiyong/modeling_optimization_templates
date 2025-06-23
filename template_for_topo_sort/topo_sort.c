#include "topo-sort.h"

/**
 * 1. 利用入度数组inDegree[]记录各节点入度，每访问一个节点，其入度减1，Kahn's Algorithm。
 *    执行效率O(|V|+|E|)。执行时间测试28us。
 * 2. 创建逆邻接表inverG，对逆邻接表进行DFS遍历。执行效率O(|V|+|E|）。执行时间测试2us。
 */

void TopoSortIndegree(ALGraph *G){
    int inDegree[maxVertices] = {0};
    for(int i = 0; i < maxVertices; i++){
        EdgeNode* p = G->vertexList[i].next;
        while(p != NULL){
            inDegree[p->destVertex]++;
            p = p->next;
        }
    }
    int Queue[maxVertices] = {0};
    int head, tail = 0;
    for(int j = 0; j < maxVertices; j++){
        if(inDegree[j] == 0){
            Queue[tail++] = j;
        }
    }
    printf("Topologic Sort is ");
    while(head != tail){
        //1. dequeue
        int vertex = Queue[head++];
        //2. access
        printf("%d ", vertex);
        ;
        //3. enqueue
        EdgeNode* p = G->vertexList[vertex].next;
        while(p != NULL){
            inDegree[p->destVertex]--;
            if(inDegree[p->destVertex] == 0){
                Queue[tail++] = p->destVertex;
            }
            p = p->next;
        }
    }
    printf("\n");
}

void TransInverseALGraph(ALGraph* G, ALGraph* inverG){
    //initial inverG
    for(int j = 0; j < maxVertices; j++){
        inverG->numEdges = G->numEdges;
        inverG->numVertices = G->numVertices;
        inverG->vertexList[j].vertex = G->vertexList[j].vertex;
        inverG->vertexList[j].next = NULL;
    }
    //create inverG
    for(int i = 0; i < maxVertices; i++){
        EdgeNode* p = G->vertexList[i].next;
        while(p != NULL){
            EdgeNode* q = (EdgeNode*)malloc(sizeof(EdgeNode));
            q->destVertex = i;
            q->edge = p->edge;
            q->next = inverG->vertexList[p->destVertex].next;
            inverG->vertexList[p->destVertex].next = q;
            p = p->next;
        }
    }
}

void TopoSortDFS(ALGraph *inverG, int vertex, int visited[]){
    EdgeNode* p = inverG->vertexList[vertex].next;
    while(p != NULL){
        if(visited[p->destVertex] == 0){
            TopoSortDFS(inverG, p->destVertex, visited);
            visited[p->destVertex] = 1;
        }
        p = p->next;
    }
    printf("%d ", vertex);
}

void TopoSortInverseALGraph(ALGraph *G){
    ALGraph inverG;
    TransInverseALGraph(G, &inverG);
    int visited[maxVertices] = {0};
    //access
    printf("\n");
    printf("Topologic sort is ");
    for(int i = 0; i < G->numVertices; i++){
        if(G->vertexList[i].next == NULL){
            TopoSortDFS(&inverG, i, visited); //对inverG，从入度为0的节点开始DFS
        }
    }
    printf("\n");
}