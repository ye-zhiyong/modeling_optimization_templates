#include "graph-traversal.h"

/**
 * 1. 分治，利用队列依次访问节点，确保避免访问重复/相等节点，BFS。
 *    执行效率O(|V|+|E|)。执行时间测试1us。
 * 
 * 2. top-down分治，但不访问相等/重复节点，DFS。
 *    执行效率O(|V|+|E|)。执行时间测试1us。
 */

void PrintGraphPath(int prev[], int src, int dest){
    if(src == dest){
        return;
    }
    PrintGraphPath(prev, src, prev[dest]);
    printf("%d ——> %d\n", prev[dest], dest);
}


int* SearchPathALGraphCircQueueBFS(ALGraph* G, int src, int dest){
    CircQueue Q;
    InitCircQueue(&Q, maxVertices);
    static int visited[maxVertices] = {0};
    static int* prev;
    prev = (int*)malloc(maxVertices * sizeof(int));
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
            if(p->destVertex == dest){
                prev[p->destVertex] = vertex;
                break;
            }
            if(visited[p->destVertex] != 1){
                EnCircQueue(&Q, p->destVertex);
                visited[p->destVertex] = 1;
                prev[p->destVertex] = vertex;
            }
            p = p->next;
        }
    }
    return prev;
}

int* SearchPathALGraphDFSRecursiveWithoutRepetition(ALGraph* G, int src, int dest){
    static int visited[maxVertices] = {0};
    static int flag = 0;
    static int* prev;
    prev = (int*)malloc(maxVertices * sizeof(int));
    if(flag == 0){
        for(int i = 0; i < maxVertices; i++){
            prev[i] = -1;
        }
        flag++;
    }
    //1. access src：self-definition
    ;
    //2. access src's adjacent graph one by one
    EdgeNode* p = G->vertexList[src].next;
    while(p != NULL){
        if(p->destVertex == dest){
            prev[p->destVertex] = src;
            return prev;
        }
        if(visited[p->destVertex] != 1){
            SearchPathALGraphDFSRecursiveWithoutRepetition(G, p->destVertex, dest);
            visited[p->destVertex] = 1;
            prev[p->destVertex] = src;
        }
        p = p->next;
    }
    return prev;
}
