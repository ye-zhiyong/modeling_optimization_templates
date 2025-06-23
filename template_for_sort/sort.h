#ifndef SORT_H_
#define SORT_H_

#include <stdio.h>
#include <stdlib.h>
#include "../adt/MinHeap.h"

#define maxArray 30

void PrintArray(int array[], int arraySize);
int BubbleSort(int array[], int arraySize);
int InsertionSort(int array[], int arraySize);
int ShellInsertionSort(int array[], int arraySize);
int SelectionSort(int array[], int arraySize);
void QuickSort(int array[], int arraySize);
int HeapSort(int array[], int arraySize);

#endif