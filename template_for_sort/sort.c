#include "sort.h"

/**
 * 1. 依次对相邻元素比较、交换。(Bubble Sort)
 *    稳定。原地。最好O(n)，最坏O(n^2)，平均O(n^2)。空间O(1)。
 * 
 * 2. 依次按顺序和前面有序元素进行比较，并插入、移动。(Insertion Sort)
 *    稳定。原地。最好O(n)，最坏O(n^2)，平均O(n^2)。空间O(1)。
 * 
 * 3. 按照步长组成列表，对列表进行插入排序，步长逐渐减小，直至为1，进行最后一次插排。(Shell Insertion Sort)
 *    不稳定。原地。时间复杂度难以分析。空间O(1)。
 * 
 * 4. 选择最小的，并交换到前面来。(Selection Sort)
 *    不稳定。原地。最好O(n^2)，最坏O(n^2)，平均O(n^2)。空间O(1)。
 * 
 * 5. 分治，[0,n-1]排序分解为三步：[0,(n-1)/2]排序、[(n-1)/2+1,n-1]排序、合并。(Merge Sort)
 *    稳定。不原地。最好、最坏和平均为O(nlogn)。空间O(n+logn)=O(n)。
 * 
 * 6. 分治，选取Pivot，小的放左边，大的放在右边，再分别对左右相同子问题作排序。(Quick Sort)。
 *    不稳定。原地。 最好O(nlogn)，最坏O(n^2)，平均O(nlohn)。空间O(n)。
 * 
 * 10. 创建优先级部队列MinHeap，依次取出最小元素，进行排序。(Heap Sort)
 *     不稳定。不原地。最好、最坏和平均是O(nlogn)。空间O(1)。
 * 
 * 
 * 7. ..... (Bucket Sort)
 * 
 * 
 * 8. ..... (Counting Sort)
 * 
 * 
 * 9. ..... (Radix Sort)
 * 
 * 
 */

void PrintArray(int array[], int arraySize){
    printf("Sorted array: ");
    for(int i = 0; i < arraySize; i++){
        printf("%d ", array[i]);
    }
    printf("\n");
}

void BubbleSort(int array[], int arraySize){
    if(arraySize <= 1)
        return -1;
    for(int i = 0; i < arraySize; i++){
        for(int j = 0; j < arraySize - i; j++){
            //1. compare
            if(array[j + 1] < array[j]){
                //2. exchange
                int tmp = array[j + 1];
                array[j + 1] = array[j];
                array[j] = tmp;
            }
        }
    }
}

void InsertionSort(int array[], int arraySize){
    if(arraySize <= 1)
        return -1;
    for(int i = 1; i < arraySize; i++){
            //1. compare one by one
            int value = array[i];
            int k = i - 1;
            for(; k >= 0; k--){
                if(value < array[k]){
                    //2. move
                    array[k + 1] = array[k];
                }
                else{
                    break;
                }
            }
            array[k + 1] = value;
    }
}

void ShellInsertionSort(int array[], int arraySize){
    ;
}

void SelectionSort(int array[], int arraySize){
    if(arraySize <= 1)
        return -1;
    for(int i = 0; i < arraySize - 1; i++){
        int minPos = i;
        //1. select minPos
        for(int j = i; j < arraySize; j++){
            if(array[j] < array[minPos]){
                minPos = j;
            }
        }
        //2. exchange
        int tmp = array[i];
        array[i] = array[minPos];
        array[minPos] = tmp; 
    }
}

void MergeSort(int array[], int arraySize){
    ;
}

void QuickSort(int array[], int arraySize)
{
    l = 0;
    r = arraySize - 1;
    if (l < r)
    {
        int i = l, j = r, x = array[l];
        while (i < j)
        {
            while(i < j && array[j] >= x)
                j--;  
            if(i < j) 
                array[i++] = array[j];
            
            while(i < j && array[i] < x) 
                i++;  
            if(i < j) 
                array[j--] = array[i];
        }
        array[i] = x;
        QuickSort(s, l, i - 1);  
        QuickSort(s, i + 1, r);
    }
}

void HeapSort(int array[], int arraySize){
    MinHeap minH;
    int tmp[maxArray] = {0};
    InitMinHeap(&minH);
    for(int k = 0; k < arraySize; k++){
        tmp[k] = array[k];
    }
    for(int i = 0; i < arraySize; i++){
        EnMinHeap(&minH, array[i], i);
    }
    
    for(int j = 0; j < arraySize; j++){
        int index = DeMinHeap(&minH);
        array[j] = tmp[index];
    }
}

void BucketSort(int array[], int arraySize){
    ;
}

void CountingSort(int array[], int arraySize){
    ;
}

void RadixSort(int array[], int arraySize){
    ;
}