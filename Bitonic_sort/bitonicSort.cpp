#include <algorithm>
#include <random>
#include <iostream>

int getNext (int inp){
    inp --;
    int i=0;
    while(inp){
        inp = inp>>1;
        i++;
    }
    inp=1;
    while(i--){
        inp = inp<<1;
    }
    return inp;
}

void compareExchange(int &a, int &b, bool direction) {
    if (direction == (a > b)) {
        // Swap a and b to sort in increasing order
        int temp = a;
        a = b;
        b = temp;
    }
}

void bitonicSort(int gpuArray[], int n) {

    for (int i = 2; i <= n; i = i << 1)
    {
        for(int j = i >> 1; j >= 1; j = j >> 1)
        {
            for (int k = 0; k < n/2; k++)
            {
                int ind1 = k, ind2 = k+n/2;
                if (ind1 < ind2)
                {
                    if ( (i & k) == 0 )
                    {
                        if (gpuArray[ind1] < gpuArray[ind2])
                        {
                            int temp = gpuArray[ind1];
                            gpuArray[ind1] = gpuArray[ind2];
                            gpuArray[ind2] = temp;
                        }
                    }
                    else
                    {
                        if (gpuArray[ind1] > gpuArray[ind2])
                        {
                            int temp = gpuArray[ind1];
                            gpuArray[ind1] = gpuArray[ind2];
                            gpuArray[ind2] = temp;
                        }
                    }
                }
            }
        }
    }
}


// void bitonicSeqMerge(int a[], int start, int BseqSize, int direction) {
//    if (BseqSize>1){
//       int k = BseqSize/2;
//       for (int i=start; i<start+k; i++)
//       if (direction==(a[i]>a[i+k]))
//       std::swap(a[i],a[i+k]);
//       bitonicSeqMerge(a, start, k, direction);
//       bitonicSeqMerge(a, start+k, k, direction);
//    }
// }
// void bitonicSortrec(int a[],int start, int BseqSize, int direction) {
//    if (BseqSize>1){
//       int k = BseqSize/2;
//       bitonicSortrec(a, start, k, 1);
//       bitonicSortrec(a, start+k, k, 0);
//       bitonicSeqMerge(a,start, BseqSize, direction);
//    }
// }
// void bitonicSort(int a[], int size, int up) {
//    bitonicSortrec(a, 0, size, up);
// }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    int newSize = getNext(size);
    int sizeDiff = newSize - size;
    // pad with these many zeroes
    int *gpuArray = (int *)malloc(newSize*sizeof(int));
    for(int i=size; i<newSize; i++){
        gpuArray[i] = 0;
    }
    for(int i=0; i<size; i++){
        gpuArray[i] = arrCpu[i];
    }

    // SORT BEGIN
    int logn = log2(newSize);
    printf("\n");

    for (int i = 1; i <= logn; i++)
    {   
        printf("i is %d\n", 1<<i);
        for(int j = i - 1; j >= 0; j--)
        {
            printf("%d : ", 1<<j);
            for (int k = 0; k < newSize; k++)
            {   
                int ind1 = k, ind2 = k ^ (1<<j);
                if (ind1 > ind2){
                    printf("%d %d , ", ind1, ind2);
                }
                // if ( ((1<<i) & k) == 0 ){
                //     if (gpuArray[ind1] > gpuArray[ind2]){
                //         int temp = gpuArray[ind1];
                //         gpuArray[ind1] = gpuArray[ind2];
                //         gpuArray[ind2] = temp;
                //     }
                // }
                // else{
                //     if (gpuArray[ind1] < gpuArray[ind2]){
                //         int temp = gpuArray[ind1];
                //         gpuArray[ind1] = gpuArray[ind2];
                //         gpuArray[ind2] = temp;
                //     }
                // }
            }
            std::cout << std::endl;

        }
        std::cout << std::endl;
    }

    // SORT ENDS

    // printiong the original array
    for(int i=0; i<size; i++)
        std::cout << arrCpu[i] << " ";

    std::cout << std::endl;
    
    for(int i=0; i<newSize; i++)
        std::cout << gpuArray[i] << " ";

    std::cout << std::endl;

    std::cout << "The size difference is: " << sizeDiff;

    std::cout << "\n\nActual Array is: " << std::endl;

    for(int i=sizeDiff; i< newSize; i++)
        std::cout << gpuArray[i] <<" ";

    std::cout << "\n\nExpected Array is: " << std::endl;

    std::sort(arrCpu, arrCpu+size);

    for(int i=0; i<size; i++)
        std::cout << arrCpu[i] << " ";

    std::cout << std::endl;

    return 0;


}