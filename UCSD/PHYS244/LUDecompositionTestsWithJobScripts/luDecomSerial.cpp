#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

double A[STEPS][STEPS];
double L[STEPS][STEPS];
double U[STEPS][STEPS];

int main(){
    //omp_set_num_threads(THREAD_NUM);

    /* Set the random seed */
    srand(time(NULL));

    /* Initialize the matrix randomly */
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j < STEPS; j++) A[i][j] = rand()%1000 + 1;
    }

    // Initialize loop counter as private variables
    int i, j, k;

    // Get start time
    auto start = high_resolution_clock::now();

    /* Executing the Doolittle algorithm using dynamic schedule*/
    #pragma omp parallel shared(A, L, U) private(i,j,k)
    {
        for (k = 0; k < STEPS; k++) {
            L[k][k] = 1;

            #pragma omp for schedule(dynamic, CHUNK)
            for (j = k; j < STEPS; j++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[k][a] * U[a][j];

                U[k][j] = A[k][j] - sum;
            }

            #pragma omp for schedule(dynamic, CHUNK)
            for (i = k+1; i < STEPS; i++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[i][a] * U[a][k];

                L[i][k] = (A[i][k] - sum)/U[k][k];
            }
        }
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout <<"Total runtime is: "<< duration.count() << " microseconds" << endl;

    return 0;
}
