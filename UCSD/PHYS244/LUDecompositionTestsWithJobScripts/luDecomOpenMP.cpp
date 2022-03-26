#include <iostream>
#include <chrono>
#include <omp.h>

// Constant STEPS is the size of the matrix STEPS x STEPS
// Constant CHUNK is used to indicate the size of the chunk
// of the data in order to but into the dynamic scheduler
#define STEPS 1024
#define CHUNK 32

using namespace std;
using namespace std::chrono;

// Initialize empty matrix then I will fill them out later
double A[STEPS][STEPS];
double L[STEPS][STEPS];
double U[STEPS][STEPS];

int main(){
    /* Set the random seed */
    srand(time(NULL));

    /* Initialize the matrix with random value */
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j < STEPS; j++) A[i][j] = rand()%1000 + 1;
    }

    // Initialize loop counter as private variables
    int i, j, k;

    // Get start time
    auto start = high_resolution_clock::now();

    /* Executing the Doolittle algorithm using dynamic schedule*/
    // Here, I set up a parallel region where the matrices are
    // shared data while the indices are private
    #pragma omp parallel shared(A, L, U) private(i,j,k)
    {
        // Index k is used to go through the submatrices of A
        for (k = 0; k < STEPS; k++) {
            // We need matrix L to be unitary so the diagonal is all 1s
            L[k][k] = 1;

            // Loop for calculating matrix U
            #pragma omp for schedule(dynamic, CHUNK)
            for (j = k; j < STEPS; j++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[k][a] * U[a][j];

                U[k][j] = A[k][j] - sum;
            }

            // Loop for calculating matrix L
            #pragma omp for schedule(dynamic, CHUNK)
            for (i = k+1; i < STEPS; i++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[i][a] * U[a][k];

                L[i][k] = (A[i][k] - sum)/U[k][k];
            }
        }
    }

    // Get the end time
    auto stop = high_resolution_clock::now();

    // Get the duration and cast it to microseconds
    auto duration = duration_cast<microseconds>(stop - start);

    // Print out the time
    cout <<"Total runtime is: "<< duration.count() << " microseconds" << endl;

    return 0;
}
