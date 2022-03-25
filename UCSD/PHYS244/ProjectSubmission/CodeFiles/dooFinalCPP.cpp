#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// This function is used for doing sequential LU decomposition
// using Doolittle's algorithm (no OMP support)
// Please find the report for better algorithm description
template <typename TwoD>
void luSeq(TwoD& A, TwoD& L, TwoD& U, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    // Index k is used to go through the submatrices of A
    for (k = 0; k < STEPS; k++) {
            // We need matrix L to be unitary so the diagonal is all 1s
            L[k][k] = 1;

            // Loop for calculating matrix U
            for (j = k; j < STEPS; j++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[k][a] * U[a][j];

                U[k][j] = A[k][j] - sum;
            }

            // Loop for calculating matrix L
            for (i = k+1; i < STEPS; i++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[i][a] * U[a][k];

                L[i][k] = (A[i][k] - sum)/U[k][k];
            }
        }
}

// LU decomposition using Doolittle's algorithm 
// with dynamic scheduler
template <typename TwoD>
void luOMPDynamic(TwoD& A, TwoD& L, TwoD& U, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    // Here, I set up a parallel region where the matrices are
    // shared data while the indices are private
    #pragma omp parallel shared(A, L, U) private(i,j,k)
    {
        // Index k is used to go through the submatrices of A
        for (k = 0; k < STEPS; k++) {
            // We need matrix L to be unitary so the diagonal is all 1s
            L[k][k] = 1;

            // Loop for calculating matrix U
            #pragma omp for schedule(dynamic)
            for (j = k; j < STEPS; j++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[k][a] * U[a][j];

                U[k][j] = A[k][j] - sum;
            }

            // Loop for calculating matrix L
            #pragma omp for schedule(dynamic)
            for (i = k+1; i < STEPS; i++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[i][a] * U[a][k];

                L[i][k] = (A[i][k] - sum)/U[k][k];
            }
        }
    }
}

// LU decomposition using Doolittle's algorithm 
// with static scheduler
template <typename TwoD>
void luOMPStatic(TwoD& A, TwoD& L, TwoD& U, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    /* Executing the Doolittle algorithm using static schedule*/
    // Here, I set up a parallel region where the matrices are
    // shared data while the indices are private
    #pragma omp parallel shared(A, L, U) private(i,j,k)
    {
        // Index k is used to go through the submatrices of A
        for (k = 0; k < STEPS; k++) {
            // We need matrix L to be unitary so the diagonal is all 1s
            L[k][k] = 1;

            // Loop for calculating matrix U
            #pragma omp for schedule(dynamic)
            for (j = k; j < STEPS; j++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[k][a] * U[a][j];

                U[k][j] = A[k][j] - sum;
            }

            // Loop for calculating matrix L
            #pragma omp for schedule(dynamic)
            for (i = k+1; i < STEPS; i++) {
                double sum = 0;
                for (int a = 0; a <= k-1; a++) sum += L[i][a] * U[a][k];

                L[i][k] = (A[i][k] - sum)/U[k][k];
            }
        }
    }
}

void runMain(int STEPS) {
    // Initialize empty matrix then I will fill them out later
    double** A = new double*[STEPS];
    double** L = new double*[STEPS];
    double** U = new double*[STEPS];

    for(int i = 0; i < STEPS; i++) {
        A[i] = new double[STEPS];
        L[i] = new double[STEPS];
        U[i] = new double[STEPS];
    }

    /* Initialize the matrix with random value */
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j < STEPS; j++) A[i][j] = rand()%1000 + 1;
    }

    // Get the start time of sequential algorithm
    auto startSeq = high_resolution_clock::now();
    luSeq(A, L, U, STEPS);
    // Get the end time of sequential algorithm
    auto stopSeq = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationSeq = duration_cast<milliseconds>(stopSeq - startSeq);
    // Print out the time
    cout <<"Total runtime for the Doolittle sequential algorithm of size "<<STEPS<<" is: "<< durationSeq.count() << " milliseconds" << endl;

    // Restart L and U back to the zero matrix after mutating it
    fill(&L[0][0], &L[0][0] + sizeof(L), 0);
    fill(&U[0][0], &U[0][0] + sizeof(U), 0);

    // Get the start time of static OMP algorithm
    auto startOMPSta = high_resolution_clock::now();
    luOMPStatic(A, L, U, STEPS);
    // Get the end time of static OMP algorithm
    auto stopOMPSta = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationOMPSta = duration_cast<milliseconds>(stopOMPSta - startOMPSta);
    // Print out the time
    cout <<"Total runtime for the Doolittle static OMP algorithm of size "<<STEPS<<" is: "<< durationOMPSta.count() << " milliseconds" << endl;

    // Restart L and U back to the zero matrix after mutating it
    fill(&L[0][0], &L[0][0] + sizeof(L), 0);
    fill(&U[0][0], &U[0][0] + sizeof(U), 0);

    // Get the start time of dynamic OMP algorithm
    auto startOMPDym = high_resolution_clock::now();
    luOMPDynamic(A, L, U, STEPS);
    // Get the end time of dynamic OMP algorithm
    auto stopOMPDym = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationOMPDym = duration_cast<milliseconds>(stopOMPDym - startOMPDym);
    // Print out the time
    cout <<"Total runtime for the Doolittle dynamic OMP algorithm of size "<<STEPS<<" is: "<< durationOMPDym.count() << " milliseconds" << endl;
    cout <<endl;

    for(int i = 0; i < STEPS; i++) {
        delete A[i];
        delete L[i];
        delete U[i];
    }

    delete A;
    delete L;
    delete U;
}

// The function is used to run the benchmarking
// from size 2^start to 2^end
void executeMain(int start, int end) {
    for (int i = start; i <= end; i++) {
        int STEPS = 2<<i;
        runMain(STEPS);
    }
}

int main(){
    /* Set the random seed */
    srand(time(NULL));

    // Size start from 2^start
    int start = 6;

    // Size end at 2^end
    int end = 12;
    
    executeMain(start, end);

    return 0;
}
