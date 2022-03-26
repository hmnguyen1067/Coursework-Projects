#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>
#include <cstring>

using namespace std;
using namespace std::chrono;

// A function used to generate Hermitian symmetric
// positive-definite matrix
template <typename TwoD>
void generateSPD(TwoD& A, int STEPS) {
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j < STEPS; j++) {
            if (i == j) { A[i][j] += STEPS; }
            else if (j>i) { A[i][j] = 0.5*(A[i][j] + A[j][i]); }
            else { A[i][j] = A[j][i]; }
        }
    }
}

// This function is used for doing sequential Cholesky
// factorization (no OMP support)
// Please find the report for better algorithm description
template <typename TwoD>
void choSeq(TwoD& A, TwoD& L, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    // Since L is a lower triangular matrix in this case
    // so we can just ignore everything above the
    // diagonal and only calculate those below it
    // The detail of the algorithm is included 
    // in the report
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;

            // Things that are on the diagonal
            if (j == i)
            {
                for (int k = 0; k < j; k++) sum += pow(L[j][k], 2);
                L[j][j] = sqrt(A[j][j] - sum);
            } else { // Things that are below the diagonal
                for (int k = 0; k < j; k++) sum += (L[i][k] * L[j][k]);
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
}

// Cholesky factorization algorithm with OMP support 
// and dynamic scheduler
template <typename TwoD>
void choOMPDynamic(TwoD& A, TwoD& L, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    // Here, I set up a parallel region where the matrices are
    // shared data while the indices are private
    #pragma omp parallel shared(A, L) private(i,j,k)
    {
        for (int i = 0; i < STEPS; i++) {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j <= i; j++) {
                double sum = 0;
    
                if (j == i)
                {
                    for (int k = 0; k < j; k++) sum += pow(L[j][k], 2);
                    L[j][j] = sqrt(A[j][j] - sum);
                } else {
                    for (int k = 0; k < j; k++) sum += (L[i][k] * L[j][k]);
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
    }
}

// Cholesky factorization algorithm with OMP support 
// and static scheduler
template <typename TwoD>
void choOMPStatic(TwoD& A, TwoD& L, int STEPS) {
    // Initialize loop counter as private variables
    int i, j, k;

    // Here, I set up a parallel region where the matrices are
    // shared data while the indices are private
    #pragma omp parallel shared(A, L) private(i,j,k)
    {
        for (int i = 0; i < STEPS; i++) {
            #pragma omp for schedule(static)
            for (int j = 0; j <= i; j++) {
                double sum = 0;
    
                if (j == i)
                {
                    for (int k = 0; k < j; k++) sum += pow(L[j][k], 2);
                    L[j][j] = sqrt(A[j][j] - sum);
                } else {
                    for (int k = 0; k < j; k++) sum += (L[i][k] * L[j][k]);
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
    }
}

// Run benchmarking of each algorithm on a size number
void runMain(int STEPS) {
    // Initialize empty matrix then I will fill them out later
    double** A = new double*[STEPS];
    double** L = new double*[STEPS];

    for(int i = 0; i < STEPS; i++) {
        A[i] = new double[STEPS];
        L[i] = new double[STEPS];
    }

    /* Initialize the matrix with random value */
    for (int i = 0; i < STEPS; i++) {
        for (int j = 0; j < STEPS; j++) A[i][j] = rand()%1000 + 1;
    }

    // Convert the matrix into a SPD matrix
    generateSPD(A, STEPS);

    // Get the start time of sequential algorithm
    auto startSeq = high_resolution_clock::now();
    choSeq(A, L, STEPS);
    // Get the end time of sequential algorithm
    auto stopSeq = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationSeq = duration_cast<milliseconds>(stopSeq - startSeq);
    // Print out the time
    cout <<"Total runtime for the Cholesky sequential algorithm of size "<<STEPS<<" is: "<< durationSeq.count() << " milliseconds" << endl;

    // Restart L back to the zero matrix after mutating it
    fill(&L[0][0], &L[0][0] + sizeof(L), 0);

    // Get the start time of static OMP algorithm
    auto startOMPSta = high_resolution_clock::now();
    choOMPStatic(A, L, STEPS);
    // Get the end time of static OMP algorithm
    auto stopOMPSta = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationOMPSta = duration_cast<milliseconds>(stopOMPSta - startOMPSta);
    // Print out the time
    cout <<"Total runtime for the Cholesky static OMP algorithm of size "<<STEPS<<" is: "<< durationOMPSta.count() << " milliseconds" << endl;

    // Restart L back to the zero matrix after mutating it
    fill(&L[0][0], &L[0][0] + sizeof(L), 0);

    // Get the start time of dynamic OMP algorithm
    auto startOMPDym = high_resolution_clock::now();
    choOMPDynamic(A, L, STEPS);
    // Get the end time of dynamic OMP algorithm
    auto stopOMPDym = high_resolution_clock::now();
    // Get the duration and cast it to microseconds
    auto durationOMPDym = duration_cast<milliseconds>(stopOMPDym - startOMPDym);
    // Print out the time
    cout <<"Total runtime for the Cholesky dynamic OMP algorithm of size "<<STEPS<<" is: "<< durationOMPDym.count() << " milliseconds" << endl;
    cout <<endl;

    for(int i = 0; i < STEPS; i++) {
        delete A[i];
        delete L[i];
    }

    delete A;
    delete L;
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
