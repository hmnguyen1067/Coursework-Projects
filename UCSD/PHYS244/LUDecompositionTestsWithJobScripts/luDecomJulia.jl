# Checking for the required packages
# Might take quite some time for compilation
import Pkg
Pkg.add("BenchmarkTools");
Pkg.add("Random");

using BenchmarkTools
using Random

# Setting the size for the matrix
const STEPS = 2048;

# Since matrix initialization involves randomized
# data so I set the seed for easy replication
Random.seed!(42);

# The sequential LU decomposition function
# Here, I use a @fastmath macro in order to speed
# up the summing process
function LUDecom(A, L, U, STEPS)
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = zero(eltype(A))
            for a = 1:k
                @fastmath sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = zero(eltype(A))
            for a = 1:k
                @fastmath sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

function runMain(STEPS)
    # Initialize matrix A to be filled with random values between 1 and 100
    # while matrix L and U are filled with zeroes
    A = rand(1.0: 1.0: 100.0, STEPS, STEPS)
    L = zeros(STEPS, STEPS)
    U = zeros(STEPS, STEPS)

    # In order to get the time, we run through multiple samples
    # record all the timings and get the one with minimum value
    ti = @benchmark LUDecom($A, $L, $U, STEPS)
    tTime = minimum(ti.times)/1e6

    # Output the result
    println("Total runtime for size $STEPS is: $tTime microseconds")
end
