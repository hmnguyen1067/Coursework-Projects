using FLoops
using BenchmarkTools
using Random
using Transducers
using .Threads

# Setting the size for the matrix
STEPS = 1024;

# Since matrix initialization involves randomized
# data so I set the seed for easy replication
Random.seed!(42);

function LUSequential(A, L, U, STEPS)
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1.0     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = 0.0
            for a = 1:k
                sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = 0.0
            for a = 1:k
                sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

function runMain(STEPS)
    # Initialize matrix A to be filled with random values between 1 and 100
    # while matrix L and U are filled with zeroes
    As = rand(1.0: 1.0: 100.0, STEPS, STEPS)
    Ls = zeros(STEPS, STEPS)
    Us = zeros(STEPS, STEPS)

    # In order to get the time, we run through multiple samples
    # record all the timings and get the one with minimum value
    #@time LUThreads(As,Ls,Us,STEPS)

    #As = rand(1.0: 1.0: 100.0, STEPS, STEPS)
    #Ls = zeros(STEPS, STEPS)
    #Us = zeros(STEPS, STEPS)

    a = @benchmark LUSequential($As,$Ls,$Us,$STEPS)
    #timeTh = minimum(tiTh.times)/1e6
    @show a
    2
    # Output the result
    #println("Total runtime for threads procedure of size $STEPS is: $timeTh miliseconds")
end
