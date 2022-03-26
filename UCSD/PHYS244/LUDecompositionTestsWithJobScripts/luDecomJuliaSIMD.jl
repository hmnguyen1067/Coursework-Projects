# Checking for the required packages
# Might take quite some time for compilation
#import Pkg
#Pkg.add("BenchmarkTools");
#Pkg.add("Random");

using BenchmarkTools
using Random

# The sequential LU decomposition function
# Here, I use a @fastmath macro in order to speed
# up the summing process
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

function LUFastmath(A, L, U, STEPS)
    # The sequential LU decomposition function
    # Here, I use a @fastmath macro in order to speed
    # up the summing process
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = 0.0
            @fastmath @inbounds @simd for a = 1:k
                sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = 0.0
            @fastmath @inbounds @simd for a = 1:k
                sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

function LUSIMD(A, L, U, STEPS)
    # The sequential LU decomposition function
    # Here, I use a @fastmath macro in order to speed
    # up the summing process
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = 0.0
            @simd for a = 1:k
                @inbounds sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = 0.0
            @simd for a = 1:k
                @inbounds sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end



# Setting the size for the matrix
const STEPS = 128;

# Since matrix initialization involves randomized
# data so I set the seed for easy replication
Random.seed!(42);

function runMain(STEPS)
    # Initialize matrix A to be filled with random values between 1 and 100
    # while matrix L and U are filled with zeroes
    As = rand(1.0: 1.0: 100.0, STEPS, STEPS)
    Ls = zeros(STEPS, STEPS)
    Us = zeros(STEPS, STEPS)

    # In order to get the time, we run through multiple samples
    # record all the timings and get the one with minimum value
    tiSeq = @btime LUSequential($As,$Ls,$Us,$STEPS)
    As = rand(1.0: 1.0: 100.0, STEPS, STEPS)
    Ls = zeros(STEPS, STEPS)
    Us = zeros(STEPS, STEPS)
    tiFM = @btime LUFastmath($As,$Ls,$Us,$STEPS)
    #tiSIMD = @benchmark LUSIMD($As,$Ls,$Us,$STEPSs)
    #timeSeq = minimum(tiSeq.times)/1e6
    #timeFM = minimum(tiFM.times)/1e6
    #timeSIMD = minimum(tiSIMD.times)/1e6

    # Output the result
    #println("Total runtime for sequential procedure of size $STEPS is: $timeSeq miliseconds")
    #println("Total runtime for fastmath procedure of size $STEPS is: $timeFM miliseconds")
    #println("Total runtime for SIMD procedure of size $STEPS is: $timeSIMD miliseconds")
end

@elapse runMain(STEPS)
