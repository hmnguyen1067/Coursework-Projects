# This section is to check the dependencies
import Pkg
try
    using BenchmarkTools
catch
    Pkg.add("BenchmarkTools")
end
try
    using LinearAlgebra
catch
    Pkg.add("LinearAlgebra")
end
try
    using Random
catch
    Pkg.add("Random")
end
using .Threads


# Sequential Doolittle algorithm
function dooSeq(A::Matrix{Float64}, Ls::Matrix{Float64}, Us::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    U = deepcopy(Us)
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1.0     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = zero(eltype(A))
            for a = 1:k
                sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = zero(eltype(A))
            for a = 1:k
                sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

# Sequential Doolittle algorithm with simd fastmath support
function dooFM(A::Matrix{Float64}, Ls::Matrix{Float64}, Us::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    U = deepcopy(Us)
    # The sequential LU decomposition function
    # Here, I use a @fastmath macro in order to speed
    # up the summing process
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1.0     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            sum = zero(eltype(A))
            @fastmath @inbounds @simd for a = 1:k
                sum += L[k, a]*U[a, j]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            sum = zero(eltype(A))
            @fastmath @inbounds @simd for a = 1:k
                sum += L[i, a]*U[a, k]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

# Threaded Doolittle algorithm
function dooTh(A::Matrix{Float64}, Ls::Matrix{Float64}, Us::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    U = deepcopy(Us)
    # Index k is used to go through submatrices of matrix A
    for k = 1:STEPS
        L[k, k] = 1.0     # The diagonal of the L matrix is set to all 1 so the matrix is unitary
        # This loop is used to solve for the U matrix
        for j = k:STEPS
            S = zeros(eltype(A), nthreads())
            sum = zero(eltype(A))

            @threads for a = 1:k
                @inbounds S[threadid()] += L[k, a]*U[a, j]
            end

            for b in eachindex(S)
                sum += S[b]
            end

            U[k, j] = A[k, j] - sum
        end

        # This loop is used to solve for the L matrix
        for i = k+1:STEPS
            S = zeros(eltype(A), nthreads())
            sum = zero(eltype(A))

            @threads for a = 1:k
                @inbounds S[threadid()] += L[i, a]*U[a, k]
            end

            for b in eachindex(S)
                sum += S[b]
            end

            L[i, k] = (A[i, k] - sum)/U[k, k]
        end
    end
end

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
    println("Total runtime for sequential Doolittle procedure of size $STEPS is: ")
    @btime dooSeq($As, $Ls, $Us, $STEPS)
    println("Total runtime for sequential Doolittle with simd fastmath procedure of size $STEPS is: ")
    @btime dooFM($As, $Ls, $Us, $STEPS)

    # Too large of the steps will make the threaded version bursts off memory
    if (STEPS < 4096)
    println("Total runtime for threaded Doolittle procedure of size $STEPS is: ")
    @btime dooTh($As, $Ls, $Us, $STEPS)
    end
    println()
end

function executeMain(nStart, nEnd)
    for i = nStart:nEnd
        STEPS = 2<<i
        runMain(STEPS)
    end
end

executeMain(6, 10)
