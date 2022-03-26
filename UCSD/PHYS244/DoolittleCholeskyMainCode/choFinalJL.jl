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

# The function is used to create a Hermitian
# positive-definite matrix
function generateSPD!(A::Matrix{Float64}, STEPS::Int64)
    A = A*A'
    A = A + STEPS*I
    return A
end

# This function is used for doing sequential Cholesky
# factorization by row
# Please find the report for better algorithm description
function choSeqRow(A::Matrix{Float64}, Ls::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    for i = 1:STEPS
        for j = 1:i
            sum = zero(eltype(A))

            if i == j
                for k = 1:j
                    sum += L[j,k]^2
                end
                L[j,j] = sqrt(A[j,j] - sum)
            else
                for k = 1:j
                    sum += L[i,k] * L[j,k]
                end
                L[i,j] = (A[i,j] - sum)/ L[j,j]
            end
        end
    end
end

# This function is used for doing sequential Cholesky
# factorization by column
# Please find the report for better algorithm description
function choSeqCol(A::Matrix{Float64}, Ls::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    for i = 1:STEPS
        sum = zero(eltype(A))
        for j = 1:i
            sum += L[j,i]^2
        end
        L[i,i] = sqrt(A[i,i]-sum)

        for k = i + 1:STEPS
            sum = zero(eltype(A))
            for l = 1:i
                sum += L[l,k]*L[l,i]
            end
            L[i,k] = (A[i,k] - sum)/L[i,i]
        end
    end
end

# This function is used for doing sequential Cholesky
# factorization by column with simd fastmath macro for speedup
# Please find the report for better algorithm description
function choSeqColFM(A::Matrix{Float64}, Ls::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    for i = 1:STEPS
        sum = zero(eltype(A))
        @fastmath @inbounds @simd for j = 1:i
            sum += L[j,i]^2
        end
        L[i,i] = sqrt(A[i,i]-sum)

        for k = i + 1:STEPS
            sum = zero(eltype(A))
            @fastmath @inbounds @simd for l = 1:i
                sum += L[l,k]*L[l,i]
            end
            L[i,k] = (A[i,k] - sum)/L[i,i]
        end
    end
end

# This function is used for doing parallel Cholesky
# factorization by column
# Please find the report for better algorithm description
function choSeqColTh(A::Matrix{Float64}, Ls::Matrix{Float64}, STEPS::Int64)
    L = deepcopy(Ls)
    for i = 1:STEPS
        # Vector S is used to hold intermediate value for each thread
        # before contributing it to the total sum
        S = zeros(eltype(A), nthreads())
        sum = zero(eltype(A))

        @threads for j = 1:i
            @inbounds S[threadid()] += L[j,i]^2
        end

        for b in eachindex(S)
            sum += S[b]
        end

        L[i,i] = sqrt(A[i,i]-sum)

        for k = i + 1:STEPS
            # Vector S is used to hold intermediate value for each thread
            # before contributing it to the total sum
            S = zeros(eltype(A), nthreads())
            sum = zero(eltype(A))

            @threads for l = 1:i
                @inbounds S[threadid()] += L[l,k]*L[l,i]
            end

            for b in eachindex(S)
                sum += S[b]
            end

            L[i,k] = (A[i,k] - sum)/L[i,i]
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
    As = generateSPD!(As, STEPS)
    Ls = zeros(STEPS, STEPS)

    # In order to get the time, we run through multiple samples
    # record all the timings and get the one with minimum value
    println("Total runtime for sequential Cholesky procedure by row of size $STEPS is: ")
    @btime choSeqRow($As, $Ls, $STEPS)
    println("Total runtime for sequential Cholesky procedure by col of size $STEPS is: ")
    @btime choSeqCol($As, $Ls, $STEPS)
    println("Total runtime for sequential Cholesky with simd fastmath procedure of size $STEPS is: ")
    @btime choSeqColFM($As, $Ls, $STEPS)

    # Too large of the steps will make the threaded version bursts off memory
    if (STEPS < 4096)
    println("Total runtime for threaded Cholesky procedure of size $STEPS is: ")
    @btime choSeqColTh($As, $Ls, $STEPS)
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
