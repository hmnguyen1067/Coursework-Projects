Files:
- choFinalCPP.cpp: Cholesky implementation in C++
- choFinalJL.jl: Cholesky implementation in Julia
- dooFinalCPP.cpp: Doolittle algorithm implemtation in C++
- dooFinalJL.jl: Doolittle algorithm implementation in Julia
- choCPP*.out: Results of Cholesky implementation in C++ where * is the number of threads used
- dooCPP*.out: Results of Doolittle implementation in C++ where * is the number of threads used
- choJL.out: Results of Cholesky implementation in Julia (with 16 threads but it doesn't matter much due to reason I wrote in the report)
- dooJL.out: Results of Doolittle implementation in Julia (with 16 threads but it doesn't matter much due to reason I wrote in the report)


In order to generate the results from the code and job scripts, please follow the instructions below.
For CPP code:
- Step 1: Log in to the cluster
- Step 2: Copy both the code and the job script into 1 folder
- Step 3: Run "g++ -fopenmp -o dooFinalCPP.exe dooFinalCPP.cpp" to generate the executable for Doolittle algorithm results
- Step 3*: Run "g++ -fopenmp -o choFinalCPP.exe choFinalCPP.cpp" to generate the executable for Cholesky algorithm results
- Step 4: Run "sbatch dooCPP*.sb" where * is either 4,8,16 or 32 for number of threads to generate job for Doolittle
- Step 4*: Run "sbatch choCPP*.sb" where * is either 4,8,16 or 32 for number of threads to generate job for Cholesky

For Julia code:
- Step 1: Log in to the cluster
- Step 2: Copy both the code and the job scripts into 1 folder
- Step 3: Run "srun --partition=shared --pty --account=csd453 --nodes=1 --ntasks-per-node=2 --mem=32G -t 02:00:00 --wait=0 --export=ALL /bin/bash" to access interactive live node
- Step 4: Run "sbatch dooJulia.sb" to generate job for Doolittle
- Step 4*: Run "sbatch choJulia.sb" to generate job for Cholesky