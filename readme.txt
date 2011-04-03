Using Multigrid To Solve 2D Discrete Poisson Equations (sequesntial code vs parallel code):

Testing sequential code against parallel code, to see the speedup achieved in parallel implementation. Parallel implementation is using Message Passing Interface (MPI).
As amount of data increases, parallel execution gives more gain as compared to sequential. In a parallel system with lesser volume of data, communication costs dominate. Observed crossover point for 4 processor system is around data-size 513-by-513. For 16 and 64 processors it’s around 1025-by 1025.


Seq Code:
Code file: multigrid_seq.cpp
To Complie: mpiCC multigrid_seq.cpp -o m1
To Run: mpirun -np 1 m1


Parallel Code:
Code file: multigrid_parallel.cpp
To Complie: mpiCC multigrid_parallel.cpp -o m
To Run: mpirun -np <number of processors> m


Assumptions:
1. Seq code works with only "-np 1"
2. Data size is n x n where n = (2^m + 1)
3. Number of processors is a square and a power of 2
4. Number of processors less than n
5. To change data size modify the value (#define SIZE).