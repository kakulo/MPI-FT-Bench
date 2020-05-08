salloc -N 4 --ntasks-per-node=2 --time=00:05:00 -ppdebug mpirun -np 8 ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp2f -cp 1 -procfi -level 1 2>&1 | tee debug.txt
