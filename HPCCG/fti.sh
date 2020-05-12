rm debug.txt -fr
salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 ./test_HPCCG config.L1.fti 64 64 128 10 -level 1 -procfi 2>&1 | tee -a debug.txt
salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 ./test_HPCCG config.L1.fti 64 64 128 10 -level 1 2>&1 | tee -a debug.txt
