salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 ./lulesh2.0 config.L1.fti -s 3 -p -cp 5 -level 1 -procfi 2>&1 | tee debug.txt
