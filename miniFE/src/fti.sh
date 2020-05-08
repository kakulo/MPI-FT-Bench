salloc -N 2 --ntasks-per-node=4 --time=00:28:00 -ppdebug mpirun -np 8 ./miniFE.x config.L1.fti -nx 66 -ny 64 -nz 64 -cp 3 -level 1 -procfi 2>&1 | tee debug.txt
