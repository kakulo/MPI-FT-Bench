rm debug.txt -fr
salloc -N 4 --ntasks-per-node=2 --time=00:28:00 -ppdebug mpirun -np 8 ./miniFE.x config.L1.fti -nx 66 -ny 64 -nz 64 -cp 3 -procfi -level 1 2>&1 | tee -a debug.txt
salloc -N 4 --ntasks-per-node=2 --time=00:28:00 -ppdebug mpirun -np 8 ./miniFE.x config.L1.fti -nx 66 -ny 64 -nz 64 -cp 3 -level 1 2>&1 | tee -a debug.txt
