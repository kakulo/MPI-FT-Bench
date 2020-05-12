rm debug.txt -fr
salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 ./CoMD-mpi config.L1.fti -xproc 2 -yproc 2 -zproc 2 -nx 25 -ny 25 -nz 25 -level 1 -cp_stride 2 -procfi 2>&1 | tee -a debug.txt
salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 ./CoMD-mpi config.L1.fti -xproc 2 -yproc 2 -zproc 2 -nx 25 -ny 25 -nz 25 -level 1 -cp_stride 2 2>&1 | tee -a debug.txt
