rm debug.txt -fr
salloc -N 2 --ntasks-per-node=2 --time=00:30:00 -ppdebug mpirun -np 4 ./miniVite config.L1.fti -p 3 -l -n 200 -LEVEL 1 -CP 1 -PROCFI 2>&1 | tee -a debug.txt
salloc -N 2 --ntasks-per-node=2 --time=00:30:00 -ppdebug mpirun -np 4 ./miniVite config.L1.fti -p 3 -l -n 200 -LEVEL 1 -CP 1 2>&1 | tee -a debug.txt
