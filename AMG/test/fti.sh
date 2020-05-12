rm debug.txt -fr
salloc -N 4 --ntasks-per-node=2 --time=00:15:00 -ppdebug mpirun -np 8 ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp2f -cp 1 -procfi -level 1 2>&1 | tee -a debug.txt 
salloc -N 4 --ntasks-per-node=2 --time=00:15:00 -ppdebug mpirun -np 8 ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp2f -cp 1 -level 1 2>&1 | tee -a debug.txt
#salloc -N 4 --ntasks-per-node=2 --time=00:15:00 -ppdebug mpirun -np 8 ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp2f -cp 1 -nodefi -level 1 2>&1 | tee -a debug.txt || mpirun -np 8 ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp2f -cp 1 -level 1 2>&1 | tee -a debug.txt
#salloc -N 4 --ntasks-per-node=2 --time=00:05:00 -ppdebug mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./amg config.L1.fti -problem 2 -n 40 40 40 -P 2 2 2 -cp 1 -procfi -level 1
