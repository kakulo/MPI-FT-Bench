salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./lulesh2.0 config.L1.fti -s 3 -p -cp 5 -level 1 -procfi 2>&1 | tee debug.txt

