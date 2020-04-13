salloc -N 2 --ntasks-per-node=4 --time=00:10:00 -ppdebug mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./lulesh2.0 -s 3 -p -cp 1 -cp2f -procfi
