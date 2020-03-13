salloc -N 2 --ntasks-per-node=4 --time=00:28:00 -ppdebug mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./miniFE.x -nx 66 -ny 64 -nz 64 -cp 1 -cp2f -procfi
