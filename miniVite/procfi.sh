salloc -N 2 --ntasks-per-node=1 --time=00:30:00 -ppdebug mpirun -np 2 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./miniVite -p 3 -l -n 200 -CP2F -PROCFI -CP 1
