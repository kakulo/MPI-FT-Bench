#salloc -N 2 --ntasks-per-node=1 --time=00:30:00 -ppdebug mpirun -np 2 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./miniVite config.L1.fti -p 3 -l -n 200 -CP2F -PROCFI -CP 2 -LEVEL 1
salloc -N 2 --ntasks-per-node=2 --time=00:30:00 -ppdebug mpirun -np 4 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./miniVite config.L1.fti -p 3 -l -n 200 -LEVEL 1 -CP 2