salloc -N 4 --ntasks-per-node=2 --time=00:10:00 -ppdebug mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" ./test_HPCCG config.L1.fti 64 64 128 10 -level 1 -procfi 2>&1 | tee debug.txt

