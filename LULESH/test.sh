#!/bin/bash
#MSUB -l walltime=00:15:00
#MSUB -q pbatch
#MSUB -e ulfm.err
#MSUB -o ulfm.out
#MSUB -l nodes=3

mpirun -np 27 --mca orte_enable_recovery 1 --mca pml ^cm ./lulesh2.0 -s 3 -p -cp 3
