#!/bin/bash
#MSUB -l walltime=00:10:00
#MSUB -q pbatch
#MSUB -e ulfm.err
#MSUB -o ulfm.out
#MSUB -l nodes=2

#mpirun -np 64 --mca orte_enable_recovery 1 --mca pml ^cm ../bin/CoMD-mpi -i4 -j4 -k4 -x 45 -y 45 -z 45
mpirun -np 8 --mca orte_enable_recovery 1 --mca pml ^cm ../bin/CoMD-mpi -i2 -j2 -k2 -x 25 -y 25 -z 25
#mpirun -np 64 --mca orte_enable_recovery 1 ../bin/CoMD-mpi -i4 -j4 -k4 -x 30 -y 30 -z 30
