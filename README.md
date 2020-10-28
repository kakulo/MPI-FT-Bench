# MATCH: An MPI Fault Tolerance Benchmark Suite 0.1
MATCH is an MPI fault tolerane benchmark suite, aiming to characterize, research, and comprehensively compare different combinations and configurations of MPI fault tolerance designs. MATCH is the first benchmark suite designed for MPI fault tolerance. MATCH showcases the implementation of three MPI fault tolerance techniques with six HPX proxy applications. 

# Three MPI fault tolerance techniques supported in MATCH
In the current version of MATCH, we support three MPI fault tolerance techniques (ULFM and Reinit++ for MPI recovery and FTI for data recovery). In addition, we also studied the performance of using ``Restart" (or start over) for MPI recovery.

- User-level Fault Mitigation (ULFM) is a leading MPI recovery framework that is in progress with the MPI Fault Tolerance Working Group. ULFM is open-source. You can download the ULFM latest version at https://bitbucket.org/icldistcomp/ulfm2/src/ulfm/. We use the latest ULFM version ``ULFM v4.0.1ulfm2.1rc1" based on Open MPI 4.0.1 in our development. Reinit/Reinit++ is an alternative solution for MPI recovery, and more efficient for MPI global backward recovery which is a natural option for HPC Bulk Synchronous Parallel (BSP) paradigm. Reinit++ is the latest versionof Reinit based on Open MPI 4.0.0 and open-source. It is available at https://github.com/ggeorgakoudis/ompi/tree/reinit. We use the latest version of Reinit++ in our development. 

- Fault Tolerance Interface (FTI) is a checkpointing library widely used by HPC developers for checkpointing. FTI provides programmers a number of APIs which are easy to use, and allows programmers to choose checkpointing strategy that fits the application. FTI enables multiple levels of reliability with different performance efficiency by utilizing local storage, data replication, and erasure codes. FTI is an application-level checkpointing. It requests users to decide which data objects to be checkpointed. Furthermore, FTI hides data processing details from users. Users only tell FTI the memory address and data size of the date object to be protected to enable checkpointing of the data object. FTI is open-source and available at https://github.com/leobago/fti. FTI is progressively maintained by its developers. We use the latest version of FTI in our development. 

# Use of MATCH
- MATCH and the MATCH paper provide hands-on instruction of implementing ULFM with FTI, Reinit with FTI, and FTI with Restart to representative HPC proxy applications. Programmers can learn with less effort on how to implement the three fault tolerance designs to an HPC application through the MATCH code. 

- MATCH can also be a foundation for future MPI fault tolerance designs. Programmers can develop new MPI fault tolerance designs on top of the three fault tolerance designs. For example, the ULFM global non-shrinking recovery can be replaced with the ULFM local forward recovery; the FTI checkpointing can be replaced with the SCR checkpointing. We encourage programmers to add new HPC applications and new MPI fault tolerance techniques to MATCH.

# Installation
- Install ULFM v4.0.1ulfm2.1rc1 version following instructions from ULFM website. Add ULFM /bin/, /lib/, and /include/ directories to PATH, LD\_LIBRARY\_PATH, and CPATH respectively. PS: you need to install Open MPI 4.0.1 first, and enable Open MPI 4.0.1 the default Open MPI library, and then build ULFM on top of Open MPI 4.0.1.

- Install Reinit++ following installation instructions from Reinit++ website. Add Reinit++ /bin/, /lib/, and /include/ directories to PATH, LD\_LIBRARY\_PATH, and CPATH respectively. PS: turn off ULFM and Open MPI 4.0.1 before installation of Reinit++. Turn on Open MPI 4.0.0 before you install Reinit++. 

- Download the Git repo: git clone git@github.com:kakulo/MPI-FT-Bench.git

- There are multiple Git branches within our Git repo. You cannot find the up-to-date code in the main branch. The main branch is an early version of MATCH, which implements ULFM, Reinit++, and a self-developed checkpointing library to three HPC proxy applications (AMG, miniFE, and miniVite). The other three branches are restart-fti, reinit-fti, and ulfm-fti. The restart-fti branch implements FTI checkpointing and uses Restart for MPI recovery. Distinctively, reinit-fti and ulfm-fti branches use Reinit++ and ULFM for MPI recovery respectively. FTI requests users to pass a configuration file to the executable for checkpointing. Please refer to the FTI library guide for further instructions.

- Please note that you must add /bin/, /lib/, and /include/ directories to the environmental path of specific library when it is in use. For example, when ULFM and FTI are in use, you must remove Reinit++ from the environment path and enable ULFM and FTI to the environmental path. Last but not least, we provide a collection of SLURM job submission scripts in every example directory. Please refer to those scripts to develop your own execution script.  

- There are some flags used in our SLURM job scripts. To name a few, `config.L1.fti' is for FTI configuration; `-level' sets the checkpointing level; `-cp' sets the checkpointing stride; `-procfi' enables an MPI process failure; `-nodefi' enables an MPI node failure. ``-np 8 --mca orte_enable_recovery 1 --mca pml ^cm --mca plm_slurm_args "--wait=0" is for Reinit++ runtime.  

# Paper
If you use MPI-FT-Bench in your research, please cite our work.

Luanzheng Guo, Giorgis Georgakoudis, Konstantinos Parasyris, Ignacio Laguna, Dong Li. MATCH: An MPI Fault Tolerance Benchmark Suite. IEEE International Symposium on Workload Characterization (IISWC'20), Oct 2020


```
@inproceedings{guo2019match,
  title={MATCH: An MPI Fault Tolerance Benchmark Suite},
  author={Guo, Luanzheng and Georgakoudis, Giorgis and Parasyris, Konstantinos and Laguna, Ignacio and Li, Dong},
  booktitle={{IEEE International Symposium on Workload Characterization (IISWC)}},
  year={2020},
  organization={IEEE Press}
}
```

- Please reach out to [me](www.luanzhengguo.com) for any questions!

# Acknowledgments 
This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 (LLNL-CONF-812453). This research was supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. This research is partially supported by U.S. National Science Foundation (CNS-1617967, CCF-1553645 and CCF-1718194). We wish to thank the [NSF Trusted CI](https://www.trustedci.org), the NSF Cybersecurity Center of Excellence, NSF Grant Number ACI-1920430, for assisting our project with cybersecurity challenges.
