/*BHEADER**********************************************************************
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Ulrike Yang (yang11@llnl.gov) et al. CODE-LLNL-738-322.
 * This file is part of AMG.  See files README and COPYRIGHT for details.
 *
 * AMG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 ***********************************************************************EHEADER*/

/*
 * File:	timer.c
 * Author:	Scott Kohn (skohn@llnl.gov)
 * Description:	somewhat portable timing routines for C++, C, and Fortran
 *
 * If TIMER_USE_MPI is defined, then the MPI timers are used to get
 * wallclock seconds, since we assume that the MPI timers have better
 * resolution than the system timers.
 */

#include "_hypre_utilities.h"

#include <time.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/times.h>
#endif
#ifdef TIMER_USE_MPI
#include "mpi.h"
#endif

HYPRE_Real time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(hypre_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((HYPRE_Real) cl)/((HYPRE_Real) CLOCKS_PER_SEC));
#else
   struct tms usage;
   hypre_longint wallclock = times(&usage);
   return(((HYPRE_Real) wallclock)/((HYPRE_Real) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

HYPRE_Real time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return(((HYPRE_Real) (cpuclock))/((HYPRE_Real) CLOCKS_PER_SEC));
#else
   return(0.0);
#endif
}

HYPRE_Real time_get_wallclock_seconds_(void)
{
   return(time_getWallclockSeconds());
}

HYPRE_Real time_get_cpu_seconds_(void)
{
   return(time_getCPUSeconds());
}
