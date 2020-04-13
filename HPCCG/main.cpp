
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// read_HPC_row - Reads in linear system

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <csetjmp>
#include <cstring>
#include <string>
#include <cmath>
#ifdef USING_MPI
#include <mpi.h> // If this routine is compiled with -DUSING_MPI
                 // then include mpi.h
#include <mpi-ext.h>
#include "make_local_matrix.hpp" // Also include this function
#endif
#ifdef USING_OMP
#include <omp.h>
#endif
#include "generate_matrix.hpp"
#include "read_HPC_row.hpp"
#include "mytimer.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"
#include "dump_matlab_matrix.hpp"

#include "YAML_Element.hpp"
#include "YAML_Doc.hpp"

#undef DEBUG

HPC_Sparse_Matrix *A;
double *x, *b, *xexact;
double norm, d;
int ierr = 0;
int i, j;
int ione = 1;
double times[7];
double t6 = 0.0;
int nx,ny,nz;
bool do_restart = false;
int cp_iters = 1;
bool cp2f = false, cp2m = false, cp2a = false;
bool procfi = false, nodefi = false;

int niters = 0;
double normr = 0.0;
int max_iter = 150;
double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations

int resilient_main(int argc, char **argv, OMPI_reinit_state_t state)
{
  if( OMPI_REINIT_REINITED == state ) {
#ifdef USING_MPI

    // Transform matrix indices from global to local values.
    // Define number of columns for the local matrix.

    // XXX: Assumes generated matrix
    generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
    t6 = mytimer(); make_local_matrix(A);  t6 = mytimer() - t6;
    times[6] = t6;

#endif

  }

  ierr = HPCCG( A, b, x, max_iter, tolerance, niters, normr, times,
          state, cp_iters, cp2f, cp2m, cp2a, procfi, nodefi );

  return 0;
}

int main(int argc, char *argv[])
{

#ifdef USING_MPI

  MPI_Init(&argc, &argv);

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //  if (size < 100) cout << "Process "<<rank<<" of "<<size<<" is alive." <<endl;

#else

  int size = 1; // Serial case (not using MPI)
  int rank = 0;

#endif


#ifdef DEBUG
  if (rank==0)
   {
    int junk = 0;
    cout << "Press enter to continue"<< endl;
    cin >> junk;
   }

  MPI_Barrier(MPI_COMM_WORLD);
#endif


  if(argc<3) {
    if (rank==0)
      cerr << "Usage:" << endl
	   << "Mode 1: " << argv[0] << " nx ny nz" << endl
	   << "     where nx, ny and nz are the local sub-block dimensions, or" << endl
	   << "Mode 2: " << argv[0] << " HPC_data_file " << endl
	   << "     where HPC_data_file is a globally accessible file containing matrix data." << endl
	   << "Both modes take a mandatory extra arguments related to checkpointing:" << endl
           <<  "     checkpoint frequency = # iters (0 means no checkpoint) >" << endl
           << "There are three optional arguments to select checkpointing mode (valid combinations permitted)" << endl
           << "     -cp2f (file CP) -cp2m (memory CP) -cp2a (adjacent rank CP)" << endl
           << "There are also two optional arguments for fault injection" << endl
           << "     -procfi (enables random process FI) -nodefi (enables random node FI)" << endl;
    exit(1);
  }

  if (argc>=5)
  {
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
    cp_iters = atoi(argv[4]);
    i = 5;
  }
  else
  {
    assert(false && "File input is not supported at this version!\n");
    read_HPC_row(argv[1], &A, &x, &b, &xexact);
    cp_iters = atoi(argv[2]);
    i = 3;
  }

  // Parse optional arguments
  while( i < argc ) {
    if( !strcmp("-restart", argv[i]) )
      do_restart = true;
    else if( !strcmp("-cp2f", argv[i]) )
      cp2f = true;
    else if( !strcmp("-cp2m", argv[i]) )
      cp2m = true;
    else if( !strcmp("-cp2a", argv[i]) )
      cp2a = true;
    else if( !strcmp("-procfi", argv[i]) )
      procfi = true;
    else if( !strcmp("-nodefi", argv[i]) )
      nodefi = true;

    i++;
  }

  if( do_restart )
    assert( cp2f && !cp2m && !cp2a && "Restarting job support ONLY file checkpoints!\n");

  // Check checkpointing options are valid
  if( cp_iters ) {
    assert( cp2f || ( cp2f && cp2m ) || ( cp2m && cp2a ) && "Requesting checkpoints with invalid or unsupported checkpoint mode!\n" );
    if( cp2m && cp2a )
      assert( size>1 && "Cannot do memory+adjacent checkpoint with a single rank!\n");
  }

  // Assert only one or none enabled FI method
  assert( ! (procfi && nodefi ) && "Cannot perform both procfi and nodefi" );

  bool dump_matrix = false;
  if (dump_matrix && size<=4) dump_matlab_matrix(A, rank);

#ifdef USING_MPI

  // Transform matrix indices from global to local values.
  // Define number of columns for the local matrix.

  t6 = mytimer(); make_local_matrix(A);  t6 = mytimer() - t6;
  times[6] = t6;

#endif

  double t1 = mytimer();   // Initialize it (if needed)
  OMPI_Reinit(0, 0, resilient_main);
  if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;

#ifdef USING_MPI
      double t4 = times[4];
      double t4min = 0.0;
      double t4max = 0.0;
      double t4avg = 0.0;
      MPI_Allreduce(&t4, &t4min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&t4, &t4avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      t4avg = t4avg/((double) size);
#endif

// initialize YAML doc

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      double fniters = niters;
      double fnrow = A->total_nrow;
      double fnnz = A->total_nnz;
      double fnops_ddot = fniters*4*fnrow;
      double fnops_waxpby = fniters*6*fnrow;
      double fnops_sparsemv = fniters*2*fnnz;
      double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv;

      YAML_Doc doc("hpccg", "1.0");

      doc.add("Parallelism","");

#ifdef USING_MPI
          doc.get("Parallelism")->add("Number of MPI ranks",size);
#else
          doc.get("Parallelism")->add("MPI not enabled","");
#endif

#ifdef USING_OMP
          int nthreads = 1;
#pragma omp parallel
          nthreads = omp_get_num_threads();
          doc.get("Parallelism")->add("Number of OpenMP threads",nthreads);
#else
          doc.get("Parallelism")->add("OpenMP not enabled","");
#endif

      doc.add("Dimensions","");
	  doc.get("Dimensions")->add("nx",nx);
	  doc.get("Dimensions")->add("ny",ny);
	  doc.get("Dimensions")->add("nz",nz);



      doc.add("Number of iterations", niters);
      doc.add("Final residual", normr);
      doc.add("#********** Performance Summary (times in sec) ***********","");

      doc.add("Time Summary","");
      doc.get("Time Summary")->add("Total   ",times[0]);
      doc.get("Time Summary")->add("DDOT    ",times[1]);
      doc.get("Time Summary")->add("WAXPBY  ",times[2]);
      doc.get("Time Summary")->add("SPARSEMV",times[3]);

      doc.add("FLOPS Summary","");
      doc.get("FLOPS Summary")->add("Total   ",fnops);
      doc.get("FLOPS Summary")->add("DDOT    ",fnops_ddot);
      doc.get("FLOPS Summary")->add("WAXPBY  ",fnops_waxpby);
      doc.get("FLOPS Summary")->add("SPARSEMV",fnops_sparsemv);

      doc.add("MFLOPS Summary","");
      doc.get("MFLOPS Summary")->add("Total   ",fnops/times[0]/1.0E6);
      doc.get("MFLOPS Summary")->add("DDOT    ",fnops_ddot/times[1]/1.0E6);
      doc.get("MFLOPS Summary")->add("WAXPBY  ",fnops_waxpby/times[2]/1.0E6);
      doc.get("MFLOPS Summary")->add("SPARSEMV",fnops_sparsemv/(times[3])/1.0E6);

#ifdef USING_MPI
      doc.add("DDOT Timing Variations","");
      doc.get("DDOT Timing Variations")->add("Min DDOT MPI_Allreduce time",t4min);
      doc.get("DDOT Timing Variations")->add("Max DDOT MPI_Allreduce time",t4max);
      doc.get("DDOT Timing Variations")->add("Avg DDOT MPI_Allreduce time",t4avg);

      double totalSparseMVTime = times[3] + times[5]+ times[6];
      doc.add("SPARSEMV OVERHEADS","");
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV MFLOPS W OVERHEAD",fnops_sparsemv/(totalSparseMVTime)/1.0E6);
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Time", (times[5]+times[6]));
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Pct", (times[5]+times[6])/totalSparseMVTime*100.0);
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Setup Time", (times[6]));
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Setup Pct", (times[6])/totalSparseMVTime*100.0);
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Bdry Exch Time", (times[5]));
      doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Bdry Exch Pct", (times[5])/totalSparseMVTime*100.0);
#endif

      if (rank == 0) { // only PE 0 needs to compute and report timing results
        std::string yaml = doc.generateYAML();
        cout << yaml;
       }
    }

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.

  double residual = 0;
  //  if ((ierr = compute_residual(A->local_nrow, x, xexact, &residual)))
  //  cerr << "Error in call to compute_residual: " << ierr << ".\n" << endl;

  // if (rank==0)
  //   cout << "Difference between computed and exact  = " 
  //        << residual << ".\n" << endl;


  // Finish up
#ifdef USING_MPI
  MPI_Finalize();
#endif
  return 0 ;
}