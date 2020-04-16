// ***********************************************************************
//
//                              miniVite
//
// ***********************************************************************
//
//       Copyright (2018) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 


#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

//#include <omp.h>
#include <mpi.h>
#include <mpi-ext.h>

#include "dspl.hpp"

/* for sleep function */
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

static std::string inputFileName;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static bool generateGraph = false;
static bool readBalanced = false;
static int randomEdgePercent = 0;
static bool randomNumberLCG = false;
static bool isUnitEdgeWeight = true;
static GraphWeight threshold = 1.0E-6;

// parse command line parameters
static void parseCommandLine(const int argc, char * const argv[]);

// FTI Protect for Graph
static void FTI_Protect_Graph(Graph &g);

int resilient_main(int argc, char** argv, OMPI_reinit_state_t state) ;


int main(int argc, char *argv[])
{
  int max_threads=0;

  //max_threads = omp_get_max_threads();

  if (max_threads > 1) {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
      if (provided < MPI_THREAD_FUNNELED) {
          std::cerr << "MPI library does not support MPI_THREAD_FUNNELED." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, -99);
      }
  } else {
      MPI_Init(&argc, &argv);
  }
  
  OMPI_Reinit(argc, argv, resilient_main);

  MPI_Finalize();

  return 0;
}

int resilient_main(int argc, char** argv, OMPI_reinit_state_t state) {

  double t0, t1, t2, t3, ti = 0.0;

if (enable_fti) {
    FTI_Init(argv[1], MPI_COMM_WORLD);
}

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  parseCommandLine(argc, argv);

   char hostname[65];
   gethostname(hostname, 65);
   printf("%s daemon %d rank %d\n", hostname, (int) getpid(), me);
   //sleep(5);

  createCommunityMPIType();
  double td0, td1, td, tdt;

  MPI_Barrier(MPI_COMM_WORLD);
  td0 = MPI_Wtime();

  Graph* g = nullptr;

  // generate graph only supports RGG as of now
  if (generateGraph) { 
      GenerateRGG gr(nvRGG);
      g = gr.generate(randomNumberLCG, isUnitEdgeWeight, randomEdgePercent);
      //g->print(false);
  }
  else { // read input graph
      BinaryEdgeList rm;
      if (readBalanced == true)
          g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName);
      else
          g = rm.read(me, nprocs, ranksPerNode, inputFileName);
      //g->print();
  }

  int recovered = 0;

  // code for FTI CPR
  if (enable_fti) {
      printf("Add FTI protection to Graph g ... \n");
      FTI_Protect_Graph(*g); 

      if ( FTI_Status() != 0){
	  printf("FTI recovery triggered for Graph g ... \n");
	  FTI_Recover();
	  recovered = 1;
      }
  }

  if (enable_fti) {
      if (!recovered) {
	  FTI_Checkpoint(0,level);
      }
  }
 
  assert(g != nullptr);
#ifdef PRINT_DIST_STATS 
  g->print_dist_stats();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_PRINTF  
  assert(g);
#endif  
  td1 = MPI_Wtime();
  td = td1 - td0;

  MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
  if (me == 0)  {
      if (!generateGraph)
          std::cout << "Time to read input file and create distributed graph (in s): " 
              << (tdt/nprocs) << std::endl;
      else
          std::cout << "Time to generate distributed graph of " 
              << nvRGG << " vertices (in s): " << (tdt/nprocs) << std::endl;
  }

  GraphWeight currMod = -1.0;
  GraphWeight prevMod = -1.0;
  double total = 0.0;

  std::vector<GraphElem> ssizes, rsizes, svdata, rvdata;
#if defined(USE_MPI_RMA)
  MPI_Win commwin;
#endif
  size_t ssz = 0, rsz = 0;
  int iters = 0;
    
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();

#if defined(USE_MPI_RMA)
  currMod = distLouvainMethod(me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters, commwin, state, level);
#else
  currMod = distLouvainMethod(me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters, state, level);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  
  if(me == 0) {
      std::cout << "Modularity: " << currMod << ", Iterations: " 
          << iters << ", Time (in s): "<<t0-t1<< std::endl;

      std::cout << "**********************************************************************" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double tot_time = 0.0;
  MPI_Reduce(&total, &tot_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  delete g;
  destroyCommunityMPIType();

if (enable_fti) {
    FTI_Finalize();
}

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  // new code for C/R implementation
  if(argc > 1) {
      int i = 1; 
      while (i<argc) {
	 int ok=0;
         if(strcmp(argv[i], "-CP") == 0) {
            if (i+1 >= argc) {
               printf("Missing integer argument to -cp\n");
            }
            ok = atoi(argv[i+1]);
            if(!ok) {
               printf("Parse Error on option -cp integer value required after argument - CP: %d \n", ok);
            }
            cp_stride=ok;
            i+=2;
         }
         else if(strcmp(argv[i], "-PROCFI") == 0) {
            procfi = 1;
            i++;
         }
         else if(strcmp(argv[i], "-NODEFI") == 0) {
            nodefi = 1;
            i++;
         }
         else if(strcmp(argv[i], "-CP2F") == 0) {
            cp2f = 1;
            i++;
         }
         else if(strcmp(argv[i], "-CP2M") == 0) {
            cp2m = 1;
            i++;
         }
         else if(strcmp(argv[i], "-CP2A") == 0) {
            cp2a = 1;
            i++;
         }
         else if(strcmp(argv[i], "-RESTART") == 0) {
            restart = 1;
            i++;
         }
         else if(strcmp(argv[i], "-LEVEL") == 0) {
            level = atoi(argv[i+1]);
            i+=2;
         }
	 else if(strcmp(argv[i], "config.L1.fti") == 0) {
	    i++;
	 }
	 else if(strcmp(argv[i], "-f") == 0) {	
	    inputFileName.assign(argv[i+1]);
	    i+=2;
	 }
         else if(strcmp(argv[i], "-b") == 0) {
            readBalanced = true;
            i++;
         }
         else if(strcmp(argv[i], "-r") == 0) {
            ranksPerNode = atoi(argv[i+1]);
            i+=2;
         }
         else if(strcmp(argv[i], "-t") == 0) {
            threshold = atof(argv[i+1]);
            i+=2;
         }
         else if(strcmp(argv[i], "-n") == 0) {
            nvRGG = atol(argv[i+1]);
	    if (nvRGG > 0)
		generateGraph = true;
            i+=2;
         }
         else if(strcmp(argv[i], "-w") == 0) {
            isUnitEdgeWeight = false;
            i++;
         }
         else if(strcmp(argv[i], "-l") == 0) {
            randomNumberLCG = true;
            i++;
         }
         else if(strcmp(argv[i], "-p") == 0) {
            randomEdgePercent = atoi(argv[i+1]);
            i+=2;
         }
         else {
	    i++;
         }
      }
  }
  // end of C/R code

/*
  while ((ret = getopt(argc, argv, "f:br:t:n:wlp:")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'b':
      readBalanced = true;
      break;
    case 'r':
      ranksPerNode = atoi(optarg);
      break;
    case 't':
      threshold = atof(optarg);
      break;
    case 'n':
      nvRGG = atol(optarg);
      if (nvRGG > 0)
          generateGraph = true; 
      break;
    case 'w':
      isUnitEdgeWeight = false;
      break;
    case 'l':
      randomNumberLCG = true;
      break;
    case 'p':
      randomEdgePercent = atoi(optarg);
      break;
    default:
      //assert(0 && "Should not reach here!!");
      break;
    }
  }
*/

  if (me == 0 && (argc == 1)) {
      std::cerr << "Must specify some options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && !generateGraph && inputFileName.empty()) {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
   
  if (me == 0 && !generateGraph && randomNumberLCG) {
      std::cerr << "Must specify -g for graph generation using LCG." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
   
  if (me == 0 && !generateGraph && randomEdgePercent) {
      std::cerr << "Must specify -g for graph generation first to add random edges to it." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
  
  if (me == 0 && !generateGraph && !isUnitEdgeWeight) {
      std::cerr << "Must specify -g for graph generation first before setting edge weights." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && generateGraph && ((randomEdgePercent < 0) || (randomEdgePercent >= 100))) {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine

// FTI protect for Graph
void FTI_Protect_Graph(Graph &g) {

  // Create a new FTI data type - FTI_EDGE
  FTIT_type FTI_EDGE;
  FTI_InitType(&FTI_EDGE, sizeof(GraphElem)+sizeof(GraphWeight)); 

  // protect edge_list_
  int size=g.edge_list_.size();
  FTI_Protect(0,&g.edge_list_[0],size,FTI_EDGE);

  // Create a new FTI data type - FTI_GraphElem
  FTIT_type FTI_GraphElem;
  FTI_InitType(&FTI_GraphElem, sizeof(GraphElem)); 

  // protect edge_indices_ 
  size=g.edge_indices_.size();
  FTI_Protect(1,&g.edge_indices_[0],size,FTI_GraphElem);

  // protect lnv_,lne_,nv_,ne_
  FTI_Protect(2,&g.lnv_,1,FTI_GraphElem);
  FTI_Protect(3,&g.lne_,1,FTI_GraphElem);
  FTI_Protect(4,&g.nv_,1,FTI_GraphElem);
  FTI_Protect(5,&g.ne_,1,FTI_GraphElem);

  // protect parts_
  size=g.parts_.size();
  FTI_Protect(6,&g.parts_[0],size,FTI_GraphElem);

} // FTI_Protect_Graph

