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

#include <omp.h>
#include <mpi.h>

#include "dspl.hpp"

/* for sleep function */
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

/* ULFM */
#include <setjmp.h>
#include "ulfm-util.hpp"

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

// write checkpoints for Graph
static void GraphCheckpointWrite(Graph &g, int rank);

// read checkpoints for Graph
static void GraphCheckpointRead(int survivor, int rank, Graph &g);

/* ULFM: world will swap between worldc[0] and worldc[1] after each respawn */
extern MPI_Comm worldc[2];
extern int worldi;
#define world (worldc[worldi])

/* ULFM */
extern jmp_buf stack_jmp_buf;


int main(int argc, char *argv[])
{
  double t0, t1, t2, t3, ti = 0.0;
  int max_threads;

  max_threads = omp_get_max_threads();

  if (max_threads > 1) {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
      if (provided < MPI_THREAD_FUNNELED) {
          std::cerr << "MPI library does not support MPI_THREAD_FUNNELED." << std::endl;
          MPI_Abort(world, -99);
      }
  } else {
      MPI_Init(&argc, &argv);
  }

/* ULFM */
  InitULFM(argv);
  
  MPI_Comm_size(world, &nprocs);
  MPI_Comm_rank(world, &me);

  parseCommandLine(argc, argv);

restart:
  int do_recover = _setjmp(stack_jmp_buf);
  /* We set an errhandler on world, so that a failure is not fatal anymore. */
  SetCommErrhandler();


   char hostname[65];
   gethostname(hostname, 65);
   printf("%s daemon %d rank %d\n", hostname, (int) getpid(), me);
   sleep(45);

  createCommunityMPIType();
  double td0, td1, td, tdt;

  MPI_Barrier(world);
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

  // read graph from checkpoints
  // Read checkpointing either because of recovery being a survivor
  int survivor = IsSurvivor();
  if (do_recover || !survivor) {
     printf("RE-Start execution ... \n");
     printf("Read checkpoint graph data ... \n");
     GraphCheckpointRead(survivor,me,*g);
  }
  // end of 
  // reading graph from checkpoints

  // code for C/R (1)
  // write graph to checkpoints
  if (cp_stride>0 && restart==0) {
      printf("Write checkpoint graph data ... \n");
      GraphCheckpointWrite(*g,me); 
  }
  // end of 
  // writing graph to checkpoints  
 
  assert(g != nullptr);
#ifdef PRINT_DIST_STATS 
  g->print_dist_stats();
#endif

  MPI_Barrier(world);
#ifdef DEBUG_PRINTF  
  assert(g);
#endif  
  td1 = MPI_Wtime();
  td = td1 - td0;

  MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_SUM, 0, world);
 
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
    
  MPI_Barrier(world);

  t1 = MPI_Wtime();

#if defined(USE_MPI_RMA)
  currMod = distLouvainMethod(do_recover, survivor, me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters, commwin);
#else
  currMod = distLouvainMethod(do_recover, survivor, me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters);
#endif
  MPI_Barrier(world);
  t0 = MPI_Wtime();
  
  if(me == 0) {
      std::cout << "Modularity: " << currMod << ", Iterations: " 
          << iters << ", Time (in s): "<<t0-t1<< std::endl;

      std::cout << "**********************************************************************" << std::endl;
  }

  MPI_Barrier(world);

  double tot_time = 0.0;
  MPI_Reduce(&total, &tot_time, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  
  delete g;
  destroyCommunityMPIType();

  MPI_Finalize();

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

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

  // new code for C/R implementation
  if(argc > 1) {
      int i = 1; 
      while (i<argc) {
	 int ok;
         if(strcmp(argv[i], "-CP") == 0) {
            if (i+1 >= argc) {
               printf("Missing integer argument to -cp\n");
            }
            ok = atoi(argv[i+1]);
            if(!ok) {
               printf("Parse Error on option -cp integer value required after argument\n");
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
         else {
            //printf("ERROR: Unknown command line argument: %s\n", argv[i]);
            i++;
         }
      }
  }
  // end of C/R code


  if (me == 0 && (argc == 1)) {
      std::cerr << "Must specify some options." << std::endl;
      MPI_Abort(world, -99);
  }
  
  if (me == 0 && !generateGraph && inputFileName.empty()) {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      MPI_Abort(world, -99);
  }
   
  if (me == 0 && !generateGraph && randomNumberLCG) {
      std::cerr << "Must specify -g for graph generation using LCG." << std::endl;
      MPI_Abort(world, -99);
  } 
   
  if (me == 0 && !generateGraph && randomEdgePercent) {
      std::cerr << "Must specify -g for graph generation first to add random edges to it." << std::endl;
      MPI_Abort(world, -99);
  } 
  
  if (me == 0 && !generateGraph && !isUnitEdgeWeight) {
      std::cerr << "Must specify -g for graph generation first before setting edge weights." << std::endl;
      MPI_Abort(world, -99);
  }
  
  if (me == 0 && generateGraph && ((randomEdgePercent < 0) || (randomEdgePercent >= 100))) {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      MPI_Abort(world, -99);
  }
} // parseCommandLine

void GraphCheckpointWrite(Graph &g, int rank) {

  std::stringstream oss( std::stringstream::out | std::stringstream::binary); 

  // checkpoint edge_list_
  int size;
  size=g.edge_list_.size();
  oss.write(reinterpret_cast<char *>(&size),sizeof(int));
  for (int i=0;i<size;i++) {
     oss.write(reinterpret_cast<char *>(&g.edge_list_[i].tail_),sizeof(GraphElem));
     oss.write(reinterpret_cast<char *>(&g.edge_list_[i].weight_),sizeof(GraphWeight));
  }
  
  // checkpoint edge_indices_ 
  size=g.edge_indices_.size();
  oss.write(reinterpret_cast<char *>(&size),sizeof(int));
  for (int i=0;i<size;i++) {
     oss.write(reinterpret_cast<char *>(&g.edge_indices_[i]),sizeof(GraphElem));
  }

  // checkpoint lnv_
  GraphElem lnvv=g.get_lnv();
  oss.write(reinterpret_cast<char *>(&lnvv),sizeof(GraphElem));

  // checkpoint lne_
  GraphElem lnee=g.get_lne();
  oss.write(reinterpret_cast<char *>(&lnee),sizeof(GraphElem));

  // checkpoint nv_
  GraphElem nvv=g.get_nv();
  oss.write(reinterpret_cast<char *>(&nvv),sizeof(GraphElem));

  // checkpoint ne_
  GraphElem nee=g.get_ne();
  oss.write(reinterpret_cast<char *>(&nee),sizeof(GraphElem));

  // checkpoint parts_
  size=g.parts_.size();
  oss.write(reinterpret_cast<char *>(&size),sizeof(int));
  for (int i=0;i<size;i++) {
     oss.write(reinterpret_cast<char *>(&g.parts_[i]),sizeof(GraphElem));
  }

  size = oss.str().size();

  write_cp_0(cp2f, cp2m, cp2a, rank, -1, const_cast<char *>( oss.str().c_str() ), size, world);
} // GraphCheckpointWrite

static void GraphCheckpointRead(int survivor, int rank, Graph &g) {

  char* data;
  // -1 means the first position to C/R outside the iteration
  size_t sizeofCP=read_cp_0(survivor, cp2f, cp2m, cp2a, rank, &data, world, -1);
  std::stringstream iss(std::string( data, data + sizeofCP ), std::stringstream::in | std::stringstream::binary );

  // checkpoint edge_list_
  int size;
  iss.read(reinterpret_cast<char *>(&size),sizeof(int));
  g.edge_list_.resize(size);
  for (int i=0;i<size;i++) {
     iss.read(reinterpret_cast<char *>(&g.edge_list_[i].tail_),sizeof(GraphElem));
     iss.read(reinterpret_cast<char *>(&g.edge_list_[i].weight_),sizeof(GraphWeight));
  }
  
  // checkpoint edge_indices_ 
  iss.read(reinterpret_cast<char *>(&size),sizeof(int));
  g.edge_indices_.resize(size);
  for (int i=0;i<size;i++) {
     iss.read(reinterpret_cast<char *>(&g.edge_indices_[i]),sizeof(GraphElem));
  }

  // checkpoint lnv_
  GraphElem lnv_new;
  iss.read(reinterpret_cast<char *>(&lnv_new),sizeof(GraphElem));
  g.set_lnv(lnv_new);

  // checkpoint lne_
  GraphElem lne_new;
  iss.read(reinterpret_cast<char *>(&lne_new),sizeof(GraphElem));
  g.set_lne(lne_new);

  // checkpoint nv_
  GraphElem nv_new;
  iss.read(reinterpret_cast<char *>(&nv_new),sizeof(GraphElem));
  g.set_nv(nv_new);

  // checkpoint ne_
  GraphElem ne_new;
  iss.read(reinterpret_cast<char *>(&ne_new),sizeof(GraphElem));
  g.set_ne(ne_new);

  // checkpoint parts_
  iss.read(reinterpret_cast<char *>(&size),sizeof(int));
  g.parts_.resize(size);
  for (int i=0;i<size;i++) {
     iss.read(reinterpret_cast<char *>(&g.parts_[i]),sizeof(GraphElem));
  }
} // GraphCheckpointRead


