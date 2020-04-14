#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <signal.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "lulesh.h"
#include "../libcheckpoint/checkpoint.h"

/* Helper function for converting strings to ints, with error checking */
int StrToInt(const char *token, int *retVal)
{
   const char *c ;
   char *endptr ;
   const int decimal_base = 10 ;

   if (token == NULL)
      return 0 ;
   
   c = token ;
   *retVal = (int)strtol(c, &endptr, decimal_base) ;
   if((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1 ;
   else
      return 0 ;
}

static void PrintCommandLineOptions(char *execname, int myRank)
{
   if (myRank == 0) {

      printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf(" -cp             : Checkpoint frequency\n");
      printf("\n\n");
   }
}

static void ParseError(const char *message, int myRank)
{
   if (myRank == 0) {
      printf("%s\n", message);
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1);
#else
      exit(-1);
#endif
   }
}

void ParseCommandLineOptions(int argc, char *argv[],
                             int myRank, struct cmdLineOpts *opts)
{
   if(argc > 1) {
      int i = 1;

      while(i < argc) {
         int ok;
         /* -i <iterations> */
         if(strcmp(argv[i], "-i") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -i", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->its));
            if(!ok) {
               ParseError("Parse Error on option -i integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -s <size, sidelength> */
         else if(strcmp(argv[i], "-s") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -s\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->nx));
            if(!ok) {
               ParseError("Parse Error on option -s integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -r <numregions> */
         else if (strcmp(argv[i], "-r") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -r\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numReg));
            if (!ok) {
               ParseError("Parse Error on option -r integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -f <numfilepieces> */
         else if (strcmp(argv[i], "-f") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -f\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numFiles));
            if (!ok) {
               ParseError("Parse Error on option -f integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -p */
         else if (strcmp(argv[i], "-p") == 0) {
            opts->showProg = 1;
            i++;
         }
         /* -q */
         else if (strcmp(argv[i], "-q") == 0) {
            opts->quiet = 1;
            i++;
         }
         else if (strcmp(argv[i], "-b") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -b\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->balance));
            if (!ok) {
               ParseError("Parse Error on option -b integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-c") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -c\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->cost));
            if (!ok) {
               ParseError("Parse Error on option -c integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -v */
         else if (strcmp(argv[i], "-v") == 0) {
#if VIZ_MESH            
            opts->viz = 1;
#else
            ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
#endif
            i++;
         }
         /* -h */
         else if (strcmp(argv[i], "-h") == 0) {
            PrintCommandLineOptions(argv[0], myRank);
#if USE_MPI            
            MPI_Abort(MPI_COMM_WORLD, 0);
#else
            exit(0);
#endif
         }
         else if(strcmp(argv[i], "-cp") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -cp", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->cpInterval));
            if(!ok) {
               ParseError("Parse Error on option -cp integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if(strcmp(argv[i], "-procfi") == 0) {
            opts->procfi = 1;
            i++;
         }
         else if(strcmp(argv[i], "-nodefi") == 0) {
            opts->nodefi = 1;
            i++;
         }
         else if(strcmp(argv[i], "-cp2f") == 0) {
            opts->cp2f = 1;
            i++;
         }
         else if(strcmp(argv[i], "-cp2m") == 0) {
            opts->cp2m = 1;
            i++;
         }
         else if(strcmp(argv[i], "-cp2a") == 0) {
            opts->cp2a = 1;
            i++;
         }
         else if(strcmp(argv[i], "-level") == 0) {
            opts->level = atoi(argv[i+1]);
	 printf("leve is %d ... \n", opts->level);
            i+=2;
         }
         //else {
            //char msg[80];
            //PrintCommandLineOptions(argv[0], myRank);
            //sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
            //ParseError(msg, myRank);
         //}
      }
   }
}

/////////////////////////////////////////////////////////////////////

void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks)
{
   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism 
   Real_t grindTime1 = ((elapsed_time*1e6)/locDom.cycle())/(nx*nx*nx);
   Real_t grindTime2 = ((elapsed_time*1e6)/locDom.cycle())/(nx*nx*nx*numRanks);

   Index_t ElemId = 0;
   printf("Run completed:  \n");
   printf("   Problem size        =  %i \n",    nx);
   printf("   MPI tasks           =  %i \n",    numRanks);
   printf("   Iteration count     =  %i \n",    locDom.cycle());
   printf("   Final Origin Energy = %12.6e \n", locDom.e(ElemId));

   Real_t   MaxAbsDiff = Real_t(0.0);
   Real_t TotalAbsDiff = Real_t(0.0);
   Real_t   MaxRelDiff = Real_t(0.0);

   for (Index_t j=0; j<nx; ++j) {
      for (Index_t k=j+1; k<nx; ++k) {
         Real_t AbsDiff = FABS(locDom.e(j*nx+k)-locDom.e(k*nx+j));
         TotalAbsDiff  += AbsDiff;

         if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

         Real_t RelDiff = AbsDiff / locDom.e(k*nx+j);

         if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
      }
   }

   // Quick symmetry check
   printf("   Testing Plane 0 of Energy Array on rank 0:\n");
   printf("        MaxAbsDiff   = %12.6e\n",   MaxAbsDiff   );
   printf("        TotalAbsDiff = %12.6e\n",   TotalAbsDiff );
   printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff   );

   // Timing information
   printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
   printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n", grindTime1, grindTime2);
   printf("FOM                  = %10.8g (z/s)\n\n", 1000.0/grindTime2); // zones per second

   return ;
}

/////////////////////////////////////////////////////////
// Write application state in checkpoint file

std::stringstream& ApplicationCheckpointWrite(Domain& locDom, struct cmdLineOpts &opts, double start) {
   // GG: block signals to avoid interruption by SIGREINIT
   /*sigset_t fullset;
   sigfillset(&fullset);
   sigprocmask(SIG_BLOCK, &fullset, NULL);*/

   std::stringstream oss;

   int size;
   size = locDom.m_x.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_x[i]), sizeof(Real_t));
   }

   size = locDom.m_y.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_y[i]), sizeof(Real_t));
   }

   size = locDom.m_z.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_z[i]), sizeof(Real_t));
   }

   size = locDom.m_xd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_xd[i]), sizeof(Real_t));
   }

   size = locDom.m_yd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_yd[i]), sizeof(Real_t));
   }

   size = locDom.m_zd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_zd[i]), sizeof(Real_t));
   }

   size = locDom.m_xdd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_xdd[i]), sizeof(Real_t));
   }

   size = locDom.m_ydd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_ydd[i]), sizeof(Real_t));
   }

   size = locDom.m_zdd.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_zdd[i]), sizeof(Real_t));
   }

   size = locDom.m_fx.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_fx[i]), sizeof(Real_t));
   }

   size = locDom.m_fy.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_fy[i]), sizeof(Real_t));
   }

   size = locDom.m_fz.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_fz[i]), sizeof(Real_t));
   }

   size = locDom.m_nodalMass.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_nodalMass[i]), sizeof(Real_t));
   }

   size = locDom.m_symmX.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_symmX[i]), sizeof(Index_t));
   }

   size = locDom.m_symmY.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_symmY[i]), sizeof(Index_t));
   }

   size = locDom.m_symmZ.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_symmZ[i]), sizeof(Index_t));
   }

   oss.write(reinterpret_cast<char *>(&locDom.m_numReg), sizeof(Int_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_cost), sizeof(Int_t));
   oss.write(reinterpret_cast<char *>(locDom.m_regElemSize), sizeof(Index_t) * locDom.m_numReg);
   oss.write(reinterpret_cast<char *>(&locDom.m_numElem), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(locDom.m_regNumList), sizeof(Index_t) * locDom.m_numElem);

   for(int i = 0; i < locDom.m_numReg; i++)
   {
      oss.write(reinterpret_cast<char *>(locDom.m_regElemlist[i]), sizeof(Index_t) * locDom.regElemSize(i));
   }

   size = locDom.m_nodelist.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_nodelist[i]), sizeof(Index_t));
   }

   size = locDom.m_lxim.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_lxim[i]), sizeof(Index_t));
   }

   size = locDom.m_lxip.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_lxip[i]), sizeof(Index_t));
   }

   size = locDom.m_letam.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_letam[i]), sizeof(Index_t));
   }

   size = locDom.m_letap.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_letap[i]), sizeof(Index_t));
   }

   size = locDom.m_lzetam.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_lzetam[i]), sizeof(Index_t));
   }

   size = locDom.m_lzetap.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_lzetap[i]), sizeof(Index_t));
   }

   size = locDom.m_elemBC.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_elemBC[i]), sizeof(Int_t));
   }

   size = locDom.m_dxx.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_dxx[i]), sizeof(Real_t));
   }

   size = locDom.m_dyy.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_dyy[i]), sizeof(Real_t));
   }

   size = locDom.m_dzz.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_dzz[i]), sizeof(Real_t));
   }

   size = locDom.m_delv_xi.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delv_xi[i]), sizeof(Real_t));
   }

   size = locDom.m_delv_eta.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delv_eta[i]), sizeof(Real_t));
   }

   size = locDom.m_delv_zeta.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delv_zeta[i]), sizeof(Real_t));
   }

   size = locDom.m_delx_xi.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delx_xi[i]), sizeof(Real_t));
   }

   size = locDom.m_delx_eta.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delx_eta[i]), sizeof(Real_t));
   }

   size = locDom.m_delx_zeta.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delx_zeta[i]), sizeof(Real_t));
   }

   size = locDom.m_e.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_e[i]), sizeof(Real_t));
   }

   size = locDom.m_p.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_p[i]), sizeof(Real_t));
   }

   size = locDom.m_q.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_q[i]), sizeof(Real_t));
   }

   size = locDom.m_ql.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_q[i]), sizeof(Real_t));
   }

   size = locDom.m_qq.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_qq[i]), sizeof(Real_t));
   }

   size = locDom.m_v.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_v[i]), sizeof(Real_t));
   }

   size = locDom.m_volo.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_volo[i]), sizeof(Real_t));
   }

   size = locDom.m_vnew.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_vnew[i]), sizeof(Real_t));
   }

   size = locDom.m_delv.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_delv[i]), sizeof(Real_t));
   }

   size = locDom.m_vdov.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_vdov[i]), sizeof(Real_t));
   }

   size = locDom.m_arealg.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_arealg[i]), sizeof(Real_t));
   }

   size = locDom.m_ss.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_ss[i]), sizeof(Real_t));
   }

   size = locDom.m_elemMass.size();
   oss.write(reinterpret_cast<char *>(&size), sizeof(int));
   for(int i = 0; i < size; i++) {
      oss.write(reinterpret_cast<char *>(&locDom.m_elemMass[i]), sizeof(Real_t));
   }

   oss.write(reinterpret_cast<char *>(&locDom.m_dtcourant), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_dthydro), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_cycle), sizeof(Int_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_dtfixed), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_time), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_deltatime), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_deltatimemultlb), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_deltatimemultub), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_dtmax), sizeof(Real_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_stoptime), sizeof(Real_t));

   oss.write(reinterpret_cast<char *>(&locDom.m_numRanks), sizeof(Int_t));

   oss.write(reinterpret_cast<char *>(&locDom.m_colLoc), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_rowLoc), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_planeLoc), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_tp), sizeof(Index_t));

   oss.write(reinterpret_cast<char *>(&locDom.m_sizeX), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_sizeY), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_sizeZ), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_numElem), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_numNode), sizeof(Index_t));

   oss.write(reinterpret_cast<char *>(&locDom.m_maxPlaneSize), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_maxEdgeSize), sizeof(Index_t));

#if _OPENMP
   oss.write(reinterpret_cast<char *>(locDom.m_nodeElemStart), sizeof(Index_t) * (locDom.m_numNode + 1));
   int elem = locDom.m_nodeElemStart[locDom.numNode()];
   oss.write(reinterpret_cast<char *>(locDom.m_nodeElemCornerList), sizeof(Index_t) * elem);
#endif

   oss.write(reinterpret_cast<char *>(&locDom.m_rowMin), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_rowMax), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_colMin), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_colMax), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_planeMin), sizeof(Index_t));
   oss.write(reinterpret_cast<char *>(&locDom.m_planeMax), sizeof(Index_t));

   //Size of comm send/recv buffer
   oss.write(reinterpret_cast<char *>(&locDom.commBufSize), sizeof(int));
   oss.write(reinterpret_cast<char *>(locDom.commDataSend), sizeof(Real_t) * locDom.commBufSize);
   oss.write(reinterpret_cast<char *>(locDom.commDataRecv), sizeof(Real_t) * locDom.commBufSize);

   //struct
   oss.write(reinterpret_cast<char *>(&opts), sizeof(opts));

   //Time
#if USE_MPI
   oss.write(reinterpret_cast<char *>(&start), sizeof(double));
#endif

   return oss;

   //size = oss.str().size();
   //write_cp(cp2f, cp2m, cp2a, rank, locDom.m_cycle, const_cast<char *>( oss.str().c_str() ), size, MPI_COMM_WORLD);

   //sigprocmask(SIG_UNBLOCK, &fullset, NULL);
}

/////////////////////////////////////////////////////////
// Read application state from checkpoint file

void ApplicationCheckpointRead(Domain& locDom, struct cmdLineOpts &opts, double &start, std::stringstream& iss) {
   //char *data;
   //size_t sizeofCP = read_cp(survivor, cp2f, cp2m, cp2a, rank, &data, MPI_COMM_WORLD);

   //std::stringstream iss( std::string( data, data + sizeofCP ), std::stringstream::in | std::stringstream::binary );

   int sz;
   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_x.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_x[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_y.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_y[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_z.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_z[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_xd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_xd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_yd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_yd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_zd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_zd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_xdd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_xdd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_ydd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_ydd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_zdd.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_zdd[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_fx.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_fx[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_fy.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_fy[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_fz.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_fz[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_nodalMass.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_nodalMass[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_symmX.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_symmX[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_symmY.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_symmY[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_symmZ.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_symmZ[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&locDom.m_numReg), sizeof(Int_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_cost), sizeof(Int_t));

   locDom.m_regElemSize = new Index_t[locDom.m_numReg];
   iss.read(reinterpret_cast<char *>(locDom.m_regElemSize), sizeof(Index_t) * locDom.m_numReg);

   Index_t nElem;
   iss.read(reinterpret_cast<char *>(&nElem), sizeof(Index_t));
   locDom.m_regNumList = new Index_t[nElem];
   iss.read(reinterpret_cast<char *>(locDom.m_regNumList), sizeof(Index_t) * nElem);

   locDom.m_regElemlist = new Index_t*[locDom.m_numReg];
   for (int i = 0; i < locDom.m_numReg; i++) {
      locDom.m_regElemlist[i] = new Index_t[locDom.regElemSize(i)];
   }
   for(int i = 0; i < locDom.m_numReg; i++)
   {
      iss.read(reinterpret_cast<char *>(locDom.m_regElemlist[i]), sizeof(Index_t) * locDom.regElemSize(i));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_nodelist.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_nodelist[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_lxim.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_lxim[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_lxip.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_lxip[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_letam.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_letam[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_letap.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_letap[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_lzetam.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_lzetam[i]), sizeof(Index_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_lzetap.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_lzetap[i]), sizeof(Index_t));
   }


   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_elemBC.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_elemBC[i]), sizeof(Int_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_dxx.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_dxx[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_dyy.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_dyy[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_dzz.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_dzz[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delv_xi.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delv_xi[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delv_eta.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delv_eta[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delv_zeta.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delv_zeta[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delx_xi.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delx_xi[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delx_eta.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delx_eta[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delx_zeta.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delx_zeta[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_e.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_e[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_p.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_p[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_q.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_q[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_ql.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_ql[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_qq.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_qq[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_v.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_v[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_volo.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_volo[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_vnew.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_vnew[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_delv.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_delv[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_vdov.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_vdov[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_arealg.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_arealg[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_ss.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_ss[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&sz), sizeof(int));
   locDom.m_elemMass.resize(sz);
   for (int i = 0; i < sz; i++) {
      iss.read(reinterpret_cast<char *>(&locDom.m_elemMass[i]), sizeof(Real_t));
   }

   iss.read(reinterpret_cast<char *>(&locDom.m_dtcourant), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_dthydro), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_cycle), sizeof(Int_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_dtfixed), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_time), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_deltatime), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_deltatimemultlb), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_deltatimemultub), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_dtmax), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_stoptime), sizeof(Real_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_numRanks), sizeof(Int_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_colLoc), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_rowLoc), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_planeLoc), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_tp), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_sizeX), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_sizeY), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_sizeZ), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_numElem), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_numNode), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_maxPlaneSize), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_maxEdgeSize), sizeof(Index_t));

#if _OPENMP
   locDom.m_nodeElemStart = new Index_t[locDom.m_numNode + 1];
   iss.read(reinterpret_cast<char *>(locDom.m_nodeElemStart), sizeof(Index_t) * (locDom.m_numNode + 1));
   int elem = locDom.m_nodeElemStart[locDom.numNode()];
   locDom.m_nodeElemCornerList = new Index_t[elem];
   iss.read(reinterpret_cast<char *>(locDom.m_nodeElemCornerList), sizeof(Index_t) * elem);
#endif

   iss.read(reinterpret_cast<char *>(&locDom.m_rowMin), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_rowMax), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_colMin), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_colMax), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_planeMin), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.m_planeMax), sizeof(Index_t));
   iss.read(reinterpret_cast<char *>(&locDom.commBufSize), sizeof(int));
   locDom.commDataSend = new Real_t[locDom.commBufSize];
   iss.read(reinterpret_cast<char *>(locDom.commDataSend), sizeof(Real_t) * locDom.commBufSize);
   locDom.commDataRecv = new Real_t[locDom.commBufSize];
   iss.read(reinterpret_cast<char *>(locDom.commDataRecv), sizeof(Real_t) * locDom.commBufSize);

   //struct
   iss.read(reinterpret_cast<char *>(&opts), sizeof(opts));

   // time
#if USE_MPI
   iss.read(reinterpret_cast<char *>(&start), sizeof(double));
#endif

   //free( data );
}
