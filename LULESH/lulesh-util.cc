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
            i+=2;
         }
	 else if(strcmp(argv[i], "config.L1.fti") == 0) {
	    i++;
	 }
         else {
            char msg[80];
            PrintCommandLineOptions(argv[0], myRank);
            sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
            ParseError(msg, myRank);
         }
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
// FTI Protect application states for checkpointing 

void FTI_Protect(Domain& locDom, struct cmdLineOpts &opts, double start) {
   // GG: block signals to avoid interruption by SIGREINIT
   /*sigset_t fullset;
   sigfillset(&fullset);
   sigprocmask(SIG_BLOCK, &fullset, NULL);*/

   int n = 0;

   int size;
   size = locDom.m_x.size();
   FTI_Protect(n,&locDom.m_x[0],size,FTI_DBLE);

   size = locDom.m_y.size();
   FTI_Protect(n+1,&locDom.m_y[0],size,FTI_DBLE);

   size = locDom.m_z.size();
   FTI_Protect(n+1,&locDom.m_z[0],size,FTI_DBLE);

   size = locDom.m_xd.size();
   FTI_Protect(n+1,&locDom.m_xd[0],size,FTI_DBLE);

   size = locDom.m_yd.size();
   FTI_Protect(n+1,&locDom.m_yd[0],size,FTI_DBLE);

   size = locDom.m_zd.size();
   FTI_Protect(n+1,&locDom.m_zd[0],size,FTI_DBLE);

   size = locDom.m_xdd.size();
   FTI_Protect(n+1,&locDom.m_xdd[0],size,FTI_DBLE);

   size = locDom.m_ydd.size();
   FTI_Protect(n+1,&locDom.m_ydd[0],size,FTI_DBLE);

   size = locDom.m_zdd.size();
   FTI_Protect(n+1,&locDom.m_zdd[0],size,FTI_DBLE);

   size = locDom.m_fx.size();
   FTI_Protect(n+1,&locDom.m_fx[0],size,FTI_DBLE);

   size = locDom.m_fy.size();
   FTI_Protect(n+1,&locDom.m_fy[0],size,FTI_DBLE);

   size = locDom.m_fz.size();
   FTI_Protect(n+1,&locDom.m_fz[0],size,FTI_DBLE);

   size = locDom.m_nodalMass.size();
   FTI_Protect(n+1,&locDom.m_nodalMass[0],size,FTI_DBLE);

   size = locDom.m_symmX.size();
   FTI_Protect(n+1,&locDom.m_symmX[0],size,FTI_INTG);

   size = locDom.m_symmY.size();
   FTI_Protect(n+1,&locDom.m_symmY[0],size,FTI_INTG);

   size = locDom.m_symmZ.size();
   FTI_Protect(n+1,&locDom.m_symmZ[0],size,FTI_INTG);

   FTI_Protect(n+1,&locDom.m_numReg,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_cost,1,FTI_INTG);
   FTI_Protect(n+1,locDom.m_regElemSize,locDom.m_numReg,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_numElem,1,FTI_INTG);
   FTI_Protect(n+1,locDom.m_regNumList,locDom.m_numElem,FTI_INTG);

   for(int i = 0; i < locDom.m_numReg; i++)
   {
      FTI_Protect(n+1,locDom.m_regElemlist[i],locDom.regElemSize(i),FTI_INTG);
   }

   size = locDom.m_nodelist.size();
   FTI_Protect(n+1,&locDom.m_nodelist[0],size,FTI_INTG);

   size = locDom.m_lxim.size();
   FTI_Protect(n+1,&locDom.m_lxim[0],size,FTI_INTG);

   size = locDom.m_lxip.size();
   FTI_Protect(n+1,&locDom.m_lxip[0],size,FTI_INTG);

   size = locDom.m_letam.size();
   FTI_Protect(n+1,&locDom.m_letam[0],size,FTI_INTG);

   size = locDom.m_letap.size();
   FTI_Protect(n+1,&locDom.m_letap[0],size,FTI_INTG);

   size = locDom.m_lzetam.size();
   FTI_Protect(n+1,&locDom.m_lzetam[0],size,FTI_INTG);

   size = locDom.m_lzetap.size();
   FTI_Protect(n+1,&locDom.m_lzetap[0],size,FTI_INTG);

   size = locDom.m_elemBC.size();
   FTI_Protect(n+1,&locDom.m_elemBC[0],size,FTI_INTG);

   size = locDom.m_dxx.size();
   FTI_Protect(n+1,&locDom.m_dxx[0],size,FTI_DBLE);

   size = locDom.m_dyy.size();
   FTI_Protect(n+1,&locDom.m_dyy[0],size,FTI_DBLE);

   size = locDom.m_dzz.size();
   FTI_Protect(n+1,&locDom.m_dzz[0],size,FTI_DBLE);

   size = locDom.m_delv_xi.size();
   FTI_Protect(n+1,&locDom.m_delv_xi[0],size,FTI_DBLE);

   size = locDom.m_delv_eta.size();
   FTI_Protect(n+1,&locDom.m_delv_eta[0],size,FTI_DBLE);

   size = locDom.m_delv_zeta.size();
   FTI_Protect(n+1,&locDom.m_delv_zeta[0],size,FTI_DBLE);

   size = locDom.m_delx_xi.size();
   FTI_Protect(n+1,&locDom.m_delx_xi[0],size,FTI_DBLE);

   size = locDom.m_delx_eta.size();
   FTI_Protect(n+1,&locDom.m_delx_eta[0],size,FTI_DBLE);

   size = locDom.m_delx_zeta.size();
   FTI_Protect(n+1,&locDom.m_delx_zeta[0],size,FTI_DBLE);

   size = locDom.m_e.size();
   FTI_Protect(n+1,&locDom.m_e[0],size,FTI_DBLE);

   size = locDom.m_p.size();
   FTI_Protect(n+1,&locDom.m_p[0],size,FTI_DBLE);

   size = locDom.m_q.size();
   FTI_Protect(n+1,&locDom.m_q[0],size,FTI_DBLE);

   size = locDom.m_ql.size();
   FTI_Protect(n+1,&locDom.m_ql[0],size,FTI_DBLE);

   size = locDom.m_qq.size();
   FTI_Protect(n+1,&locDom.m_qq[0],size,FTI_DBLE);

   size = locDom.m_v.size();
   FTI_Protect(n+1,&locDom.m_v[0],size,FTI_DBLE);

   size = locDom.m_volo.size();
   FTI_Protect(n+1,&locDom.m_volo[0],size,FTI_DBLE);

   size = locDom.m_vnew.size();
   FTI_Protect(n+1,&locDom.m_vnew[0],size,FTI_DBLE);

   size = locDom.m_delv.size();
   FTI_Protect(n+1,&locDom.m_delv[0],size,FTI_DBLE);

   size = locDom.m_vdov.size();
   FTI_Protect(n+1,&locDom.m_vdov[0],size,FTI_DBLE);

   size = locDom.m_arealg.size();
   FTI_Protect(n+1,&locDom.m_arealg[0],size,FTI_DBLE);

   size = locDom.m_ss.size();
   FTI_Protect(n+1,&locDom.m_ss[0],size,FTI_DBLE);

   size = locDom.m_elemMass.size();
   FTI_Protect(n+1,&locDom.m_elemMass[0],size,FTI_DBLE);

   FTI_Protect(n+1,&locDom.m_dtcourant,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_dthydro,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_cycle,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_dtfixed,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_time,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_deltatime,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_deltatimemultlb,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_deltatimemultub,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_dtmax,1,FTI_DBLE);
   FTI_Protect(n+1,&locDom.m_stoptime,1,FTI_DBLE);

   FTI_Protect(n+1,&locDom.m_numRanks,1,FTI_INTG);

   FTI_Protect(n+1,&locDom.m_colLoc,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_rowLoc,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_planeLoc,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_tp,1,FTI_INTG);

   FTI_Protect(n+1,&locDom.m_sizeX,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_sizeY,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_sizeZ,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_numElem,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_numNode,1,FTI_INTG);

   FTI_Protect(n+1,&locDom.m_maxPlaneSize,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_maxEdgeSize,1,FTI_INTG);

#if _OPENMP
   FTI_Protect(n+1,locDom.m_nodeElemStart,locDom.m_numNode+1,FTI_INTG);
   int elem = locDom.m_nodeElemStart[locDom.numNode()];
   FTI_Protect(n+1,locDom.m_nodeElemCornerList,elem,FTI_INTG);
#endif

   FTI_Protect(n+1,&locDom.m_rowMin,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_rowMax,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_colMin,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_colMax,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_planeMin,1,FTI_INTG);
   FTI_Protect(n+1,&locDom.m_planeMax,1,FTI_INTG);

   //Size of comm send/recv buffer
   FTI_Protect(n+1,&locDom.commBufSize,1,FTI_INTG);
   FTI_Protect(n+1,locDom.commDataSend,locDom.commBufSize,FTI_DBLE);
   FTI_Protect(n+1,locDom.commDataRecv,locDom.commBufSize,FTI_DBLE);

   //struct
   // define a new FTI type
   FTIT_type FTI_CMD;
   FTI_InitType(&FTI_CMD, 16*sizeof(int));
   FTI_Protect(n+1,&opts,1,FTI_CMD);

   //Time
#if USE_MPI
   FTI_Protect(n+1,&start,1,FTI_DBLE);
#endif

   //std::string sos = oss.str();

   //return sos;

   //size = oss.str().size();
   //write_cp(cp2f, cp2m, cp2a, rank, locDom.m_cycle, const_cast<char *>( oss.str().c_str() ), size, MPI_COMM_WORLD);

   //sigprocmask(SIG_UNBLOCK, &fullset, NULL);
}

