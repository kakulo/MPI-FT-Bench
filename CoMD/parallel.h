/// \file
/// Wrappers for MPI functions.

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "mytype.h"

/// Structure for use with MPI_MINLOC and MPI_MAXLOC operations.
typedef struct RankReduceDataSt
{
   double val;
   int rank;
} RankReduceData;

/// Return total number of processors.
int getNRanks(void);

/// Return local rank.
int getMyRank(void);

/// Return non-zero if printing occurs from this rank.
int printRank(void);

/// Print a timestamp and message when all tasks arrive.
void timestampBarrier(const char* msg);

/// Wrapper for MPI_Init.
void initParallel(int *argc, char ***argv);

/// Wrapper for MPI_Finalize.
void destroyParallel(void);

/// Wrapper for MPI_Barrier(MPI_COMM_WORLD).
void barrierParallel(void);

/// Wrapper for MPI_Sendrecv.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source);

/// Wrapper for MPI_Allreduce integer sum.
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce real sum.
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// Wrapper for MPI_Allreduce double sum.
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// Wrapper for MPI_Allreduce integer max.
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce double min with rank.
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Allreduce double max with rank.
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Bcast
void bcastParallel(void* buf, int len, int root);

/// Set the UFLM errhandler on the world comm
void setCommErrhandler();

/// Returns whether the rank on the world comm is a survivor
int isSurvivor();

/// Write a checkpoint (possibly involving communication)
void writeCheckpoint(int cp2f, int cp2m, int cp2a, int rank, int iter, char *data, int size);

/// Read a checkpoint (possible involving communication)
int readCheckpoint(int survivor, int cp2f, int cp2m, int cp2a, int rank, char **data);

///  Return non-zero if code was built with MPI active.
int builtWithMpi(void);

///  FTI Initialization
void ftiInitialParallel(char** argv);

///  FTI Finalization
void ftifinalizeParallel();


#endif

