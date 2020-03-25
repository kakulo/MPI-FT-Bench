#ifndef _CHECKPOINT_H
#define _CHECKPOINT_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

void write_cp(int cp2f, int cp2m, int cp2a, int rank, int iter, char *data, int size, MPI_Comm comm);
int read_cp(int survivor, int cp2f, int cp2m, int cp2a, int rank, char **data, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif
