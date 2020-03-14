#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include "checkpoint.h"

typedef struct {
    int iter;
    int size;
    char *data;
} cp_data_t;

cp_data_t cp = { -1, 0, 0},  cp_prev = { -1, 0, 0 };
cp_data_t cp_adj = { -1 , 0, 0 },  cp_adj_prev = { -1, 0, 0 };

void write_cp(int cp2f, int cp2m, int cp2a, int rank, int iter, char *data, int size, MPI_Comm comm)
{
  if( cp2m ) {
    free( cp_prev.data );
    cp_prev = cp;
    cp.data = malloc( sizeof(char) * size );
    memcpy( cp.data, data, size );
    cp.iter = iter;
    cp.size = size;
  }

  if( cp2a ) {
    int nranks;
    MPI_Comm_size( comm, &nranks );
    int prev_rank = ( ( rank == 0 ) ? ( nranks - 1 ) : ( rank - 1 ) );
    int next_rank = ( rank+1 ) % nranks;
    MPI_Request req;
    MPI_Status status;

    // Send data to neighbor
    MPI_Isend( &size, 1, MPI_INT, next_rank, 0, comm, &req );
    printf("Rank %d isend size %d to rank %d\n", rank, size, next_rank);
    MPI_Isend( data, size, MPI_CHAR, next_rank, 1, comm, &req );
    MPI_Isend( &iter, 1, MPI_INT, next_rank, 2, comm, &req );

    free( cp_adj_prev.data );
    cp_adj_prev = cp_adj;
    cp_adj.iter = INT_MAX;

    // Recv data, first the size to allocate
    MPI_Recv( &cp_adj.size, 1, MPI_INT, prev_rank, 0, comm, &status );
    printf("Rank %d recv size %d from rank %d\n", rank, cp_adj.size, prev_rank);
    cp_adj.data = malloc( sizeof(char) * cp_adj.size );
    MPI_Recv( cp_adj.data, cp_adj.size, MPI_CHAR, prev_rank, 1, comm, &status );
    MPI_Recv( &cp_adj.iter, 1, MPI_INT, prev_rank, 2, comm, &status );
  }

  if( cp2f ) {
    // XXX: cp.iter, c_prev are NOT updated when reading TODO
    int del_iter = cp_prev.iter;
    cp_prev.iter = cp.iter;
    cp.iter = iter;
    char filename[64];
    printf("del_iter %d prev_iter %d iter %d\n", del_iter, cp_prev.iter , iter); //ggout
    if( 0 <= del_iter ) {
      printf("Remove %d\n", del_iter ); //ggout
      sprintf( filename, "check_%d_%d", rank, del_iter );
      remove( filename );
    }
    sprintf( filename, "check_%d_%d", rank, iter);
    FILE *fp = fopen( filename, "wb" );
    assert( NULL != fp );
    fwrite( &size, sizeof(int), 1, fp );
    fwrite( data, size, 1, fp );
    fclose( fp );

    sprintf( filename, "tmp_%d", rank );
    fp = fopen( filename, "wb" );
    assert( NULL != fp );
    fwrite( (char *)&iter, sizeof(int), 1, fp);
    fclose( fp );
  }
}

int read_cp(int survivor, int cp2f, int cp2m, int cp2a, int rank, char **data, MPI_Comm comm)
{
  // Reading a checkpoint goes through two phases:
  // 1. Getting the last consistent checkpoint through a global reduction
  // 2. Reading the checkpoint from a valid source (depending on the checkpointing mode)

  int size = 0;

  if( cp2m ) {
    if( survivor ) {
      // read my last iter to reduce
      int global_iter, local_iter = cp.iter;
      // XXX: cp2a assumes a single failure
      // if cp2a enabled, reduce also with the iter of the remote CP
      if( cp2a ) {
	int nranks;
	MPI_Comm_size( comm, &nranks );
	int local_iter_adj = ( cp_adj.iter <= cp_adj_prev.iter ) ? cp_adj.iter : cp_adj_prev.iter;
	local_iter = ( local_iter <= local_iter_adj ) ? local_iter : local_iter_adj;
      }

      MPI_Allreduce(&local_iter, &global_iter, 1, MPI_INT, MPI_MIN, comm);

      printf("Rank %d READCP mem\n", rank); //ggout
      if( global_iter == cp.iter ) {
	*data = malloc( cp.size );
	size = cp.size;
	memcpy(*data, cp.data, cp.size);
      }
      else if( global_iter == cp_prev.iter ) {
	*data = malloc( cp_prev.size );
	size = cp_prev.size;
	memcpy(*data, cp_prev.data, cp_prev.size);
      }
      else {
	fprintf( stderr, "Invalid iter %d to read_cp\n", global_iter );
	assert(0);
      }

      // If adjacent is enabled, send cp to neighbor rank
      if( cp2a ) {
	int nranks;
	MPI_Comm_size( comm, &nranks );
	int prev_rank = ( rank == 0 ) ? ( nranks - 1 ) : ( rank - 1 );
	int next_rank = ( rank + 1 ) % nranks;
	MPI_Request req;
	MPI_Status status;
	int next_survivor;

	// Send my survivor status to the rank it checkpoints me
	MPI_Isend( &survivor, 1, MPI_INT, next_rank, 3, comm, &req);

	// Receive the survivor status of the rank I checkpoint
	MPI_Recv( &next_survivor, 1, MPI_INT, prev_rank, 3, comm, &status);

	// Send checkpoint *only* if the rank I checkpoint is respawned
	if( !next_survivor ) {
	  MPI_Request req;
	  if( global_iter == cp_adj.iter ) {
	    MPI_Isend( &cp_adj.size, 1, MPI_INT, prev_rank, 4, comm, &req);
	    MPI_Isend( cp_adj.data, cp_adj.size, MPI_CHAR, prev_rank, 5, comm, &req);
	  }
	  else if( global_iter == cp_adj_prev.iter ) {
	    MPI_Isend( &cp_adj_prev.size, 1, MPI_INT, prev_rank, 4, comm, &req);
	    MPI_Isend( cp_adj_prev.data, cp_adj_prev.size, MPI_CHAR, prev_rank, 5, comm, &req);
	  }
	  else {
	    fprintf( stderr, "Invalid iter %d to Isend\n", global_iter);
	    assert(0);
	  }
	}
      }
    }
    else {
      if( cp2a ) {
	int nranks;
	MPI_Comm_size( comm, &nranks );
	int prev_rank = ( rank == 0 ) ? ( nranks - 1 ) : ( rank - 1 );
	int next_rank = ( rank + 1 ) % nranks;
	MPI_Request req;
	MPI_Status status;
	int global_iter, max_int = INT_MAX;
	int next_survivor;

	// I'm a respawned rank so I reduce with INT_MAX to get global_iter from the comm
	MPI_Allreduce(&max_int, &global_iter, 1, MPI_INT, MPI_MIN, comm);

	// Send my survivor status to the rank it checkpoints me
	MPI_Isend( &survivor, 1, MPI_INT, next_rank, 3, comm, &req);

	// Receive the survivor status of the rank I checkpoint
	// Not needed but avoids buffer leaks
	MPI_Recv( &next_survivor, 1, MPI_INT, prev_rank, 3, comm, &status);

	// Receive the CP from my adjacent rank
	MPI_Recv( &size, 1, MPI_INT, next_rank, 4, comm, &status );
	*data = malloc( size );
	MPI_Recv( *data, size, MPI_CHAR, next_rank, 5, comm, &status );
	printf("Rank %d READCP adj\n", rank); //ggout
      }
      else if( cp2f ) {
	int global_iter, local_iter;
	char filename[64];
	sprintf( filename, "tmp_%d", rank);
	FILE *fp = fopen( filename, "rb" );
	assert( NULL != fp );
	fread( (char *)&local_iter, sizeof(int), 1, fp );
	fclose( fp );

	MPI_Allreduce(&local_iter, &global_iter, 1, MPI_INT, MPI_MIN, comm);

	sprintf( filename, "check_%d_%d", rank, global_iter);
	fp = fopen( filename, "rb" );
	assert( NULL != fp );
	fread( &size, sizeof(int), 1, fp );
	*data = malloc( size );
	fread( *data, size, 1, fp );
	fclose( fp );
	printf("Rank %d READCP file\n", rank); //ggout
      }
      else
	assert(0&& "Cannot load checkpoint!\n");
    }
  }
  else if( cp2f ) {
    int global_iter, local_iter;
    char filename[64];
    sprintf( filename, "tmp_%d", rank);
    FILE *fp = fopen( filename, "rb" );
    assert( NULL != fp );
    fread( (char *)&local_iter, sizeof(int), 1, fp );
    fclose( fp );

    MPI_Allreduce(&local_iter, &global_iter, 1, MPI_INT, MPI_MIN, comm);

    sprintf( filename, "check_%d_%d", rank, global_iter);
    fp = fopen( filename, "rb" );
    assert( NULL != fp );
    fread( &size, sizeof(int), 1, fp );
    *data = malloc( size );
    fread( *data, size, 1, fp );
    fclose( fp );
    printf("Rank %d READCP file\n", rank); //ggout
  }
  else
    assert(0&& "Cannot load checkpoint!\n");

  return size;
}
