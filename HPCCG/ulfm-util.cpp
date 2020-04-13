#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>
#include "ulfm-util.hpp"
#include "../libcheckpoint/checkpoint.h"
#include <sys/time.h>

static char **gargv;
static int do_recover;
static int survivor;
jmp_buf stack_jmp_buf;
static MPI_Errhandler errh;

static int last_dead = MPI_PROC_NULL;
static int rank = MPI_PROC_NULL, verbose = 1; /* makes this global (for printfs) */
static char estr[MPI_MAX_ERROR_STRING]=""; static int strl; /* error messages */

/* world will swap between worldc[0] and worldc[1] after each respawn */
MPI_Comm worldc[2] = { MPI_COMM_NULL, MPI_COMM_NULL };
int worldi = 0;
#define world (worldc[worldi])

static int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm *newcomm)
{
    MPI_Comm icomm, /* the intercomm between the spawnees and the old (shrinked) world */
        scomm, /* the local comm for each sides of icomm */
        mcomm; /* the intracomm, merged from icomm */
    MPI_Group cgrp, sgrp, dgrp;
    int rc, flag, rflag, i, nc, ns, nd, crank, srank, drank;

 redo:
    if( comm == MPI_COMM_NULL ) { /* am I a new process? */
        /* I am a new spawnee, waiting for my new rank assignment
         * it will be sent by rank 0 in the old world */
        MPI_Comm_get_parent(&icomm);
        scomm = MPI_COMM_WORLD;
        MPI_Recv(&crank, 1, MPI_INT, 0, 1, icomm, MPI_STATUS_IGNORE);
        if( verbose ) {
            MPI_Comm_rank(scomm, &srank);
            printf("Spawnee %d: crank=%d\n", srank, crank);
        }
    } else {
        /* I am a survivor: Spawn the appropriate number
         * of replacement processes (we check that this operation worked
         * before we procees further) */
        /* First: remove dead processes */
        MPIX_Comm_shrink(comm, &scomm);
        MPI_Comm_size(scomm, &ns);
        MPI_Comm_size(comm, &nc);
        nd = nc-ns; /* number of deads */
        if( 0 == nd ) {
            /* Nobody was dead to start with. We are done here */
            MPI_Comm_free(&scomm);
            *newcomm = comm;
            return MPI_SUCCESS;
        }
        /* We handle failures during this function ourselves... */
        MPI_Comm_set_errhandler( scomm, MPI_ERRORS_RETURN );

        rc = MPI_Comm_spawn(gargv[0], &gargv[1], nd, MPI_INFO_NULL,
                            0, scomm, &icomm, MPI_ERRCODES_IGNORE);
        flag = (MPI_SUCCESS == rc);
        MPIX_Comm_agree(scomm, &flag);
        if( !flag ) {
            if( MPI_SUCCESS == rc ) {
                MPIX_Comm_revoke(icomm);
                MPI_Comm_free(&icomm);
            }
            MPI_Comm_free(&scomm);
            if( verbose ) fprintf(stderr, "%04d: comm_spawn failed, redo\n", rank);
            goto redo;
        }

        /* remembering the former rank: we will reassign the same
         * ranks in the new world. */
        MPI_Comm_rank(comm, &crank);
        MPI_Comm_rank(scomm, &srank);
        /* the rank 0 in the scomm comm is going to determine the
         * ranks at which the spares need to be inserted. */
        if(0 == srank) {
            /* getting the group of dead processes:
             *   those in comm, but not in scomm are the deads */
            MPI_Comm_group(comm, &cgrp);
            MPI_Comm_group(scomm, &sgrp);
            MPI_Group_difference(cgrp, sgrp, &dgrp);
            /* Computing the rank assignment for the newly inserted spares */
            for(i=0; i<nd; i++) {
                MPI_Group_translate_ranks(dgrp, 1, &i, cgrp, &drank);
                /* sending their new assignment to all new procs */
                MPI_Send(&drank, 1, MPI_INT, i, 1, icomm);
                last_dead = drank;
            }
            MPI_Group_free(&cgrp); MPI_Group_free(&sgrp); MPI_Group_free(&dgrp);
        }
    }

    /* Merge the intercomm, to reconstruct an intracomm (we check
     * that this operation worked before we proceed further) */
    rc = MPI_Intercomm_merge(icomm, 1, &mcomm);
    rflag = flag = (MPI_SUCCESS==rc);
    MPIX_Comm_agree(scomm, &flag);
    if( MPI_COMM_WORLD != scomm ) MPI_Comm_free(&scomm);
    MPIX_Comm_agree(icomm, &rflag);
    MPI_Comm_free(&icomm);
    if( !(flag && rflag) ) {
        if( MPI_SUCCESS == rc ) {
            MPI_Comm_free(&mcomm);
        }
        if( verbose ) fprintf(stderr, "%04d: Intercomm_merge failed, redo\n", rank);
        goto redo;
    }

    /* Now, reorder mcomm according to original rank ordering in comm
     * Split does the magic: removing spare processes and reordering ranks
     * so that all surviving processes remain at their former place */
    rc = MPI_Comm_split(mcomm, 1, crank, newcomm);

    /* Split or some of the communications above may have failed if
     * new failures have disrupted the process: we need to
     * make sure we succeeded at all ranks, or retry until it works. */
    flag = (MPI_SUCCESS==rc);
    MPIX_Comm_agree(mcomm, &flag);
    MPI_Comm_free(&mcomm);
    if( !flag ) {
        if( MPI_SUCCESS == rc ) {
            MPI_Comm_free( newcomm );
        }
        if( verbose ) fprintf(stderr, "%04d: comm_split failed, redo\n", rank);
        goto redo;
    }

    /* restore the error handler */
    if( MPI_COMM_NULL != comm ) {
        MPI_Errhandler errh;
        MPI_Comm_get_errhandler( comm, &errh );
        MPI_Comm_set_errhandler( *newcomm, errh );
    }
    printf("Done with the recovery (rank %d)\n", crank);

    return MPI_SUCCESS;
}
/* repair comm world, reload checkpoints, etc...
 *  Return: true: the app needs to redo some iterations
 *          false: no failure was fixed, we do not need to redo any work.
 */
static int app_needs_repair(MPI_Comm comm)
{
    /* This is the first time we see an error on this comm, do the swap of the
     * worlds. Next time we will have nothing to do. */
    if( comm == world ) {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        /* swap the worlds */
        worldi = (worldi+1)%2;
        /* We keep comm around so that the error handler remains attached until the
         * user has completed all pending ops; it is expected that the user will
         * complete all ops on comm before posting new ops in the new world.
         * Beware that if the user does not complete all ops on comm and the handler
         * is invoked on the new world inbetween, comm may be freed while
         * operations are still pending on it, and a fatal error may be
         * triggered when these ops are finally completed (possibly in Finalize)*/
        if( MPI_COMM_NULL != world ) MPI_Comm_free(&world);
        MPIX_Comm_replace(comm, &world);
        if( MPI_COMM_NULL == comm ) return false; /* ok, we repaired nothing, no need to redo any work */

        gettimeofday(&end, NULL);
        double dtime = (double)( end.tv_sec - start.tv_sec ) + ( end.tv_usec - start.tv_usec ) / 1000000.0;
        printf("TIME RECOVER app %lf s %d\n", dtime, rank );

        _longjmp( stack_jmp_buf, 1 );
    }
    return true; /* we have repaired the world, we need to reexecute */
}
/* Do all the magic in the error handler */
static void errhandler_respawn(MPI_Comm* pcomm, int* errcode, ...)
{
    int eclass;
    MPI_Error_class(*errcode, &eclass);

    if( verbose ) {
        MPI_Error_string(*errcode, estr, &strl);
        fprintf(stderr, "%04d: errhandler invoked with error %s\n", rank, estr);
    }

    if( MPIX_ERR_PROC_FAILED != eclass &&
        MPIX_ERR_REVOKED != eclass ) {
        MPI_Abort(MPI_COMM_WORLD, *errcode);
    }

    struct timeval tv;
    gettimeofday( &tv, NULL );
    double ts = tv.tv_sec + tv.tv_usec / 1000000.0;
    printf("TIMESTAMP DETECT %lf s rank %d\n", ts, rank );

    MPIX_Comm_revoke(world);

    app_needs_repair(world);
}

void InitULFM(char **argv)
{
    gargv = argv;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm parent;
    MPI_Comm_create_errhandler(&errhandler_respawn, &errh);
    /* Am I a spare ? */
    MPI_Comm_get_parent( &parent );
    if( MPI_COMM_NULL == parent ) {
        /* First run: Let's create an initial world,
         * a copy of MPI_COMM_WORLD */
        MPI_Comm_dup( MPI_COMM_WORLD, &world );
        survivor = 1;
    } else {
        survivor = 0;
        /* I am a spare, lets get the repaired world */
        app_needs_repair(MPI_COMM_NULL);
    }
}

void SetCommErrhandler()
{
  MPI_Comm_set_errhandler( world, errh );
}


void writeCheckpoint(int cp2f, int cp2m, int cp2a, int rank, int iter, char *data, int size)
{
    write_cp( cp2f, cp2m, cp2a, rank, iter, data, size, world );
}

int readCheckpoint(int survivor, int cp2f, int cp2m, int cp2a, int rank, char **data)
{
    return read_cp( survivor, cp2f, cp2m, cp2a, rank, data, world );
}

int IsSurvivor()
{
    return survivor;
}
