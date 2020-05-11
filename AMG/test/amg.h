#ifndef AMG_HEADER
#define AMG_HEADER

/* new variables for C/R */
extern int cp_stride;
extern int procfi;
extern int nodefi;
extern int cp2f;
extern int cp2m;
extern int cp2a;
extern int restart;
extern int nprocs;
extern int myrank;
extern int survivor;
extern int level;
#ifdef TIMER
extern double acc_write_time;
#endif

#endif
