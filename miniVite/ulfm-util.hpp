#include <setjmp.h>

void InitULFM(char **argv);
void SetCommErrhandler();
int IsSurvivor();
void writeCheckpoint(int cp2f, int cp2m, int cp2a, int rank, int iter, char *data, int size);
int readCheckpoint(int survivor, int cp2f, int cp2m, int cp2a, int rank, char **data);
