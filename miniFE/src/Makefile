#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.


FTI_INC = -I/g/g90/guo7/mylibs/FTI/include/
FTI_LIB = -L/g/g90/guo7/mylibs/FTI/lib64/ -lfti

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL="long long int"

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
#MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

# DEBUG
DEBUG = 1

# GNU 
CFLAGS = -g -O0 -fopenmp
ifeq ($(DEBUG), 1) 
CFLAGS += -DDEBUG
endif

CFLAGS += -std=c++11

# BG/Q
# CFLAGS = -DIBM_HPM -qlist -qsource  -g -O3 -qsmp=omp
# CFLAGS = -qlist -qsource  -g -O3 -qsmp=omp
CXXFLAGS = $(CFLAGS)

# To enable memory high water mark reporting uncomment the
# report rusage
CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) \
	$(FTI_INC) $(MINIFE_MATRIX_TYPE) \
	-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK #-DMINIFE_REPORT_RUSAGE

LDFLAGS= -g -O0 $(CFLAGS) $(FTI_LIB)
LIBS=

# GNU 
CXX=mpicxx
# GNU 
CC=mpicc

# BG/Q
# CXX=mpixlcxx_r
# CC=mpixlc_r


include make_targets
