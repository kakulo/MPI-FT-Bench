/// \file
/// Wrappers for memory allocation.

#ifndef _MEMUTILS_H_
#define _MEMUTILS_H_

#include <stdlib.h>
#include <string.h>

#define freeMe(s,element) {if(s->element) comdFree(s->element);  s->element = NULL;}

static void* comdMalloc(size_t iSize)
{
   return malloc(iSize);
}

static void mwrite(void *src, size_t size, size_t n_iters, char **data)
{
   memcpy(*data, src, size * n_iters);
   *data += ( size * n_iters );
}

static void mread(void *dst, size_t size, size_t n_iters, char **data)
{
   memcpy(dst, *data, size * n_iters);
   *data += ( size * n_iters );
}

static void* comdCalloc(size_t num, size_t iSize)
{
   return calloc(num, iSize);
}

static void* comdRealloc(void* ptr, size_t iSize)
{
   return realloc(ptr, iSize);
}

static void comdFree(void *ptr)
{
   free(ptr);
}
#endif
