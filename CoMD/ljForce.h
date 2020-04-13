/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.

#ifndef _LJTYPES_H_
#define _LJTYPES_H_

#include <stdio.h>

struct BasePotentialSt;
struct BasePotentialSt* initLjPot(void);

/// Write Base Potential structure to file
void writeLJForce(char **data, struct BasePotentialSt* pot);

/// Read Base Potential structure from file
struct BasePotentialSt* readLJForce(char **data);

/// Get checkpoint size
size_t sizeofLJForce();

#endif

