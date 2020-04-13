/// \file
/// Compute forces for the Embedded Atom Model (EAM).

#ifndef __EAM_H
#define __EAM_H

#include "mytype.h"

struct BasePotentialSt;
struct LinkCellSt;

/// Pointers to the data that is needed in the load and unload functions
/// for the force halo exchange.
/// \see loadForceBuffer
/// \see unloadForceBuffer
typedef struct ForceExchangeDataSt
{
   real_t* dfEmbed; //<! derivative of embedding energy
   struct LinkCellSt* boxes;
}ForceExchangeData;

struct BasePotentialSt* initEamPot(const char* dir, const char* file, const char* type);

/// Write BasePotential object to checkpoint file
void writeEamPotential(char **data, struct BasePotentialSt* p, struct LinkCellSt* boxes);

/// Read BasePotential object from checkpoint file
struct BasePotentialSt* readEamPotential(char **data, struct LinkCellSt* boxes);

/// Get checkpoint size
size_t sizeofEamPotential(struct BasePotentialSt* p, struct LinkCellSt* boxes);
#endif
