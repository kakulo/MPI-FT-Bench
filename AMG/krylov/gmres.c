/*BHEADER**********************************************************************
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Ulrike Yang (yang11@llnl.gov) et al. CODE-LLNL-738-322.
 * This file is part of AMG.  See files README and COPYRIGHT for details.
 *
 * AMG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "seq_mv.h"
#include "gmres.h"
#include "../test/amg.h"

//#include "par_vector.h"
#include <sstream>
#include <sys/time.h>
#include <fstream>
#include <signal.h>
#include <unistd.h>
#include "../../libcheckpoint/checkpoint.h"

using namespace std;

//static void AMGCheckpointWrite(HYPRE_Int iter,HYPRE_Real *rs,HYPRE_Real *c,HYPRE_Real *s,HYPRE_Real **hh,HYPRE_Real epsilon,HYPRE_Int max_iter,HYPRE_Real epsmac,void *p,HYPRE_Int *precond_data,HYPRE_Real b_norm,HYPRE_Real *norms, HYPRE_Real r_norm_0, void *A, void *x, void *w, void *b, HYPRE_Int rank,HYPRE_Int k_dim);
//static void AMGCheckpointRead(HYPRE_Int &iter,HYPRE_Real *rs,HYPRE_Real *c,HYPRE_Real *s,HYPRE_Real **hh,HYPRE_Real &epsilon,HYPRE_Int &max_iter,HYPRE_Real &epsmac,void *p,HYPRE_Int *precond_data,HYPRE_Real &b_norm,HYPRE_Real *norms, HYPRE_Real &r_norm_0, void *A, void *x, void *w, void *b, HYPRE_Int rank, HYPRE_Int survivor,HYPRE_Int &k_dim); 

/*--------------------------------------------------------------------------
 * hypre_GMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_GMRESFunctions *
hypre_GMRESFunctionsCreate(
   char *       (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_GMRESFunctions * gmres_functions;
   gmres_functions = (hypre_GMRESFunctions *)
      CAlloc( 1, sizeof(hypre_GMRESFunctions) );

   gmres_functions->CAlloc = CAlloc;
   gmres_functions->Free = Free;
   gmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   gmres_functions->CreateVector = CreateVector;
   gmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   gmres_functions->DestroyVector = DestroyVector;
   gmres_functions->MatvecCreate = MatvecCreate;
   gmres_functions->Matvec = Matvec;
   gmres_functions->MatvecDestroy = MatvecDestroy;
   gmres_functions->InnerProd = InnerProd;
   gmres_functions->CopyVector = CopyVector;
   gmres_functions->ClearVector = ClearVector;
   gmres_functions->ScaleVector = ScaleVector;
   gmres_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   gmres_functions->precond_setup = PrecondSetup;
   gmres_functions->precond       = Precond;

   return gmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions )
{
   hypre_GMRESData *gmres_data;
 
   gmres_data = hypre_CTAllocF(hypre_GMRESData, 1, gmres_functions);

   gmres_data->functions = gmres_functions;
 
   /* set defaults */
   (gmres_data -> k_dim)          = 5;
   (gmres_data -> tol)            = 1.0e-06; /* relative residual tol */
   (gmres_data -> cf_tol)         = 0.0;
   (gmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (gmres_data -> min_iter)       = 0;
   (gmres_data -> max_iter)       = 1000;
   (gmres_data -> rel_change)     = 0;
   (gmres_data -> skip_real_r_check) = 0;
   (gmres_data -> stop_crit)      = 0; /* rel. residual norm  - this is obsolete!*/
   (gmres_data -> converged)      = 0;
   (gmres_data -> precond_data)   = NULL;
   (gmres_data -> print_level)    = 0;
   (gmres_data -> logging)        = 0;
   (gmres_data -> p)              = NULL;
   (gmres_data -> r)              = NULL;
   (gmres_data -> w)              = NULL;
   (gmres_data -> w_2)            = NULL;
   (gmres_data -> matvec_data)    = NULL;
   (gmres_data -> norms)          = NULL;
   (gmres_data -> log_file_name)  = NULL;
 
   return (void *) gmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESDestroy
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESDestroy( void *gmres_vdata )
{
	hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;
   HYPRE_Int i;
 
   if (gmres_data)
   {
      hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
      if ( (gmres_data->logging>0) || (gmres_data->print_level) > 0 )
      {
         if ( (gmres_data -> norms) != NULL )
            hypre_TFreeF( gmres_data -> norms, gmres_functions );
      }
 
      if ( (gmres_data -> matvec_data) != NULL )
         (*(gmres_functions->MatvecDestroy))(gmres_data -> matvec_data);
 
      if ( (gmres_data -> r) != NULL )
         (*(gmres_functions->DestroyVector))(gmres_data -> r);
      if ( (gmres_data -> w) != NULL )
         (*(gmres_functions->DestroyVector))(gmres_data -> w);
      if ( (gmres_data -> w_2) != NULL )
         (*(gmres_functions->DestroyVector))(gmres_data -> w_2);


      if ( (gmres_data -> p) != NULL )
      {
         for (i = 0; i < (gmres_data -> k_dim+1); i++)
         {
            if ( (gmres_data -> p)[i] != NULL )
	       (*(gmres_functions->DestroyVector))( (gmres_data -> p) [i]);
         }
         hypre_TFreeF( gmres_data->p, gmres_functions );
      }
      hypre_TFreeF( gmres_data, gmres_functions );
      hypre_TFreeF( gmres_functions, gmres_functions );
   }
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_GMRESGetResidual( void *gmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */

   hypre_GMRESData  *gmres_data     = (hypre_GMRESData *)gmres_vdata;
   *residual = gmres_data->r;
   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetup
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetup( void *gmres_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_GMRESData *gmres_data     = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;

   HYPRE_Int            k_dim            = (gmres_data -> k_dim);
   HYPRE_Int            max_iter         = (gmres_data -> max_iter);
   HYPRE_Int          (*precond_setup)(void*,void*,void*,void*) = (gmres_functions->precond_setup);
   void          *precond_data     = (gmres_data -> precond_data);

   HYPRE_Int            rel_change       = (gmres_data -> rel_change);
   

 
   (gmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((gmres_data -> p) == NULL)
	   (gmres_data -> p) = (void**)(*(gmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((gmres_data -> r) == NULL)
      (gmres_data -> r) = (*(gmres_functions->CreateVector))(b);
   if ((gmres_data -> w) == NULL)
      (gmres_data -> w) = (*(gmres_functions->CreateVector))(b);
 
   if (rel_change)
   {  
      if ((gmres_data -> w_2) == NULL)
         (gmres_data -> w_2) = (*(gmres_functions->CreateVector))(b);
   }
   

   if ((gmres_data -> matvec_data) == NULL)
      (gmres_data -> matvec_data) = (*(gmres_functions->MatvecCreate))(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ( (gmres_data->logging)>0 || (gmres_data->print_level) > 0 )
   {
      if ((gmres_data -> norms) == NULL)
         (gmres_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1,gmres_functions);
   }
   if ( (gmres_data->print_level) > 0 ) {
      if ((gmres_data -> log_file_name) == NULL)
		  (gmres_data -> log_file_name) = (char*)"gmres.out.log";
   }
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSolve(void  *gmres_vdata,
                 void  *A,
                 void  *b,
		 void  *x,
		 OMPI_reinit_state_t state)
{
  printf("Enter the hypre_GMRESSolve function ... \n");
   hypre_GMRESData  *gmres_data   = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   HYPRE_Int 		     k_dim        = (gmres_data -> k_dim);
   HYPRE_Int               min_iter     = (gmres_data -> min_iter);
   HYPRE_Int 		     max_iter     = (gmres_data -> max_iter);
   HYPRE_Int               rel_change   = (gmres_data -> rel_change);
   HYPRE_Int         skip_real_r_check  = (gmres_data -> skip_real_r_check);
   HYPRE_Real 	     r_tol        = (gmres_data -> tol);
   HYPRE_Real 	     cf_tol       = (gmres_data -> cf_tol);
   HYPRE_Real        a_tol        = (gmres_data -> a_tol);
   void             *matvec_data  = (gmres_data -> matvec_data);

   void             *r            = (gmres_data -> r);
   void             *w            = (gmres_data -> w);
   /* note: w_2 is only allocated if rel_change = 1 */
   void             *w_2          = (gmres_data -> w_2); 

   void            **p            = (gmres_data -> p);


   HYPRE_Int 	           (*precond)(void*,void*,void*,void*)   = (gmres_functions -> precond);
   HYPRE_Int 	            *precond_data = (HYPRE_Int*)(gmres_data -> precond_data);

   HYPRE_Int             print_level    = (gmres_data -> print_level);
   HYPRE_Int             logging        = (gmres_data -> logging);

   HYPRE_Real     *norms          = (gmres_data -> norms);
/* not used yet   char           *log_file_name  = (gmres_data -> log_file_name);*/
/*   FILE           *fp; */
   
   HYPRE_Int        break_value = 0;
   HYPRE_Int	      i, j, k;
   HYPRE_Real *rs, **hh, *c, *s, *rs_2; 
   HYPRE_Int        iter; 
   HYPRE_Int        my_id, num_procs;
   HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   HYPRE_Real w_norm;
   
   HYPRE_Real epsmac = 1.e-16; 
   HYPRE_Real ieee_check = 0.;

   HYPRE_Real guard_zero_residual; 
   HYPRE_Real cf_ave_0 = 0.0;
   HYPRE_Real cf_ave_1 = 0.0;
   HYPRE_Real weight;
   HYPRE_Real r_norm_0;
   HYPRE_Real relative_error = 1.0;

   HYPRE_Int        rel_change_passed = 0, num_rel_change_check = 0;

   HYPRE_Real real_r_norm_old, real_r_norm_new;

   (gmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,gmres_functions); 
   c = hypre_CTAllocF(HYPRE_Real,k_dim,gmres_functions); 
   s = hypre_CTAllocF(HYPRE_Real,k_dim,gmres_functions); 
   if (rel_change) rs_2 = hypre_CTAllocF(HYPRE_Real,k_dim+1,gmres_functions); 
   
  int recovered = 0;

   hh = hypre_CTAllocF(HYPRE_Real*,k_dim+1,gmres_functions); 
   for (i=0; i < k_dim+1; i++)
   {	
   	hh[i] = hypre_CTAllocF(HYPRE_Real,k_dim,gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(b,b));
   real_r_norm_old = b_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
        hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        hypre_printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied b.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(p[0],p[0]));
   r_norm_0 = r_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
	printf("r_norm is %f \n", r_norm);
      if (logging > 0 || print_level > 0)
      {
        hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        hypre_printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   if ( logging>0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level>1 && my_id == 0 )
      {
  	 hypre_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
      
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
     /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
     den_norm= b_norm;
   }
   else
   {
     /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
     den_norm= r_norm;
   };


   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = hypre_max(a_tol,r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level>1 && my_id == 0 )
   {
      if (b_norm > 0.0)
         {hypre_printf("=============================================\n\n");
          hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          hypre_printf("-----    ------------    ---------- ------------\n");
      
          }

      else
         {hypre_printf("=============================================\n\n");
          hypre_printf("Iters     resid.norm     conv.rate\n");
          hypre_printf("-----    ------------    ----------\n");
      
          };
   }

 // FTI CPR code
    if (enable_fti) {
	recovered = 0;
	printf("Add FTI protection to data objects ... \n");
  	FTI_Protect(1, &iter, 1, FTI_INTG);
  	FTIT_type FTI_REAL;
  	FTI_InitType(&FTI_REAL, sizeof(HYPRE_Real));
  	FTI_Protect(2, &rs[0], k_dim+1, FTI_REAL);
  	FTI_Protect(3, &c[0], k_dim, FTI_REAL);
  	FTI_Protect(4, &s[0], k_dim, FTI_REAL);
/*
  	FTIT_type FTI_REALX;
  	FTI_InitType(&FTI_REALX, sizeof(HYPRE_Real)*k_dim);
  	//FTI_Protect(5, &hh[0], k_dim+1, FTI_REALX);
  	FTI_Protect(5, &hh[0], 1, FTI_REALX);
  	FTI_Protect(6, &hh[1], 1, FTI_REALX);
  	FTI_Protect(7, &hh[2], 1, FTI_REALX);
  	FTI_Protect(8, &hh[3], 1, FTI_REALX);
  	FTI_Protect(9, &hh[4], 1, FTI_REALX);
  	FTI_Protect(10, &hh[5], 1, FTI_REALX);
  	FTI_Protect(11, &hh[6], 1, FTI_REALX);
  	FTI_Protect(12, &hh[7], 1, FTI_REALX);
  	FTI_Protect(13, &hh[8], 1, FTI_REALX);
  	FTI_Protect(14, &hh[9], 1, FTI_REALX);
  	FTI_Protect(15, &hh[10], 1, FTI_REALX);
  	FTI_Protect(16, &hh[11], 1, FTI_REALX);
  	FTI_Protect(17, &hh[12], 1, FTI_REALX);
  	FTI_Protect(18, &hh[13], 1, FTI_REALX);
  	FTI_Protect(19, &hh[14], 1, FTI_REALX);
  	FTI_Protect(20, &hh[15], 1, FTI_REALX);
  	FTI_Protect(21, &hh[16], 1, FTI_REALX);
  	FTI_Protect(22, &hh[17], 1, FTI_REALX);
  	FTI_Protect(23, &hh[18], 1, FTI_REALX);
  	FTI_Protect(24, &hh[19], 1, FTI_REALX);
  	FTI_Protect(25, &hh[20], 1, FTI_REALX);
*/
  	FTI_Protect(26, &epsilon, 1, FTI_REAL);
  	FTI_Protect(27, &max_iter, 1, FTI_INTG);
  	FTI_Protect(28, &epsmac, 1, FTI_REAL);
  	FTI_Protect(29, &b_norm, 1, FTI_REAL);
  	FTI_Protect(30, &norms[0], max_iter+1, FTI_REAL);
  	FTI_Protect(31, &r_norm_0, 1, FTI_REAL);

  	int size;
  	size = (*(hypre_Vector *)(*(hypre_ParVector *)b).local_vector).size;
  	FTI_Protect(32, &(*(hypre_Vector *)(*(hypre_ParVector *)b).local_vector).data[0], size, FTI_INTG);
  	size = (*(hypre_Vector *)(*(hypre_ParVector *)x).local_vector).size;
  	FTI_Protect(33, &(*(hypre_Vector *)(*(hypre_ParVector *)x).local_vector).data[0], size, FTI_INTG);
  	size = (*(hypre_Vector *)(*(hypre_ParVector *)w).local_vector).size;
  	FTI_Protect(34, &(*(hypre_Vector *)(*(hypre_ParVector *)w).local_vector).data[0], size, FTI_INTG);
/*
        FTI_Protect(35, &((*(hypre_ParVector *)b).global_size),1,FTI_INTG);
        FTI_Protect(36, &((*(hypre_ParVector *)x).global_size),1,FTI_INTG);
        FTI_Protect(37, &((*(hypre_ParVector *)w).global_size),1,FTI_INTG);

        FTI_Protect(38, &((*(hypre_ParVector *)b).first_index),1,FTI_INTG);
        FTI_Protect(39, &((*(hypre_ParVector *)x).first_index),1,FTI_INTG);
        FTI_Protect(40, &((*(hypre_ParVector *)w).first_index),1,FTI_INTG);
	
        FTI_Protect(41, &((*(hypre_ParVector *)b).last_index),1,FTI_INTG);
        FTI_Protect(42, &((*(hypre_ParVector *)x).last_index),1,FTI_INTG);
        FTI_Protect(43, &((*(hypre_ParVector *)w).last_index),1,FTI_INTG);
	
        FTI_Protect(38, (*(hypre_ParVector *)b).partitioning,num_procs+1,FTI_INTG);
        FTI_Protect(39, (*(hypre_ParVector *)x).partitioning,num_procs+1,FTI_INTG);
        FTI_Protect(40, (*(hypre_ParVector *)w).partitioning,num_procs+1,FTI_INTG);
	
        FTI_Protect(41, &((*(hypre_ParVector *)b).actual_local_size),1,FTI_INTG);
        FTI_Protect(42, &((*(hypre_ParVector *)x).actual_local_size),1,FTI_INTG);
        FTI_Protect(43, &((*(hypre_ParVector *)w).actual_local_size),1,FTI_INTG);
	
        FTI_Protect(44, &((*(hypre_ParVector *)b).owns_data),1,FTI_INTG);
        FTI_Protect(45, &((*(hypre_ParVector *)x).owns_data),1,FTI_INTG);
        FTI_Protect(46, &((*(hypre_ParVector *)w).owns_data),1,FTI_INTG);
/-*	
        FTI_Protect(47, &((*(hypre_ParVector *)b).owns_partitioning),1,FTI_INTG);
        FTI_Protect(48, &((*(hypre_ParVector *)x).owns_partitioning),1,FTI_INTG);
        FTI_Protect(49, &((*(hypre_ParVector *)w).owns_partitioning),1,FTI_INTG);
	
        FTI_Protect(50, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).length,1,FTI_INTG);
        FTI_Protect(51, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).length,1,FTI_INTG);
        FTI_Protect(52, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).length,1,FTI_INTG);
	
        FTI_Protect(53, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).row_start,1,FTI_INTG);
        FTI_Protect(54, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).row_start,1,FTI_INTG);
        FTI_Protect(55, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).row_start,1,FTI_INTG);
	
        FTI_Protect(56, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).row_end,1,FTI_INTG);
        FTI_Protect(57, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).row_end,1,FTI_INTG);
        FTI_Protect(58, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).row_end,1,FTI_INTG);
	
        FTI_Protect(59, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).storage_length,1,FTI_INTG);
        FTI_Protect(60, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).storage_length,1,FTI_INTG);
        FTI_Protect(61, &(*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).storage_length,1,FTI_INTG);
/-*	
	int b_sl = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).storage_length;
	int x_sl = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).storage_length;
	int w_sl = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).storage_length;
        FTI_Protect(62, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).proc_list,b_sl,FTI_INTG);
        FTI_Protect(63, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).proc_list,x_sl,FTI_INTG);
        FTI_Protect(64, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).proc_list,w_sl,FTI_INTG);
	
        FTI_Protect(65, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).row_start_list,b_sl,FTI_INTG);
        FTI_Protect(66, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).row_start_list,x_sl,FTI_INTG);
        FTI_Protect(67, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).row_start_list,w_sl,FTI_INTG);
	
        FTI_Protect(68, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).row_end_list,b_sl,FTI_INTG);
        FTI_Protect(69, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).row_end_list,x_sl,FTI_INTG);
        FTI_Protect(70, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).row_end_list,w_sl,FTI_INTG);
	
	int b_l = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).length;
	int x_l = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).length;
	int w_l = (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).length;
        FTI_Protect(71, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)b).assumed_partition).sort_index,b_l,FTI_INTG);
        FTI_Protect(72, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)x).assumed_partition).sort_index,x_l,FTI_INTG);
        FTI_Protect(73, (*(hypre_IJAssumedPart *)(*(hypre_ParVector *)w).assumed_partition).sort_index,w_l,FTI_INTG);
*/	
int n = 0;
for (int i=1;i<k_dim;i++) {
   for (int j=0;j<i;j++) {
      FTI_Protect(74+n, &hh[j][i-1], 1, FTI_REAL);
      n=n+1;
   }

}
 
	printf("Done: Add FTI protection to data objects ... \n");

    }
// FTI CPR code

  /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   /* outer iteration cycle */
   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           hypre_TFreeF(c,gmres_functions); 
           hypre_TFreeF(s,gmres_functions); 
           hypre_TFreeF(rs,gmres_functions);
           if (rel_change)  hypre_TFreeF(rs_2,gmres_functions);
           for (i=0; i < k_dim+1; i++) hypre_TFreeF(hh[i],gmres_functions);
           hypre_TFreeF(hh,gmres_functions); 
	   return hypre_error_flag;
           
	}

        /* see if we are already converged and 
           should print the final norm and exit */
	if (r_norm  <= epsilon && iter >= min_iter) 
        {
           if (!rel_change) /* shouldn't exit after no iterations if
                             * relative change is on*/
           {
              (*(gmres_functions->CopyVector))(b,r);
              (*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
              r_norm = sqrt((*(gmres_functions->InnerProd))(r,r));
              if (r_norm  <= epsilon)
              {
                 if ( print_level>1 && my_id == 0)
                 {
                    hypre_printf("\n\n");
                    hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 break;
              }
              else
                 if ( print_level>0 && my_id == 0)
                    hypre_printf("false convergence 1\n");
           }
	}

      
        
      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t,p[0]);
	i = 0;

        /***RESTART CYCLE (right-preconditioning) ***/
        while (i < k_dim && iter < max_iter)
	{
HYPRE_Int rank;
HYPRE_Int procsize;
hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD,&rank);
hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD,&procsize);

/* code for C/R implementation */
// write and read checkpoints
	//if (procfi == 1 || nodefi == 1) {
	if (state == OMPI_REINIT_RESTARTED || state == OMPI_REINIT_REINITED) {
	   procfi = 0;
	   nodefi = 0; 
	}
	//}

    	sleep(1);
	      
	if (enable_fti) {
	   if ( FTI_Status() != 0){ 
#ifdef TIMER
   double elapsed_time;
   timeval start;
   timeval end;
   gettimeofday(&start, NULL) ;
#endif
	   	printf("Do FTI Recover to data objects from failure ... \n");
	   	FTI_Recover();
	   	printf("Done: FTI Recover data objects from failure ... \n");
#ifdef TIMER
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
   printf("READ CP TIME: %lf (s) Rank %d \n", elapsed_time, rank);
   fflush(stdout);
#endif
	   	recovered = 1;
           	procfi = 0;
           	nodefi = 0;
	   }
	}

	// do FTI CPR
	if (enable_fti){
#ifdef TIMER
   double elapsed_time;
   timeval start;
   timeval end;
   gettimeofday(&start, NULL) ;
#endif
	    if ( (!recovered) && cp_stride > 0 && (iter%cp_stride +1) == cp_stride ){ 
		FTI_Checkpoint(iter, level);
	    }
	    recovered = 0;
#ifdef TIMER
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
   acc_write_time+=elapsed_time;

#endif
	}
	// do FTI CPR


/* code for C/R implementation */
           i++;
           iter++;
           (*(gmres_functions->ClearVector))(r);
           precond(precond_data, A, p[i-1], r);
           (*(gmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
           /* modified Gram_Schmidt */
           for (j=0; j < i; j++)
           {
              hh[j][i-1] = (*(gmres_functions->InnerProd))(p[j],p[i]);
              (*(gmres_functions->Axpy))(-hh[j][i-1],p[j],p[i]);
           }
           t = sqrt((*(gmres_functions->InnerProd))(p[i],p[i]));
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0/t;
              (*(gmres_functions->ScaleVector))(t,p[i]);
           }
           /* done with modified Gram_schmidt and Arnoldi step.
              update factorization of hh */
           for (j = 1; j < i; j++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t= hh[i][i-1]*hh[i][i-1];
           t+= hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i]/=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if ( print_level>0 )
           {
              norms[iter] = r_norm;
              if ( print_level>1 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                    hypre_printf("% 5d    %e    %f   %e\n", iter, 
                           norms[iter],norms[iter]/norms[iter-1],
                           norms[iter]/b_norm);
                 else
                    hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                           norms[iter]/norms[iter-1]);
              }
           }
           /*convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));
              
              weight   = fabs(cf_ave_1 - cf_ave_0);
              weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
              weight   = 1.0 - weight;
#if 0
              hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                     i, cf_ave_1, cf_ave_0, weight );
#endif
              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              if (rel_change && !rel_change_passed)
              {
                 
                 /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)

                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
                 
                 for (k=0; k<i; k++) /* extra copy of rs so we don't need
                                        to change the later solve */
                    rs_2[k] = rs[k];

                 /* solve tri. system*/
                 rs_2[i-1] = rs_2[i-1]/hh[i-1][i-1];
                 for (k = i-2; k >= 0; k--)
                 {
                    t = 0.0;
                    for (j = k+1; j < i; j++)
                    {
                       t -= hh[k][j]*rs_2[j];
                    }
                    t+= rs_2[k];
                    rs_2[k] = t/hh[k][k];
                 }
                 
                 (*(gmres_functions->CopyVector))(p[i-1],w);
                 (*(gmres_functions->ScaleVector))(rs_2[i-1],w);
                 for (j = i-2; j >=0; j--)
                    (*(gmres_functions->Axpy))(rs_2[j], p[j], w);
                    
                 (*(gmres_functions->ClearVector))(r);
                 /* find correction (in r) */
                 precond(precond_data, A, w, r);
                 /* copy current solution (x) to w (don't want to over-write x)*/
                 (*(gmres_functions->CopyVector))(x,w);

                 /* add the correction */
                 (*(gmres_functions->Axpy))(1.0,r,w);

                 /* now w is the approx solution  - get the norm*/
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(w,w) );

                 if ( !(x_norm <= guard_zero_residual ))
                    /* don't divide by zero */
                 {  /* now get  x_i - x_i-1 */
                    
                    if (num_rel_change_check)
                    {
                       /* have already checked once so we can avoid another precond.
                          solve */
                       (*(gmres_functions->CopyVector))(w, r);
                       (*(gmres_functions->Axpy))(-1.0, w_2, r);
                       /* now r contains x_i - x_i-1*/

                       /* save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(w, w_2);
                    }
                    else
                    {
                       /* first time to check rel change*/

                       /* first save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(w, w_2);

                       /* for relative change take x_(i-1) to be 
                          x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                          Now
                          x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                          - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                          = M^{-1} rs_{i-1}{p_{i-1}} */
                       
                       (*(gmres_functions->ClearVector))(w);
                       (*(gmres_functions->Axpy))(rs_2[i-1], p[i-1], w);
                       (*(gmres_functions->ClearVector))(r);
                       /* apply the preconditioner */
                       precond(precond_data, A, w, r);
                       /* now r contains x_i - x_i-1 */          
                    }
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
                    relative_error = w_norm/x_norm;
                    if (relative_error <= r_tol)
                    {
                       rel_change_passed = 1;
                       break;
                    }
                 }
                 else
                 {
                    rel_change_passed = 1;
                    break;

                 }
                 num_rel_change_check++;
              }
           else /* no relative change */
              {
                 break;
              }
           }
         
	 // new code for C/R implementation  
         /* calculate actual residual norm*/
         //(*(gmres_functions->CopyVector))(b,r);
         //(*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         //real_r_norm_new = r_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
         //if (rank == 0 )
         //{
           // hypre_printf("GMRES Iterations = %d\n", iter);
         //}

    	 if (procfi == 1 && iter==12){
#ifdef TIMER
   	      printf("WRITE CP TIME: %lf (s) Rank %d \n", acc_write_time, rank);
		fflush(stdout);
#endif
              if(rank==(procsize-1)){
#ifdef TIMER
   	   struct timeval tv;
   	   gettimeofday( &tv, NULL );
   	   double ts = tv.tv_sec + tv.tv_usec / 1000000.0;
     	   char hostname[64];
   	   gethostname(hostname, 64);
   	   printf("TIMESTAMP KILL: %lf (s) node %s daemon %d\n", ts, hostname, getpid());
   	   fflush(stdout);
#endif
      	   printf("KILL rank %d\n", rank);
      	   kill(getpid(), SIGTERM);
    	      }
	 }

    	 if (nodefi == 1 && iter==12){
#ifdef TIMER
   	      printf("WRITE CP TIME: %lf (s) Rank %d \n", acc_write_time, rank);
		fflush(stdout);
#endif
              if(rank==(procsize-1)){
#ifdef TIMER
   	   struct timeval tv;
   	   gettimeofday( &tv, NULL );
   	   double ts = tv.tv_sec + tv.tv_usec / 1000000.0;
     	   char hostname[64];
   	   gethostname(hostname, 64);
   	   printf("TIMESTAMP KILL: %lf (s) node %s daemon %d\n", ts, hostname, getpid());
   	   fflush(stdout);
#endif
      	   gethostname(hostname, 64);
      	   printf("KILL %s daemon %d rank %d\n", hostname, (int) getppid(), rank);
      	   kill(getppid(), SIGTERM);
    	      }
	 }

	 // new code for C/R implementation

	} /*** end of restart cycle ***/

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1]/hh[i-1][i-1];
	for (k = i-2; k >= 0; k--)
	{
           t = 0.0;
           for (j = k+1; j < i; j++)
           {
              t -= hh[k][j]*rs[j];
           }
           t+= rs[k];
           rs[k] = t/hh[k][k];
	}

        (*(gmres_functions->CopyVector))(p[i-1],w);
        (*(gmres_functions->ScaleVector))(rs[i-1],w);
        for (j = i-2; j >=0; j--)
                (*(gmres_functions->Axpy))(rs[j], p[j], w);

	(*(gmres_functions->ClearVector))(r);
	/* find correction (in r) */
        precond(precond_data, A, w, r);

        /* update current solution x (in x) */
	(*(gmres_functions->Axpy))(1.0,r,x);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm  <= epsilon && iter >= min_iter)
        {
           if (skip_real_r_check)
           {
              (gmres_data -> converged) = 1;
              break;
           }

           /* calculate actual residual norm*/
           (*(gmres_functions->CopyVector))(b,r);
           (*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
           real_r_norm_new = r_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );

           if (r_norm <= epsilon)
           {
              if (rel_change && !rel_change_passed) /* calculate the relative change */
              {

                 /* calculate the norm of the solution */
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(x,x) );
               
                 if ( !(x_norm <= guard_zero_residual ))
                    /* don't divide by zero */
                 {
                    
                    /* for relative change take x_(i-1) to be 
                       x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                       Now
                       x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                       - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                       = M^{-1} rs_{i-1}{p_{i-1}} */
                    (*(gmres_functions->ClearVector))(w);
                    (*(gmres_functions->Axpy))(rs[i-1], p[i-1], w);
                    (*(gmres_functions->ClearVector))(r);
                    /* apply the preconditioner */
                    precond(precond_data, A, w, r);
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
                    relative_error= w_norm/x_norm;
                    if ( relative_error < r_tol )
                    {
                       (gmres_data -> converged) = 1;
                       if ( print_level>1 && my_id == 0 )
                       {
                          hypre_printf("\n\n");
                          hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                       }
                       break;
                    }
                 }
                 else
                 {
                    (gmres_data -> converged) = 1;
                    if ( print_level>1 && my_id == 0 )
                    {
                       hypre_printf("\n\n");
                       hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                    }
                    break;
                 }

              }
              else /* don't need to check rel. change */
              {
                 if ( print_level>1 && my_id == 0 )
                 {
                    hypre_printf("\n\n");
                    hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (gmres_data -> converged) = 1;
                 break;
              }
           }
           else /* conv. has not occurred, according to true residual */
           {
              /* exit if the real residual norm has not decreased */
              if (real_r_norm_new >= real_r_norm_old)
              {
                 if (print_level > 1 && my_id == 0)
                 {
                    hypre_printf("\n\n");
                    hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (gmres_data -> converged) = 1;
                 break;
              }

              /* report discrepancy between real/GMRES residuals and restart */
              if ( print_level>0 && my_id == 0)
                 hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
              (*(gmres_functions->CopyVector))(r,p[0]);
              i = 0;
              real_r_norm_old = real_r_norm_new;
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j=i ; j > 0; j--)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) (*(gmres_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
        for (j=i-1 ; j > 0; j--)
           (*(gmres_functions->Axpy))(rs[j],p[j],p[i]);
        
        if (i)
        {
           (*(gmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
           (*(gmres_functions->Axpy))(1.0,p[i],p[0]);
        }
   } /* END of iteration while loop */


   if ( print_level>1 && my_id == 0 )
          hypre_printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (gmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (gmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) hypre_error(HYPRE_ERROR_CONV);

   hypre_TFreeF(c,gmres_functions); 
   hypre_TFreeF(s,gmres_functions); 
   hypre_TFreeF(rs,gmres_functions);
   if (rel_change)  hypre_TFreeF(rs_2,gmres_functions);

   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFreeF(hh[i],gmres_functions);
   }
   hypre_TFreeF(hh,gmres_functions); 

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetKDim, hypre_GMRESGetKDim
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetKDim( void   *gmres_vdata,
                    HYPRE_Int   k_dim )
{
	hypre_GMRESData *gmres_data =(hypre_GMRESData *) gmres_vdata;

   
   (gmres_data -> k_dim) = k_dim;
 
   return hypre_error_flag;
   
}

HYPRE_Int
hypre_GMRESGetKDim( void   *gmres_vdata,
                    HYPRE_Int * k_dim )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *k_dim = (gmres_data -> k_dim);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetTol, hypre_GMRESGetTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetTol( void   *gmres_vdata,
                   HYPRE_Real  tol       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> tol) = tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetTol( void   *gmres_vdata,
                   HYPRE_Real  * tol      )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *tol = (gmres_data -> tol);
 
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_GMRESSetAbsoluteTol, hypre_GMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetAbsoluteTol( void   *gmres_vdata,
                   HYPRE_Real  a_tol       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> a_tol) = a_tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetAbsoluteTol( void   *gmres_vdata,
                   HYPRE_Real  * a_tol      )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *a_tol = (gmres_data -> a_tol);
 
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_GMRESSetConvergenceFactorTol, hypre_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetConvergenceFactorTol( void   *gmres_vdata,
                   HYPRE_Real  cf_tol       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> cf_tol) = cf_tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetConvergenceFactorTol( void   *gmres_vdata,
                   HYPRE_Real * cf_tol       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *cf_tol = (gmres_data -> cf_tol);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMinIter, hypre_GMRESGetMinIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetMinIter( void *gmres_vdata,
                       HYPRE_Int   min_iter  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> min_iter) = min_iter;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetMinIter( void *gmres_vdata,
                       HYPRE_Int * min_iter  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *min_iter = (gmres_data -> min_iter);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMaxIter, hypre_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetMaxIter( void *gmres_vdata,
                       HYPRE_Int   max_iter  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> max_iter) = max_iter;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetMaxIter( void *gmres_vdata,
                       HYPRE_Int * max_iter  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *max_iter = (gmres_data -> max_iter);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetRelChange, hypre_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetRelChange( void *gmres_vdata,
                         HYPRE_Int   rel_change  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> rel_change) = rel_change;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetRelChange( void *gmres_vdata,
                         HYPRE_Int * rel_change  )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *rel_change = (gmres_data -> rel_change);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetSkipRealResidualCheck, hypre_GMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetSkipRealResidualCheck( void *gmres_vdata,
                                     HYPRE_Int skip_real_r_check )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   (gmres_data -> skip_real_r_check) = skip_real_r_check;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetSkipRealResidualCheck( void *gmres_vdata,
                                     HYPRE_Int *skip_real_r_check)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   *skip_real_r_check = (gmres_data -> skip_real_r_check);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetStopCrit, hypre_GMRESGetStopCrit
 *
 *  OBSOLETE 
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetStopCrit( void   *gmres_vdata,
                        HYPRE_Int  stop_crit       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> stop_crit) = stop_crit;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetStopCrit( void   *gmres_vdata,
                        HYPRE_Int * stop_crit       )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *stop_crit = (gmres_data -> stop_crit);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESSetPrecond( void  *gmres_vdata,
                       HYPRE_Int  (*precond)(void*,void*,void*,void*),
                       HYPRE_Int  (*precond_setup)(void*,void*,void*,void*),
                       void  *precond_data )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;

 
   (gmres_functions -> precond)        = precond;
   (gmres_functions -> precond_setup)  = precond_setup;
   (gmres_data -> precond_data)   = precond_data;
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESGetPrecond( void         *gmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *precond_data_ptr = (HYPRE_Solver)(gmres_data -> precond_data);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrintLevel, hypre_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetPrintLevel( void *gmres_vdata,
                        HYPRE_Int   level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> print_level) = level;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetPrintLevel( void *gmres_vdata,
                        HYPRE_Int * level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *level = (gmres_data -> print_level);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetLogging, hypre_GMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetLogging( void *gmres_vdata,
                      HYPRE_Int   level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   (gmres_data -> logging) = level;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetLogging( void *gmres_vdata,
                      HYPRE_Int * level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *level = (gmres_data -> logging);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESGetNumIterations( void *gmres_vdata,
                             HYPRE_Int  *num_iterations )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *num_iterations = (gmres_data -> num_iterations);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetConverged
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESGetConverged( void *gmres_vdata,
                             HYPRE_Int  *converged )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *converged = (gmres_data -> converged);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_GMRESGetFinalRelativeResidualNorm( void   *gmres_vdata,
                                         HYPRE_Real *relative_residual_norm )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

 
   *relative_residual_norm = (gmres_data -> rel_residual_norm);
   
   return hypre_error_flag;
} 


// write checkpoints for the computation iterations
// variables: iter, rs[],  c, s, hh[], epsilon, max_iter,epsmac,p[10],precond_data[10], b_norm, norms[], r_norm_0
// A, x, w, b
static void AMGCheckpointWrite(HYPRE_Int iter,HYPRE_Real *rs,HYPRE_Real *c,HYPRE_Real *s,HYPRE_Real **hh,HYPRE_Real epsilon,HYPRE_Int max_iter,HYPRE_Real epsmac,void *p,HYPRE_Int *precond_data,HYPRE_Real b_norm,HYPRE_Real *norms, HYPRE_Real r_norm_0, void *A, void *x, void *w, void *b, HYPRE_Int rank, HYPRE_Int k_dim) {
  stringstream oss( stringstream::out | stringstream::binary);
/*
  // checkpoint iter
  oss.write(reinterpret_cast<char *>(&iter), sizeof(HYPRE_Int));
//// testing ////
   //az: printf("** write checkpoint for iteration [%d] \n", iter);

  // checkpoint rs
  oss.write(reinterpret_cast<char *>(&rs[0]), sizeof(HYPRE_Real)*(k_dim+1));

  // checkpoint c
  oss.write(reinterpret_cast<char *>(&c[0]), sizeof(HYPRE_Real)*(k_dim));

  // checkpoint s
  oss.write(reinterpret_cast<char *>(&s[0]), sizeof(HYPRE_Real)*(k_dim));

  // checkpoint hh
  for (int i=0;i<k_dim+1;i++) {
     oss.write(reinterpret_cast<char *>(&hh[i]), sizeof(HYPRE_Real)*(k_dim));
  }

*/
/*
  // checkpoint epsilon
  oss.write(reinterpret_cast<char *>(&epsilon), sizeof(HYPRE_Real));

  // checkpoint max_iter
  oss.write(reinterpret_cast<char *>(&max_iter), sizeof(HYPRE_Int));

  // checkpoint epsmac
  oss.write(reinterpret_cast<char *>(&epsmac), sizeof(HYPRE_Real));

  // checkpoint p
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  // int size = (*(*(hypre_ParVector *) p).local_vector).size;
  //az: int size = (*(hypre_Vector *)(*(hypre_ParVector *)p).local_vector).size;
  //az: oss.write(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  //az: for (int i=0;i<size;i++) {
     //az: oss.write(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)p).local_vector).data[i]), sizeof(HYPRE_Int));
  //az: }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition
  
  // checkpoint precond_data
  // very complicated, not checkpoint for now
  
  // checkpoint b_norm
  oss.write(reinterpret_cast<char *>(&b_norm), sizeof(HYPRE_Real));
  

  // checkpoint norms
  oss.write(reinterpret_cast<char *>(&norms[0]), sizeof(HYPRE_Real)*(max_iter+1));
  

  // checkpoint r_norm_0
  oss.write(reinterpret_cast<char *>(&r_norm_0), sizeof(HYPRE_Real));

*/
  // checkpoint A
  // too complicated, not checkpoint for now
  //

/*
  // checkpoint b
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  //size = (*(*(hypre_ParVector *) b).local_vector).size;
  int size;
  size = (*(hypre_Vector *)(*(hypre_ParVector *)b).local_vector).size;
  oss.write(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  for (int i=0;i<size;i++) {
     oss.write(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)b).local_vector).data[i]), sizeof(HYPRE_Int));
  }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition
  
*/
  
  // checkpoint w
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  //size = (*(*(hypre_ParVector *) w).local_vector).size;
  int size = (*(hypre_Vector *)(*(hypre_ParVector *)w).local_vector).size;
  oss.write(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  for (int i=0;i<size;i++) {
     oss.write(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)w).local_vector).data[i]), sizeof(HYPRE_Int));
  }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition

  // checkpoint x
  // too complicated, not checkpoint for now
  //
  
  //size = oss.str().size();
/*
    char filename[64];
    sprintf( filename, "check_%d_%d", rank, iter);
//// testing ////
    printf("write check_%d_%d \n", rank, iter);
    FILE *fp = fopen( filename, "wb" );
    fwrite( &size, sizeof(int), 1, fp );
    //fwrite( data, size, 1, fp );
    fwrite( const_cast<char *>( oss.str().c_str() ), size, 1, fp );
    fclose( fp );
*/
  //printf("Checkpoint size is %d bytes for rank %d ...\n", size, rank);
  write_cp(cp2f, cp2m, cp2a, rank, iter, const_cast<char *>( oss.str().c_str() ), size, MPI_COMM_WORLD);

} // AMGCheckpointWrite

// write checkpoints for the computation iterations
// variables: iter, rs[],  c, s, hh[], epsilon, max_iter,epsmac,p[10],precond_data[10], b_norm, norms[], r_norm_0
// A, x, w, b
static void AMGCheckpointRead(HYPRE_Int &iter,HYPRE_Real *rs,HYPRE_Real *c,HYPRE_Real *s,HYPRE_Real **hh,HYPRE_Real &epsilon,HYPRE_Int &max_iter,HYPRE_Real &epsmac,void *p,HYPRE_Int *precond_data,HYPRE_Real &b_norm,HYPRE_Real *norms, HYPRE_Real &r_norm_0, void *A, void *x, void *w, void *b, HYPRE_Int rank, HYPRE_Int survivor, HYPRE_Int &k_dim) {

  char *data;
  size_t sizeofCP=read_cp(survivor, cp2f, cp2m, cp2a, rank, &data, MPI_COMM_WORLD);
  stringstream iss(string( data, data + sizeofCP ), stringstream::in | stringstream::binary );

  // checkpoint iter
  iss.read(reinterpret_cast<char *>(&iter), sizeof(HYPRE_Int));
//// testing ////
   //az: printf("** read checkpoint from iteration [%d] \n", iter);

  // checkpoint rs
  for (int i=0;i<k_dim+1;i++) {
     iss.read(reinterpret_cast<char *>(&rs[i]), sizeof(HYPRE_Real));
  }

  // checkpoint c
  for (int i=0;i<k_dim;i++) {
     iss.read(reinterpret_cast<char *>(&c[i]), sizeof(HYPRE_Real));
  }

  // checkpoint s
  for (int i=0;i<k_dim;i++) {
     iss.read(reinterpret_cast<char *>(&s[i]), sizeof(HYPRE_Real));
  }

  // checkpoint hh
  for (int i=0;i<k_dim+1;i++) {
     for (int j=0;j<k_dim;j++) {
        iss.read(reinterpret_cast<char *>(&hh[i][j]), sizeof(HYPRE_Real));
     }
  }

  // checkpoint epsilon
  iss.read(reinterpret_cast<char *>(&epsilon), sizeof(HYPRE_Real));

  // checkpoint max_iter
  iss.read(reinterpret_cast<char *>(&max_iter), sizeof(HYPRE_Int));

  // checkpoint epsmac
  iss.read(reinterpret_cast<char *>(&epsmac), sizeof(HYPRE_Real));

  // checkpoint p
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  ///az: int size;
  //az: iss.read(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  //az: for (int i=0;i<size;i++) {
     //az: iss.read(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)p).local_vector).data[i]), sizeof(HYPRE_Int));
  //az: }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition
  
  // checkpoint precond_data
  // very complicated, not checkpoint for now
  
  // checkpoint b_norm
  iss.read(reinterpret_cast<char *>(&b_norm), sizeof(HYPRE_Real));
  
  // checkpoint norms
  for (int i=0;i<max_iter+1;i++) {
     iss.read(reinterpret_cast<char *>(&norms[i]), sizeof(HYPRE_Real));
  }

  // checkpoint r_norm_0
  iss.read(reinterpret_cast<char *>(&r_norm_0), sizeof(HYPRE_Real));

  // checkpoint A
  // too complicated, not checkpoint for now
  //

  // checkpoint b
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  int size;  
  iss.read(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  for (int i=0;i<size;i++) {
     iss.read(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)b).local_vector).data[i]), sizeof(HYPRE_Int));
     //iss.read(reinterpret_cast<char *>(&(*(*(hypre_ParVector *) b).local_vector).data[i]), sizeof(HYPRE_Int));
  }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition
  

  // checkpoint w
  // only checkpoint partial data because of the complexity
  /// global_size
  /// first_index
  /// last_index
  /// partitioning
  /// actual_local_size
  /// data size
  // int size;
  iss.read(reinterpret_cast<char *>(&size), sizeof(HYPRE_Int));
  /// data/local_vector
  for (int i=0;i<size;i++) {
     iss.read(reinterpret_cast<char *>(&(*(hypre_Vector *)(*(hypre_ParVector *)w).local_vector).data[i]), sizeof(HYPRE_Int));
     //iss.read(reinterpret_cast<char *>(&(*(*(hypre_ParVector *) w).local_vector).data[i]), sizeof(HYPRE_Int));
  }
  /// owns_data
  /// owns_partitioning
  /// assumed_partition
  

  // checkpoint x
  // too complicated, not checkpoint for now
  //
} // AMGCheckpointRead

