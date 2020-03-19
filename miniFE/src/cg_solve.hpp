#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia	Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>

#include "../../libcheckpoint/checkpoint.h"

#include "ulfm-util.hpp"

/* world will swap between worldc[0] and worldc[1] after each respawn */
extern MPI_Comm worldc[2];
extern int worldi;
#define world (worldc[worldi])

static int survivor;

namespace miniFE {

template<typename OperatorType=miniFE::CSRMatrix<double, int, long long int>,
         typename VectorType=miniFE::Vector<double, int, long long int>,
         typename Matvec=miniFE::matvec_overlap<miniFE::CSRMatrix<double, int, long long int>, miniFE::Vector<double, int, long long int> >,
	 typename Scalar=double,
  	 typename LocalOrdinal=int,
 	 typename GlobalOrdinal=long long int>
void 
ApplicationCheckpointWrite(miniFE::CSRMatrix<double, int, long long int> &A, miniFE::Vector<double, int, long long int> &b, miniFE::Vector<double, int, long long int> &x,miniFE::TypeTraits<double>::magnitude_type& normr, miniFE::timer_type*&  my_cg_times, miniFE::Vector<double, int, long long int> &r, miniFE::Vector<double, int, long long int> &p, typename TypeTraits<double>::magnitude_type & rtrans,typename TypeTraits<double>::magnitude_type &oldrtrans,miniFE::CSRMatrix<double, int, long long int>::LocalOrdinalType &num_iters,int rank, int cp2f, int cp2m, int cp2a) {

  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  std::stringstream oss( std::stringstream::out | std::stringstream::binary );  

  // checkpoint A
  int size;
  size=A.external_index.size();
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.external_index[i]), sizeof(GlobalOrdinal));
  }
 
  size=A.external_local_index.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.external_local_index[i]), sizeof(GlobalOrdinal));
  }
  
  size=A.elements_to_send.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.elements_to_send[i]), sizeof(GlobalOrdinal));
  }

  size=A.neighbors.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.neighbors[i]), sizeof(int));
  }

  size=A.recv_length.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.recv_length[i]), sizeof(LocalOrdinal));
  }

  size=A.send_length.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.send_length[i]), sizeof(LocalOrdinal));
  }

  size=A.send_buffer.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.send_buffer[i]), sizeof(Scalar));
  }

  size=A.request.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.request[i]), sizeof(MPI_Request));
  }

  size=A.rows.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.rows[i]), sizeof(GlobalOrdinal));
  }

  size=A.row_offsets.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.row_offsets[i]), sizeof(LocalOrdinal));
  }

  size=A.row_offsets_external.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.row_offsets_external[i]), sizeof(LocalOrdinal));
  }

  size=A.packed_cols.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.packed_cols[i]), sizeof(GlobalOrdinal));
  }

  size=A.packed_coefs.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&A.packed_coefs[i]), sizeof(Scalar));
  }

  oss.write(reinterpret_cast<char *>(&A.has_local_indices), sizeof(bool));

  oss.write(reinterpret_cast<char *>(&A.num_cols), sizeof(LocalOrdinal));


  // checkpoint b
  oss.write(reinterpret_cast<char *>(&b.startIndex), sizeof(GlobalOrdinal));
  
  oss.write(reinterpret_cast<char *>(&b.local_size), sizeof(LocalOrdinal));

  size=b.coefs.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&b.coefs[i]), sizeof(Scalar));
  }

  // checkpoint x
  oss.write(reinterpret_cast<char *>(&x.startIndex), sizeof(GlobalOrdinal));
  
  oss.write(reinterpret_cast<char *>(&x.local_size), sizeof(LocalOrdinal));

  size=x.coefs.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&x.coefs[i]), sizeof(Scalar));
  }

  // checkpoint r
  oss.write(reinterpret_cast<char *>(&r.startIndex), sizeof(GlobalOrdinal));
  
  oss.write(reinterpret_cast<char *>(&r.local_size), sizeof(LocalOrdinal));

  size=r.coefs.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&r.coefs[i]), sizeof(Scalar));
  }

  // checkpoint p
  oss.write(reinterpret_cast<char *>(&p.startIndex), sizeof(GlobalOrdinal));
  
  oss.write(reinterpret_cast<char *>(&p.local_size), sizeof(LocalOrdinal));

  size=p.coefs.size(); 
  oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  for (int i=0;i<size;i++) {
    oss.write(reinterpret_cast<char *>(&p.coefs[i]), sizeof(Scalar));
  }

  // checkpoint Ap
  //oss.write(reinterpret_cast<char *>(&Ap.startIndex), sizeof(GlobalOrdinal));
  
  //oss.write(reinterpret_cast<char *>(&Ap.local_size), sizeof(LocalOrdinal));

  //size=Ap.coefs.size(); 
  //oss.write(reinterpret_cast<char *>(&size), sizeof(int));
  //for (int i=0;i<size;i++) {
  //  oss.write(reinterpret_cast<char *>(&Ap.coefs[i]), sizeof(Scalar));
  //}

  // checkpoint normr ?
  oss.write(reinterpret_cast<char *>(&normr),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint mg_cg_times
  oss.write(reinterpret_cast<char *>(my_cg_times),sizeof(timer_type));
  
  // checkpoint rtrans
  oss.write(reinterpret_cast<char *>(&rtrans),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint oldrtrans
  oss.write(reinterpret_cast<char *>(&oldrtrans),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint num_iters
  oss.write(reinterpret_cast<char *>(&num_iters),sizeof(LocalOrdinalType));
  
  size = oss.str().size();
  
  write_cp(cp2f, cp2m, cp2a, rank, num_iters, const_cast<char *>( oss.str().c_str() ), size, world);  

}


template<typename OperatorType=miniFE::CSRMatrix<double, int, long long int>,
         typename VectorType=miniFE::Vector<double, int, long long int>,
         typename Matvec=miniFE::matvec_overlap<miniFE::CSRMatrix<double, int, long long int>, miniFE::Vector<double, int, long long int> >,
	 typename Scalar=double,
  	 typename LocalOrdinal=int,
 	 typename GlobalOrdinal=long long int>
void 
ApplicationCheckpointRead(int survivor, int cp2f, int cp2m, int cp2a, int rank, miniFE::CSRMatrix<double, int, long long int> &A, miniFE::Vector<double, int, long long int> &b, miniFE::Vector<double, int, long long int> &x,miniFE::TypeTraits<double>::magnitude_type *normr, miniFE::timer_type*&  my_cg_times, miniFE::Vector<double, int, long long int> &r, miniFE::Vector<double, int, long long int> &p, typename TypeTraits<double>::magnitude_type *rtrans,typename TypeTraits<double>::magnitude_type *oldrtrans,miniFE::CSRMatrix<double, int, long long int>::LocalOrdinalType *num_iters) {

  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  //typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  char* data;
  size_t sizeofCP=read_cp(survivor, cp2f, cp2m, cp2a, rank, &data, world);

  std::stringstream iss(std::string( data, data + sizeofCP ), std::stringstream::in | std::stringstream::binary );  

  // checkpoint A
  int size;
  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.external_index.resize(size);
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.external_index[i]), sizeof(GlobalOrdinal));
  }
 
  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.external_local_index.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.external_local_index[i]), sizeof(GlobalOrdinal));
  }
  
  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.elements_to_send.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.elements_to_send[i]), sizeof(GlobalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.neighbors.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.neighbors[i]), sizeof(int));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.recv_length.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.recv_length[i]), sizeof(LocalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.send_length.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.send_length[i]), sizeof(LocalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.send_buffer.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.send_buffer[i]), sizeof(Scalar));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.request.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.request[i]), sizeof(MPI_Request));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.rows.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.rows[i]), sizeof(GlobalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.row_offsets.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.row_offsets[i]), sizeof(LocalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.row_offsets_external.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.row_offsets_external[i]), sizeof(LocalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.packed_cols.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.packed_cols[i]), sizeof(GlobalOrdinal));
  }

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  A.packed_coefs.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&A.packed_coefs[i]), sizeof(Scalar));
  }

  iss.read(reinterpret_cast<char *>(&A.has_local_indices), sizeof(bool));

  iss.read(reinterpret_cast<char *>(&A.num_cols), sizeof(LocalOrdinal));

  // checkpoint x
  iss.read(reinterpret_cast<char *>(&x.startIndex), sizeof(GlobalOrdinal));
  
  iss.read(reinterpret_cast<char *>(&x.local_size), sizeof(LocalOrdinal));

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  x.coefs.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&x.coefs[i]), sizeof(Scalar));
  }


  // checkpoint b
  iss.read(reinterpret_cast<char *>(&b.startIndex), sizeof(GlobalOrdinal));
  
  iss.read(reinterpret_cast<char *>(&b.local_size), sizeof(LocalOrdinal));

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  b.coefs.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&b.coefs[i]), sizeof(Scalar));
  }

  // checkpoint r
  iss.read(reinterpret_cast<char *>(&r.startIndex), sizeof(GlobalOrdinal));
  
  iss.read(reinterpret_cast<char *>(&r.local_size), sizeof(LocalOrdinal));

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  r.coefs.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&r.coefs[i]), sizeof(Scalar));
  }

  // checkpoint p
  iss.read(reinterpret_cast<char *>(&p.startIndex), sizeof(GlobalOrdinal));
  
  iss.read(reinterpret_cast<char *>(&p.local_size), sizeof(LocalOrdinal));

  iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  p.coefs.resize(size); 
  for (int i=0;i<size;i++) {
    iss.read(reinterpret_cast<char *>(&p.coefs[i]), sizeof(Scalar));
  }

  // checkpoint Ap
  //iss.read(reinterpret_cast<char *>(&Ap.startIndex), sizeof(GlobalOrdinal));
  
  //iss.read(reinterpret_cast<char *>(&Ap.local_size), sizeof(LocalOrdinal));

  //iss.read(reinterpret_cast<char *>(&size), sizeof(int));
  //Ap.coefs.resize(size); 
  //for (int i=0;i<size;i++) {
  //  iss.read(reinterpret_cast<char *>(&Ap.coefs[i]), sizeof(Scalar));
  //}

  // checkpoint normr ?
  iss.read(reinterpret_cast<char *>(normr),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint mg_cg_times
  iss.read(reinterpret_cast<char *>(my_cg_times),sizeof(timer_type));
  
  // checkpoint rtrans
  iss.read(reinterpret_cast<char *>(rtrans),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint oldrtrans
  iss.read(reinterpret_cast<char *>(oldrtrans),sizeof(typename TypeTraits<ScalarType>::magnitude_type));
  
  // checkpoint num_iters
  iss.read(reinterpret_cast<char *>(num_iters),sizeof(LocalOrdinalType));
  
  // free data
  free(data);

}


template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
               const VectorType& v,
               const VectorType& w)
{
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

//This is code that was copied from Aztec, and originally written
//by my hero, Ray Tuminaro.
//
//Assuming that inner = <v,w> (inner product of v and w),
//v and w are considered orthogonal if
//  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v,v));
  magnitude wnorm = std::sqrt(dot(w,w));
  return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

template<typename OperatorType,
         typename VectorType,
         typename Matvec>
void
cg_solve(OperatorType& A,
         VectorType& b,
         VectorType& x,
         Matvec matvec,
         typename OperatorType::LocalOrdinalType max_iter,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
         typename OperatorType::LocalOrdinalType& num_iters,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
         timer_type* my_cg_times,
	 Parameters& params,
	 int do_recover)
{
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  //typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = mytimer();

  int myproc = 0;
  int procsize = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(world, &myproc);
  MPI_Comm_size(world, &procsize);
#endif

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
       << "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
       << std::endl;
    return;
  }

  size_t nrows = A.rows.size();
  LocalOrdinalType ncols = A.num_cols;

  VectorType r(b.startIndex, nrows);
  VectorType p(0, ncols);
  VectorType Ap(b.startIndex, nrows);

  normr = 0;
  typename TypeTraits<ScalarType>::magnitude_type rtrans = 0;
  typename TypeTraits<ScalarType>::magnitude_type oldrtrans = 0;

  LocalOrdinalType print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  TICK(); waxpby(one, x, zero, x, p); TOCK(tWAXPY);

//  print_vec(p.coefs, "p");

  TICK();
  matvec(A, p, Ap);
  TOCK(tMATVEC);

  TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

  TICK(); rtrans = dot_r2(r); TOCK(tDOT);

//std::cout << "rtrans="<<rtrans<<std::endl;

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  typename TypeTraits<ScalarType>::magnitude_type brkdown_tol = std::numeric_limits<typename TypeTraits<ScalarType>::magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

#ifdef MINIFE_DEBUG_OPENMP
  std::cout << "Starting CG Solve Phase..." << std::endl;
#endif

  printf("Enter into the cg iterations ...\n");

  LocalOrdinalType k=1;

  // read checkpoints
  // Read checkpointing either because of recovery being a survivor
  survivor = IsSurvivor();
  if (do_recover || !survivor) {
    printf("RE-Start execution ... \n");
    ApplicationCheckpointRead(survivor, params.cp2f, params.cp2m, params.cp2a, myproc, A, b, x, &normr,my_cg_times,r,p,&rtrans,&oldrtrans,&num_iters);
    k = num_iters;
  }

  for(; k <= max_iter && normr > tolerance; ++k) {

    // write checkpoints
    //printf("cp stride: %d \n", params.cp_stride);
    if ((num_iters%params.cp_stride)==0) {
      ApplicationCheckpointWrite(A, b, x, normr,my_cg_times,r,p,rtrans,oldrtrans, k,myproc,params.cp2f,params.cp2m,params.cp2a);
    }

    if (k == 1) {
      TICK(); waxpby(one, r, zero, r, p); TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot_r2(r); TOCK(tDOT);
      typename TypeTraits<ScalarType>::magnitude_type beta = rtrans/oldrtrans;
      TICK(); daxpby(one, r, beta, p); TOCK(tWAXPY);
    }

    normr = sqrt(rtrans);

    if (params.procfi == 1 && myproc == (procsize-1) && k==61){
      printf("KILL rank %d\n", myproc);
      raise(SIGKILL);
    }

    if (params.nodefi == 1 && myproc == (procsize-1) && k==61){
      char hostname[64];
      gethostname(hostname, 64);
      printf("KILL %s daemon %d rank %d\n", hostname, (int) getppid(), myproc);
      kill(getppid(), SIGKILL);
    }

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    typename TypeTraits<ScalarType>::magnitude_type alpha = 0;
    typename TypeTraits<ScalarType>::magnitude_type p_ap_dot = 0;

    TICK(); matvec(A, p, Ap); TOCK(tMATVEC);
    TICK(); p_ap_dot = dot(Ap, p); TOCK(tDOT);

#ifdef MINIFE_DEBUG
    os << "iter " << k << ", p_ap_dot = " << p_ap_dot;
    os.flush();
#endif
    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 || breakdown(p_ap_dot, Ap, p)) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
#ifdef MINIFE_DEBUG
        os << "ERROR, numerical breakdown!"<<std::endl;
#endif
        //update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = mytimer() - total_time;
        return;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;
#ifdef MINIFE_DEBUG
    os << ", rtrans = " << rtrans << ", alpha = " << alpha << std::endl;
#endif

    TICK(); daxpby(alpha, p, one, x);
            daxpby(-alpha, Ap, one, r); TOCK(tWAXPY);

    num_iters = k;

  }
  printf("Exit the cg iterations ...\n");

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;
}

}//namespace miniFE

#endif
