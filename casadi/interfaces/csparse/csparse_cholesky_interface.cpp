/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "csparse_cholesky_interface.hpp"

/// \cond INTERFACE

using namespace std;
namespace casadi {

  extern "C"
  s_t CASADI_LINSOL_CSPARSECHOLESKY_EXPORT
  casadi_register_linsol_csparsecholesky(LinsolInternal::Plugin* plugin) {
    plugin->creator = CSparseCholeskyInterface::creator;
    plugin->name = "csparsecholesky";
    plugin->doc = CSparseCholeskyInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &CSparseCholeskyInterface::options_;
    return 0;
  }

  extern "C"
  void CASADI_LINSOL_CSPARSECHOLESKY_EXPORT casadi_load_linsol_csparsecholesky() {
    LinsolInternal::registerPlugin(casadi_register_linsol_csparsecholesky);
  }

  CSparseCholeskyInterface::
  CSparseCholeskyInterface(const std::string& name, const Sparsity& sp) :
    LinsolInternal(name, sp) {
  }

  CSparseCholeskyInterface::~CSparseCholeskyInterface() {
    clear_mem();
  }

  CsparseCholMemory::~CsparseCholMemory() {
    if (this->S) cs_sfree(this->S);
    if (this->L) cs_nfree(this->L);
  }

  void CSparseCholeskyInterface::init(const Dict& opts) {
    // Call the init method of the base class
    LinsolInternal::init(opts);
  }

  r_t CSparseCholeskyInterface::init_mem(void* mem) const {
    if (LinsolInternal::init_mem(mem)) return 1;
    auto m = static_cast<CsparseCholMemory*>(mem);

    m->L = 0;
    m->S = 0;
    m->A.nzmax = this->nnz();  // maximum number of entries
    m->A.m = this->nrow(); // number of columns
    m->A.n = this->ncol(); // number of rows
    m->A.p = const_cast<s_t*>(this->colind()); // row pointers (size n+1)
    // or row indices (size nzmax)
    m->A.i = const_cast<s_t*>(this->row()); // column indices, size nzmax
    m->A.x = 0; // numerical values, size nzmax
    m->A.nz = -1; // of entries in triplet matrix, -1 for compressed-row

    // Temporary
    m->temp.resize(m->A.n);

    return 0;
  }

  r_t CSparseCholeskyInterface::sfact(void* mem, const double* A) const {
    auto m = static_cast<CsparseCholMemory*>(mem);

    // Set the nonzeros of the matrix
    m->A.x = const_cast<double*>(A);

    // ordering and symbolic analysis
    s_t order = 0; // ordering?
    m->S = cs_schol(order, &m->A);
    return 0;
  }

  r_t CSparseCholeskyInterface::nfact(void* mem, const double* A) const {
    auto m = static_cast<CsparseCholMemory*>(mem);

    // Set the nonzeros of the matrix
    m->A.x = const_cast<double*>(A);

    // Make sure that all entries of the linear system are valid
    s_t nnz = this->nnz();
    for (s_t k=0; k<nnz; ++k) {
      casadi_assert(!isnan(A[k]),
        "Nonzero " + str(k) + " is not-a-number");
      casadi_assert(!isinf(A[k]),
        "Nonzero " + str(k) + " is infinite");
    }

    if (m->L) cs_nfree(m->L);
    m->L = cs_chol(&m->A, m->S) ;                 // numeric Cholesky factorization
    casadi_assert_dev(m->L!=0);
    return 0;
  }

  s_t CSparseCholeskyInterface::
  solve(void* mem, const double* A, double* x, s_t nrhs, bool tr) const {
    auto m = static_cast<CsparseCholMemory*>(mem);

    casadi_assert_dev(m->L!=0);

    double *t = &m->temp.front();
    for (s_t k=0; k<nrhs; ++k) {
      if (tr) {
        cs_pvec(m->S->q, x, t, m->A.n) ;   // t = P1\b
        cs_ltsolve(m->L->L, t) ;               // t = L\t
        cs_lsolve(m->L->L, t) ;              // t = U\t
        cs_pvec(m->L->pinv, t, x, m->A.n) ;      // x = P2\t
      } else {
        cs_ipvec(m->L->pinv, x, t, m->A.n) ;   // t = P1\b
        cs_lsolve(m->L->L, t) ;               // t = L\t
        cs_ltsolve(m->L->L, t) ;              // t = U\t
        cs_ipvec(m->S->q, t, x, m->A.n) ;      // x = P2\t
      }
      x += this->ncol();
    }
    return 0;
  }

} // namespace casadi

/// \endcond
