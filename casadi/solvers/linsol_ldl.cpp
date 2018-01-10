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


#include "linsol_ldl.hpp"
#include "casadi/core/global_options.hpp"

using namespace std;
namespace casadi {

  extern "C"
  r_t CASADI_LINSOL_LDL_EXPORT
  casadi_register_linsol_ldl(LinsolInternal::Plugin* plugin) {
    plugin->creator = LinsolLdl::creator;
    plugin->name = "ldl";
    plugin->doc = LinsolLdl::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &LinsolLdl::options_;
    return 0;
  }

  extern "C"
  void CASADI_LINSOL_LDL_EXPORT casadi_load_linsol_ldl() {
    LinsolInternal::registerPlugin(casadi_register_linsol_ldl);
  }

  LinsolLdl::LinsolLdl(const std::string& name, const Sparsity& sp)
    : LinsolInternal(name, sp) {
  }

  LinsolLdl::~LinsolLdl() {
    clear_mem();
  }

  void LinsolLdl::init(const Dict& opts) {
    // Call the init method of the base class
    LinsolInternal::init(opts);

    // Symbolic factorization
    sp_L_ = sp_.ldl(parent_);
  }

  r_t LinsolLdl::init_mem(void* mem) const {
    if (LinsolInternal::init_mem(mem)) return 1;
    auto m = static_cast<LinsolLdlMemory*>(mem);

    // Work vectors
    s_t nrow = this->nrow();
    m->d.resize(nrow);
    m->l.resize(sp_L_.nnz());
    m->iw.resize(2*nrow);
    m->w.resize(nrow);

    return 0;
  }

  r_t LinsolLdl::sfact(void* mem, const double* A) const {
    return 0;
  }

  r_t LinsolLdl::nfact(void* mem, const double* A) const {
    auto m = static_cast<LinsolLdlMemory*>(mem);
    casadi_ldl(sp_, get_ptr(parent_), sp_L_,
               A, get_ptr(m->l), get_ptr(m->d), get_ptr(m->iw), get_ptr(m->w));
    return 0;
  }

  r_t LinsolLdl::solve(void* mem, const double* A, double* x, s_t nrhs, bool tr) const {
    auto m = static_cast<LinsolLdlMemory*>(mem);
    casadi_ldl_solve(x, nrhs, sp_L_, get_ptr(m->l), get_ptr(m->d));
    return 0;
  }

  s_t LinsolLdl::neig(void* mem, const double* A) const {
    // Count number of negative eigenvalues
    auto m = static_cast<LinsolLdlMemory*>(mem);
    s_t nrow = this->nrow();
    s_t ret = 0;
    for (s_t i=0; i<nrow; ++i) if (m->d[i]<0) ret++;
    return ret;
  }

  s_t LinsolLdl::rank(void* mem, const double* A) const {
    // Count number of nonzero eigenvalues
    auto m = static_cast<LinsolLdlMemory*>(mem);
    s_t nrow = this->nrow();
    s_t ret = 0;
    for (s_t i=0; i<nrow; ++i) if (m->d[i]!=0) ret++;
    return ret;
  }

} // namespace casadi
