
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


#include "ma27_interface.hpp"
#include "casadi/core/global_options.hpp"

using namespace std;
namespace casadi {

  extern "C"
  s_t CASADI_LINSOL_MA27_EXPORT
  casadi_register_linsol_ma27(LinsolInternal::Plugin* plugin) {
    plugin->creator = Ma27Interface::creator;
    plugin->name = "ma27";
    plugin->doc = Ma27Interface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &Ma27Interface::options_;
    return 0;
  }

  extern "C"
  void CASADI_LINSOL_MA27_EXPORT casadi_load_linsol_ma27() {
    LinsolInternal::registerPlugin(casadi_register_linsol_ma27);
  }

  Ma27Interface::Ma27Interface(const std::string& name, const Sparsity& sp)
    : LinsolInternal(name, sp) {
  }

  Ma27Interface::~Ma27Interface() {
    clear_mem();
  }

  void Ma27Interface::init(const Dict& opts) {
    // Call the init method of the base class
    LinsolInternal::init(opts);

  }

  r_t Ma27Interface::init_mem(void* mem) const {
    if (LinsolInternal::init_mem(mem)) return 1;
    auto m = static_cast<Ma27Memory*>(mem);

    // Set default options for MA27
    ma27id_(m->icntl, m->cntl);
    m->icntl[0] = 0;       // Suppress error messages
    m->icntl[1] = 0;       // Suppress diagnostic messages
    m->cntl[0] = 1e-8;     // Set pivot tolerance

    // Dynamically resized work vectors
    s_t N = this->ncol();
    s_t nnz = this->nnz();
    double liw_factor = 2;
    m->iw.resize(ceil(liw_factor * (2*nnz+3*N+1)));
    double la_factor = 2;
    m->nz.resize(ceil(la_factor * nnz));
    m->irn.resize(nnz);
    m->jcn.resize(nnz);
    m->iw1.resize(2*N);
    m->ikeep.resize(3*N);
    return 0;
  }

  r_t Ma27Interface::nfact(void* mem, const double* A) const {
    auto m = static_cast<Ma27Memory*>(mem);
    casadi_assert_dev(A!=0);

    // Get sparsity
    const s_t ncol = this->ncol();
    const s_t* colind = this->colind();
    const s_t* row = this->row();

    // Get actual nonzeros
    s_t nnz=0;
    for (s_t cc=0; cc<ncol; ++cc) {
      for (s_t el=colind[cc]; el<colind[cc+1]; ++el) {
        s_t rr=row[el];
        if (rr>cc) continue; // only upper triangular part
        if (A[el]!=0) {
          m->nz[nnz] = A[el];
          m->irn[nnz] = rr+1;
          m->jcn[nnz] = cc+1;
          nnz++;
        }
      }
    }
    m->nnz = nnz;

    // Order of the matrix
    s_t N = this->ncol();

    // Symbolic factorization (MA27AD)
    s_t LIW = m->iw.size();
    s_t iflag = 0;
    s_t info[20];
    double ops;
    ma27ad_(&N, &nnz, get_ptr(m->irn), get_ptr(m->jcn), &m->iw[0], &LIW,
            get_ptr(m->ikeep), get_ptr(m->iw1), &m->nsteps, &iflag, m->icntl, m->cntl,
            info, &ops);
    iflag = info[0];   // Information flag
    s_t ierror = info[1];  // Error flag
    //s_t nrlnec = info[4];  // recommended value for la
    s_t nirnec = info[5];  // recommended value for liw
    casadi_assert(iflag==0,
      "ma27ad_ returns iflag = " + str(iflag) + " with ierror = " + str(ierror));

    // Allocate more memory?
    double la_init_factor = 20.0; // This could be an option.
    s_t la_min = ceil(la_init_factor * nirnec);
    if (la_min > m->nz.size()) m->nz.resize(la_min);
    double liw_init_factor = 5.0; // This could be an option.
    s_t liw_min = ceil(liw_init_factor * nirnec);
    if (liw_min > m->iw.size()) m->iw.resize(liw_min);

    // Numerical factorization (MA27BD)
    s_t LA = m->nz.size();
    LIW = m->iw.size();
    ma27bd_(&N, &nnz, get_ptr(m->irn), get_ptr(m->jcn), get_ptr(m->nz),
           &LA, get_ptr(m->iw), &LIW, get_ptr(m->ikeep), &m->nsteps,
           &m->maxfrt, get_ptr(m->iw1), m->icntl, m->cntl, info);
    iflag = info[0];   // Information flag
    ierror = info[1];  // Error flag
    m->neig = info[14];   // Number of negative eigenvalues
    if (iflag == 3) {
      m->rank = info[1];
    } else if (iflag == -5) {
      //DJ: I think this is more severe. Can this actually happen?
      m->rank = -1;
    } else if (iflag != 0) {
      casadi_error("ma2bd_ returns iflag = " + str(iflag)
        + " with ierror = " + str(ierror));
    } else {
      m->rank = N;
    }

    // Real work array
    if (m->w.size() < m->maxfrt) m->w.resize(m->maxfrt);
    return 0;
  }

  s_t Ma27Interface::neig(void* mem, const double* A) const {
    auto m = static_cast<Ma27Memory*>(mem);
    casadi_assert_dev(m->is_nfact);
    return m->neig;
  }

  s_t Ma27Interface::rank(void* mem, const double* A) const {
    auto m = static_cast<Ma27Memory*>(mem);
    casadi_assert_dev(m->is_nfact);
    return m->rank;
  }

  r_t Ma27Interface::solve(void* mem, const double* A, double* x, s_t nrhs, bool tr) const {
    auto m = static_cast<Ma27Memory*>(mem);

    // Solve for each right-hand-side
    s_t N = this->ncol();
    s_t LA = m->nz.size();
    s_t LIW = m->iw.size();
    for (s_t k=0; k<nrhs; ++k) {
      ma27cd_(&N, get_ptr(m->nz), &LA, get_ptr(m->iw), &LIW, get_ptr(m->w),
              &m->maxfrt, x, get_ptr(m->iw1), &m->nsteps, m->icntl, m->cntl);
      x += N;
    }
    return 0;
  }

  Ma27Memory::Ma27Memory() {
    nnz = -1;
    neig = -1;
    rank = -1;

    nsteps = -1;
    maxfrt = -1;
  }

  Ma27Memory::~Ma27Memory() {
  }

} // namespace casadi
