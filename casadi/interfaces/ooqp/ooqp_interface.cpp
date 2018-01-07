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


#include "ooqp_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

// OOQP headers
#include <cQpGenSparse.h>
#include <Status.h>
#include <GondzioSolver.h>

// A variable that controls the printlevel of OOQP
// This is the only possible way to access it using the C++ interface
extern s_t gOoqpPrintLevel;

using namespace std;
namespace casadi {

  extern "C"
  s_t CASADI_CONIC_OOQP_EXPORT
  casadi_register_conic_ooqp(Conic::Plugin* plugin) {
    plugin->creator = OoqpInterface::creator;
    plugin->name = "ooqp";
    plugin->doc = OoqpInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &OoqpInterface::options_;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_OOQP_EXPORT casadi_load_conic_ooqp() {
    Conic::registerPlugin(casadi_register_conic_ooqp);
  }

  OoqpInterface::OoqpInterface(const std::string& name,
                               const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {
  }

  OoqpInterface::~OoqpInterface() {
  }

  Options OoqpInterface::options_
  = {{&Conic::options_},
     {{"print_level",
       {OT_INT,
        "Print level. OOQP listens to print_level 0, 10 and 100"}},
      {"mutol",
       {OT_DOUBLE,
        "tolerance as provided with setMuTol to OOQP"}},
      {"artol",
       {OT_DOUBLE,
        "tolerance as provided with setArTol to OOQP"}}
     }
  };

  void OoqpInterface::init(const Dict& opts) {
    // Initialize the base classes
    Conic::init(opts);

    // Default options
    print_level_ = 0;
    mutol_ = 1e-8;
    artol_ = 1e-8;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="print_level") {
        print_level_ = op.second;
      } else if (op.first=="mutol") {
        mutol_ = op.second;
      } else if (op.first=="artol") {
        artol_ = op.second;
      }
    }

    // Allocate memory for problem
    nQ_ = H_.nnz_upper();
    nA_ = nnz_in(CONIC_A);
    nH_ = nnz_in(CONIC_H);
    spAT_ = A_.T();

    // Allocate work vectors
    alloc_w(nx_, true); // g
    alloc_w(nx_, true); // lbx
    alloc_w(nx_, true); // ubx
    alloc_w(na_, true); // lba
    alloc_w(na_, true); // uba
    alloc_w(nH_, true); // H
    alloc_w(nA_, true); // A
    alloc_w(nx_, true); // c_
    alloc_w(na_, true); // bA_
    alloc_w(nx_, true); // xlow_
    alloc_w(nx_, true); // xupp_
    alloc_w(na_, true); // clow_
    alloc_w(na_, true); // cupp_
    alloc_w(nx_, true); // x_
    alloc_w(nx_, true); // gamma_
    alloc_w(nx_, true); // phi_
    alloc_w(na_, true); // y_
    alloc_w(na_, true); // z_
    alloc_w(na_, true); // lambda_
    alloc_w(na_, true); // pi_
    alloc_iw(nx_, true); // ixlow_
    alloc_iw(nx_, true); // ixupp_
    alloc_iw(na_, true); // iclow_
    alloc_iw(na_, true); // icupp_
    alloc_w(nQ_, true); // dQ_
    alloc_w(nA_, true); // dA_
    alloc_w(nA_, true); // dC_
    alloc_iw(nQ_, true); // irowQ_
    alloc_iw(nQ_, true); // jcolQ_
    alloc_iw(nA_, true); // irowA_
    alloc_iw(nA_, true); // jcolA_
    alloc_iw(nA_, true); // irowC_
    alloc_iw(nA_, true); // jcolC_
    alloc_iw(nx_, true); // x_index_
    alloc_iw(na_, true); // c_index_
    alloc_w(nx_, true); // p_
    alloc_w(nA_, true); // AT
    alloc_iw(na_); // casadi_trans
  }

  s_t OoqpInterface::
  eval(const double** arg, double** res, s_t* iw, double* w, void* mem) const {
    if (inputs_check_) {
      check_inputs(arg[CONIC_LBX], arg[CONIC_UBX], arg[CONIC_LBA], arg[CONIC_UBA]);
    }

    // Get problem data
    double* g=w; w += nx_;
    casadi_copy(arg[CONIC_G], nx_, g);
    double* lbx=w; w += nx_;
    casadi_copy(arg[CONIC_LBX], nx_, lbx);
    double* ubx=w; w += nx_;
    casadi_copy(arg[CONIC_UBX], nx_, ubx);
    double* lba=w; w += na_;
    casadi_copy(arg[CONIC_LBA], na_, lba);
    double* uba=w; w += na_;
    casadi_copy(arg[CONIC_UBA], na_, uba);
    double* H=w; w += nnz_in(CONIC_H);
    casadi_copy(arg[CONIC_H], nnz_in(CONIC_H), H);
    double* A=w; w += nnz_in(CONIC_A);
    casadi_copy(arg[CONIC_A], nnz_in(CONIC_A), A);

    // Temporary memory
    double* c_ = w; w += nx_;
    double* bA_ = w; w += na_;
    double* xlow_ = w; w += nx_;
    double* xupp_ = w; w += nx_;
    double* clow_ = w; w += na_;
    double* cupp_ = w; w += na_;
    double* x_ = w; w += nx_;
    double* gamma_ = w; w += nx_;
    double* phi_ = w; w += nx_;
    double* y_ = w; w += na_;
    double* z_ = w; w += na_;
    double* lambda_ = w; w += na_;
    double* pi_ = w; w += na_;
    char* ixlow_ = reinterpret_cast<char*>(iw); iw += nx_;
    char* ixupp_ = reinterpret_cast<char*>(iw); iw += nx_;
    char* iclow_ = reinterpret_cast<char*>(iw); iw += na_;
    char* icupp_ = reinterpret_cast<char*>(iw); iw += na_;
    double* dQ_ = w; w += nQ_;
    double* dA_ = w; w += nA_;
    double* dC_ = w; w += nA_;
    s_t* irowQ_ = iw; iw += nQ_;
    s_t* jcolQ_ = iw; iw += nQ_;
    s_t* irowA_ = iw; iw += nA_;
    s_t* jcolA_ = iw; iw += nA_;
    s_t* irowC_ = iw; iw += nA_;
    s_t* jcolC_ = iw; iw += nA_;
    s_t* x_index_ = iw; iw += nx_;
    s_t* c_index_ = iw; iw += na_;
    double* p_ = w; w += nx_;
    double* AT = w; w += nA_;

    // Parameter contribution to the objective
    double objParam = 0;

    // Get the number of free variables and their types
    s_t nx = 0, np=0;
    for (s_t i=0; i<nx_; ++i) {
      if (lbx[i]==ubx[i]) {
        // Save parameter
        p_[np] = lbx[i];

        // Add contribution to objective
        objParam += g[i]*p_[np];

        // Save index
        x_index_[i] = -1-np++;

      } else {
        // True free variable
        if (lbx[i]==-numeric_limits<double>::infinity()) {
          xlow_[nx] = 0;
          ixlow_[nx] = 0;
        } else {
          xlow_[nx] = lbx[i];
          ixlow_[nx] = 1;
        }
        if (ubx[i]==numeric_limits<double>::infinity()) {
          xupp_[nx] = 0;
          ixupp_[nx] = 0;
        } else {
          xupp_[nx] = ubx[i];
          ixupp_[nx] = 1;
        }
        c_[nx] = g[i];
        x_index_[i] = nx++;
      }
    }

    // Get quadratic term
    const s_t* H_colind = H_.colind();
    const s_t* H_row = H_.row();
    s_t nnzQ = 0;
    // Loop over the columns of the quadratic term
    for (s_t cc=0; cc<nx_; ++cc) {

      // Loop over nonzero elements of the column
      for (s_t el=H_colind[cc]; el<H_colind[cc+1]; ++el) {

        // Only upper triangular part
        s_t rr=H_row[el];
        if (rr>cc) break;

        // Get variable types
        s_t icc=x_index_[cc];
        s_t irr=x_index_[rr];

        if (icc<0) {
          if (irr<0) {
            // Add contribution to objective
            objParam += icc==irr ? H[el]*sq(p_[-1-icc])/2 : H[el]*p_[-1-irr]*p_[-1-icc];
          } else {
            // Add contribution to gradient term
            c_[irr] += H[el]*p_[-1-icc];
          }
        } else {
          if (irr<0) {
            // Add contribution to gradient term
            c_[icc] += H[el]*p_[-1-irr];
          } else {
            // Add to sparsity pattern
            irowQ_[nnzQ] = icc; // row-major --> indices swapped
            jcolQ_[nnzQ] = irr; // row-major --> indices swapped
            dQ_[nnzQ++] = H[el];
          }
        }
      }
    }

    // Get the transpose of the sparsity pattern to be able to loop over the constraints
    casadi_trans(A, A_, AT, spAT_, iw);

    // Loop over constraints
    const s_t* A_colind = A_.colind();
    const s_t* A_row = A_.row();
    const s_t* AT_colind = spAT_.colind();
    const s_t* AT_row = spAT_.row();
    s_t nA=0, nC=0, /*mz=0, */ nnzA=0, nnzC=0;
    for (s_t j=0; j<na_; ++j) {
      if (lba[j] == -numeric_limits<double>::infinity() &&
          uba[j] ==  numeric_limits<double>::infinity()) {
        // Redundant constraint
        c_index_[j] = 0;
      } else if (lba[j]==uba[j]) {
        // Equality constraint
        bA_[nA] = lba[j];

        // Add to A
        for (s_t el=AT_colind[j]; el<AT_colind[j+1]; ++el) {
          s_t i=AT_row[el];
          if (x_index_[i]<0) {
            // Parameter
            bA_[nA] -= AT[el]*p_[-x_index_[i]-1];
          } else {
            // Free variable
            irowA_[nnzA] = nA;
            jcolA_[nnzA] = x_index_[i];
            dA_[nnzA++] = AT[el];
          }
        }
        c_index_[j] = -1-nA++;
      } else {
        // Inequality constraint
        if (lba[j]==-numeric_limits<double>::infinity()) {
          clow_[nC] = 0;
          iclow_[nC] = 0;
        } else {
          clow_[nC] = lba[j];
          iclow_[nC] = 1;
        }
        if (uba[j]==numeric_limits<double>::infinity()) {
          cupp_[nC] = 0;
          icupp_[nC] = 0;
        } else {
          cupp_[nC] = uba[j];
          icupp_[nC] = 1;
        }

        // Add to C
        for (s_t el=AT_colind[j]; el<AT_colind[j+1]; ++el) {
          s_t i=AT_row[el];
          if (x_index_[i]<0) {
            // Parameter
            if (iclow_[nC]==1) clow_[nC] -= AT[el]*p_[-x_index_[i]-1];
            if (icupp_[nC]==1) cupp_[nC] -= AT[el]*p_[-x_index_[i]-1];
          } else {
            // Free variable
            irowC_[nnzC] = nC;
            jcolC_[nnzC] = x_index_[i];
            dC_[nnzC++] = AT[el];
          }
        }
        c_index_[j] = 1+nC++;
      }
    }

    // Reset the solution
    casadi_fill(x_, nx_, 0.);
    casadi_fill(gamma_, nx_, 0.);
    casadi_fill(phi_, nx_, 0.);
    casadi_fill(y_, na_, 0.);
    casadi_fill(z_, na_, 0.);
    casadi_fill(lambda_, na_, 0.);
    casadi_fill(pi_, na_, 0.);

    // Solve the QP
    double objectiveValue;

    s_t ierr;
    if (false) { // Use C interface
      // TODO(jgillis): Change to conicvehb, see OOQP users guide
      qpsolvesp(c_, nx,
                irowQ_,  nnzQ, jcolQ_, dQ_,
                xlow_, ixlow_,
                xupp_, ixupp_,
                irowA_, nnzA, jcolA_, dA_,
                bA_, nA,
                irowC_, nnzC, jcolC_, dC_,
                clow_, nC, iclow_,
                cupp_, icupp_,
                x_, gamma_, phi_,
                y_,
                z_, lambda_, pi_,
                &objectiveValue,
                print_level_, &ierr);
    } else { // Use C++ interface
      ierr=0;
      // All OOQP related allocations in evaluate

      std::vector<s_t> krowQ(nx+1);
      std::vector<s_t> krowA(nA+1);
      std::vector<s_t> krowC(nC+1);

      //s_t status_code = 0;
      makehb(irowQ_, nnzQ, get_ptr(krowQ), nx, &ierr);
      if (ierr == 0) makehb(irowA_, nnzA, get_ptr(krowA), nA, &ierr);
      if (ierr == 0) makehb(irowC_, nnzC, get_ptr(krowC), nC, &ierr);

      if (ierr == 0) {
        QpGenContext ctx;

        QpGenHbGondzioSetup(c_, nx, get_ptr(krowQ), jcolQ_, dQ_,
                            xlow_, ixlow_, xupp_, ixupp_,
                            get_ptr(krowA), nA, jcolA_, dA_, bA_,
                            get_ptr(krowC), nC, jcolC_, dC_,
                            clow_, iclow_, cupp_, icupp_, &ctx,
                            &ierr);
        if (ierr == 0) {
          Solver* solver = static_cast<Solver *>(ctx.solver);
          gOoqpPrintLevel = print_level_;
          solver->monitorSelf();
          solver->setMuTol(mutol_);
          solver->setMuTol(mutol_);

          QpGenFinish(&ctx, x_, gamma_, phi_,
                      y_, z_, lambda_, pi_,
                      &objectiveValue, &ierr);
        }

        QpGenCleanup(&ctx);
      }
    }

    if (ierr>0) {
      casadi_warning("Unable to solve problem: " + str(errFlag(ierr)));
    } else if (ierr<0) {
      casadi_error("Fatal error: " + str(errFlag(ierr)));
    }

    // Retrieve eliminated decision variables
    for (s_t i=nx_-1; i>=0; --i) {
      s_t ii = x_index_[i];
      if (ii<0) {
        x_[i] = p_[-1-ii];
      } else {
        x_[i] = x_[ii];
      }
    }

    // Retreive eliminated dual variables (linear bounds)
    for (s_t j=na_-1; j>=0; --j) {
      s_t jj = c_index_[j];
      if (jj==0) {
        lambda_[j] = 0;
      } else if (jj<0) {
        lambda_[j] = -y_[-1-jj];
      } else {
        lambda_[j] = pi_[-1+jj]-lambda_[-1+jj];
      }
    }

    // Retreive eliminated dual variables (simple bounds)
    for (s_t i=nx_-1; i>=0; --i) {
      s_t ii = x_index_[i];
      if (ii<0) {
        // The dual solution for the fixed parameters follows from the KKT conditions
        gamma_[i] = -g[i];
        for (s_t el=H_colind[i]; el<H_colind[i+1]; ++el) {
          s_t j=H_row[el];
          gamma_[i] -= H[el]*x_[j];
        }
        for (s_t el=A_colind[i]; el<A_colind[i+1]; ++el) {
          s_t j=A_row[el];
          gamma_[i] -= A[el]*lambda_[j];
        }
      } else {
        gamma_[i] = phi_[ii]-gamma_[ii];
      }
    }

    // Save optimal cost
    if (res[CONIC_COST]) *res[CONIC_COST] = objectiveValue + objParam;

    // Save primal solution
    casadi_copy(x_, nx_, res[CONIC_X]);

    // Save dual solution (linear bounds)
    casadi_copy(lambda_, na_, res[CONIC_LAM_A]);

    // Save dual solution (simple bounds)
    casadi_copy(gamma_, nx_, res[CONIC_LAM_X]);
    return 0;
  }

  const char* OoqpInterface::errFlag(s_t flag) {
    // Find the error
    //const char* msg;
    switch (flag) {
    case SUCCESSFUL_TERMINATION: return  "SUCCESSFUL_TERMINATION";
    case NOT_FINISHED:           return  "NOT_FINISHED";
    case MAX_ITS_EXCEEDED:       return  "MAX_ITS_EXCEEDED";
    case INFEASIBLE:             return  "INFEASIBLE";
    case UNKNOWN:                return  "UNKNOWN";
    default:                     return  "N/A";
    }
  }

  std::string OoqpInterface::printBounds(const std::vector<double>& b,
                                        const std::vector<char>& ib, s_t n, const char *sign) {
    stringstream ss;
    ss << "[";
    for (s_t i=0; i<n; ++i) {
      if (i!=0) ss << ", ";
      if (ib[i]==0) {
        ss << sign << "inf";
      } else {
        ss << b[i];
      }
    }
    ss << "]";
    return ss.str();
  }


} // namespace casadi
