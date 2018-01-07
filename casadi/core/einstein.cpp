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


#ifndef CASADI_EINSTEIN_CPP
#define CASADI_EINSTEIN_CPP

#include "einstein.hpp"
#include "casadi_misc.hpp"
#include "function_internal.hpp"
#include "runtime/shared.hpp"

using namespace std;

namespace casadi {

  Einstein::Einstein(const MX& C, const MX& A, const MX& B,
    const std::vector<int>& dim_c, const std::vector<int>& dim_a, const std::vector<int>& dim_b,
    const std::vector<int>& c, const std::vector<int>& a, const std::vector<int>& b):
      dim_c_(dim_c), dim_a_(dim_a), dim_b_(dim_b), c_(c), a_(a), b_(b) {

    set_dep(C, A, B);
    set_sparsity(C.sparsity());

    n_iter_ = einstein_process(A, B, C, dim_a, dim_b, dim_c, a, b, c,
      iter_dims_, strides_a_, strides_b_, strides_c_);

  }

  std::string Einstein::disp(const std::vector<std::string>& arg) const {
    return "einstein(" + arg.at(0) + "," + arg.at(1) + "," + arg.at(2) + ")";
  }

  r_t Einstein::eval(const double** arg, double** res, int* iw, double* w) const {
    return eval_gen<double>(arg, res, iw, w);
  }

  r_t Einstein::eval_sx(const SXElem** arg, SXElem** res, int* iw, SXElem* w) const {
    return eval_gen<SXElem>(arg, res, iw, w);
  }

  template<typename T>
  r_t Einstein::eval_gen(const T** arg, T** res, int* iw, T* w) const {
    if (arg[0]!=res[0]) copy(arg[0], arg[0]+dep(0).nnz(), res[0]);

    einstein_eval(n_iter_, iter_dims_, strides_a_, strides_b_, strides_c_, arg[1], arg[2], res[0]);
    return 0;
  }

  void Einstein::ad_forward(const std::vector<std::vector<MX> >& fseed,
                               std::vector<std::vector<MX> >& fsens) const {
    for (int d=0; d<fsens.size(); ++d) {
      fsens[d][0] = fseed[d][0]
       + MX::einstein(dep(1), fseed[d][2], dim_a_, dim_b_, dim_c_, a_, b_, c_)
       + MX::einstein(fseed[d][1], dep(2), dim_a_, dim_b_, dim_c_, a_, b_, c_);
    }
  }

  void Einstein::ad_reverse(const std::vector<std::vector<MX> >& aseed,
                               std::vector<std::vector<MX> >& asens) const {
    for (int d=0; d<aseed.size(); ++d) {
      asens[d][1] += MX::einstein(aseed[d][0], dep(2), dim_c_, dim_b_, dim_a_, c_, b_, a_);
      asens[d][2] += MX::einstein(dep(1), aseed[d][0], dim_a_, dim_c_, dim_b_, a_, c_, b_);
      asens[d][0] += aseed[d][0];
    }
  }



  r_t Einstein::sp_forward(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const {
    return eval_gen<bvec_t>(arg, res, iw, w);
  }

  r_t Einstein::sp_reverse(bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const {
    //int* ind = iw;
    //int cumprod;

    // Main loop
    for (int i=0;i<n_iter_;++i) {

      // Data pointers
      bvec_t* a = arg[1]+strides_a_[0];
      bvec_t* b = arg[2]+strides_b_[0];
      bvec_t* c = res[0]+strides_c_[0];

      // Construct indices
      int sub = i;
      for (int j=0;j<iter_dims_.size();++j) {
        int ind = sub % iter_dims_[j];
        sub/= iter_dims_[j];
        a+= strides_a_[1+j]*ind;
        b+= strides_b_[1+j]*ind;
        c+= strides_c_[1+j]*ind;
      }

      // Perform the actual multiplication
      Contraction<bvec_t>(*c, 0, *a);
      Contraction<bvec_t>(0, *c, *b);
    }
    copy_rev(arg[0], res[0], nnz());
    return 0;
  }

  void Einstein::eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const {
    res[0] = einstein(arg[1], arg[2], arg[0], dim_a_, dim_b_, dim_c_, a_, b_, c_);
  }

  void Einstein::generate(CodeGenerator& g,
                          const std::vector<int>& arg, const std::vector<int>& res) const {

    // Copy first argument if not inplace
    if (arg[0]!=res[0]) {
      g << g.copy(g.work(arg[0], nnz()), nnz(), g.work(res[0], nnz()));
    }

    // main loop
    g.local("i", "int");
    g << "for (i=0; i<" << n_iter_ << "; ++i) {\n";

    // Data pointers
    g.local("cr", "const casadi_real", "*");
    g.local("cs", "const casadi_real", "*");
    g.local("rr", "casadi_real", "*");
    g << "cr = " << g.work(arg[1], dep(1).nnz()) << "+" << strides_a_[0] << ";\n";
    g << "cs = " << g.work(arg[2], dep(2).nnz()) << "+" << strides_b_[0] << ";\n";
    g << "rr = " << g.work(res[0], dep(0).nnz()) << "+" << strides_c_[0] << ";\n";

    // Construct indices
    for (int j=0; j<iter_dims_.size(); ++j) {
      if (j==0) {
        g.local("k", "int");
        g << "k = i;\n";
        g.local("j", "int");
      }
      g << "j = k % " << iter_dims_[j] << ";\n";
      if (j+1<iter_dims_.size()) g << "k /= " << iter_dims_[j] << ";\n";
      if (strides_a_[1+j]) g << "cr += j*" << strides_a_[1+j] << ";\n";
      if (strides_b_[1+j]) g << "cs += j*" << strides_b_[1+j] << ";\n";
      if (strides_c_[1+j]) g << "rr += j*" << strides_c_[1+j] << ";\n";
    }

    // Perform the actual multiplication
    g << "*rr += *cr**cs;\n";

    g << "}\n";
  }

} // namespace casadi

#endif // CASADI_EINSTEIN_CPP
