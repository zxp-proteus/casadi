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


#include "project.hpp"
#include <vector>
#include <sstream>
#include "casadi_misc.hpp"

using namespace std;

namespace casadi {

  Project::Project(const MX& x, const Sparsity& sp) {
    set_dep(x);
    set_sparsity(Sparsity(sp));
  }

  std::string Project::disp(const std::vector<std::string>& arg) const {
    if (sparsity().is_dense()) {
      return "dense(" + arg.at(0) + ")";
    } else {
      return "project(" + arg.at(0) + ")";
    }
  }

  template<typename T>
  r_t Project::eval_gen(const T** arg, T** res, s_t* iw, T* w) const {
    casadi_project(arg[0], dep().sparsity(), res[0], sparsity(), w);
    return 0;
  }

  r_t Project::eval(const double** arg, double** res, s_t* iw, double* w) const {
    return eval_gen<double>(arg, res, iw, w);
  }

  r_t Project::eval_sx(const SXElem** arg, SXElem** res, s_t* iw, SXElem* w) const {
    return eval_gen<SXElem>(arg, res, iw, w);
  }

  void Project::eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const {
    res[0] = project(arg[0], sparsity());
  }

  void Project::ad_forward(const std::vector<std::vector<MX> >& fseed,
                          std::vector<std::vector<MX> >& fsens) const {
    s_t nfwd = fsens.size();
    for (s_t d=0; d<nfwd; ++d) {
      fsens[d][0] = project(fseed[d][0], sparsity(), true);
    }
  }

  void Project::ad_reverse(const std::vector<std::vector<MX> >& aseed,
                          std::vector<std::vector<MX> >& asens) const {
    s_t nadj = aseed.size();
    for (s_t d=0; d<nadj; ++d) {
      asens[d][0] += project(aseed[d][0], dep().sparsity(), true);
    }
  }

  r_t Project::sp_forward(const bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w) const {
    sparsity().set(res[0], arg[0], dep().sparsity());
    return 0;
  }

  r_t Project::sp_reverse(bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w) const {
    dep().sparsity().bor(arg[0], res[0], sparsity());
    fill(res[0], res[0]+nnz(), 0);
    return 0;
  }

  void Project::generate(CodeGenerator& g,
                         const std::vector<s_t>& arg, const std::vector<s_t>& res) const {
    g << g.project(g.work(arg.front(), dep().nnz()), dep(0).sparsity(),
                           g.work(res.front(), nnz()), sparsity(), "w") << "\n";
  }


} // namespace casadi
