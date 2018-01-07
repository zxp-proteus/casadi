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


#include "map.hpp"

using namespace std;

namespace casadi {

  Function Map::create(const std::string& parallelization, const Function& f, s_t n) {
    // Create instance of the right class
    string name = f.name() + "_" + str(n);
    if (parallelization == "serial") {
      return Function::create(new Map(name, f, n), Dict());
    } else if (parallelization== "openmp") {
      return Function::create(new MapOmp(name, f, n), Dict());
    } else {
      casadi_error("Unknown parallelization: " + parallelization);
    }
  }

  Map::Map(const std::string& name, const Function& f, s_t n)
    : FunctionInternal(name), f_(f), n_(n) {
  }

  Map::~Map() {
  }

  void Map::init(const Dict& opts) {
    // Call the initialization method of the base class
    FunctionInternal::init(opts);

    // Allocate sufficient memory for serial evaluation
    alloc_arg(f_.sz_arg());
    alloc_res(f_.sz_res());
    alloc_w(f_.sz_w());
    alloc_iw(f_.sz_iw());
  }

  template<typename T>
  r_t Map::eval_gen(const T** arg, T** res, s_t* iw, T* w) const {
    const T** arg1 = arg+n_in_;
    copy_n(arg, n_in_, arg1);
    T** res1 = res+n_out_;
    copy_n(res, n_out_, res1);
    for (s_t i=0; i<n_; ++i) {
      if (f_(arg1, res1, iw, w)) return 1;
      for (s_t j=0; j<n_in_; ++j) {
        if (arg1[j]) arg1[j] += f_.nnz_in(j);
      }
      for (s_t j=0; j<n_out_; ++j) {
        if (res1[j]) res1[j] += f_.nnz_out(j);
      }
    }
    return 0;
  }

  r_t Map::eval_sx(const SXElem** arg, SXElem** res, s_t* iw, SXElem* w, void* mem) const {
    return eval_gen(arg, res, iw, w);
  }

  r_t Map::sp_forward(const bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w, void* mem) const {
    return eval_gen(arg, res, iw, w);
  }

  r_t Map::sp_reverse(bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w, void* mem) const {
    bvec_t** arg1 = arg+n_in_;
    copy_n(arg, n_in_, arg1);
    bvec_t** res1 = res+n_out_;
    copy_n(res, n_out_, res1);
    for (s_t i=0; i<n_; ++i) {
      if (f_.rev(arg1, res1, iw, w)) return 1;
      for (s_t j=0; j<n_in_; ++j) {
        if (arg1[j]) arg1[j] += f_.nnz_in(j);
      }
      for (s_t j=0; j<n_out_; ++j) {
        if (res1[j]) res1[j] += f_.nnz_out(j);
      }
    }
    return 0;
  }

  void Map::codegen_declarations(CodeGenerator& g) const {
    g.add_dependency(f_);
  }

  void Map::codegen_body(CodeGenerator& g) const {
    g << "s_t i;\n";
    // Input buffer
    g << "const casadi_real** arg1 = arg+" << n_in_ << ";\n"
      << "for (i=0; i<" << n_in_ << "; ++i) arg1[i]=arg[i];\n";
    // Output buffer
    g << "casadi_real** res1 = res+" << n_out_ << ";\n"
      << "for (i=0; i<" << n_out_ << "; ++i) res1[i]=res[i];\n"
      << "for (i=0; i<" << n_ << "; ++i) {\n";
    // Evaluate
    g << "if (" << g(f_, "arg1", "res1", "iw", "w") << ") return 1;\n";
    // Update input buffers
    for (s_t j=0; j<n_in_; ++j) {
      g << "if (arg1[" << j << "]) arg1[" << j << "]+=" << f_.nnz_in(j) << ";\n";
    }
    // Update output buffers
    for (s_t j=0; j<n_out_; ++j) {
      g << "if (res1[" << j << "]) res1[" << j << "]+=" << f_.nnz_out(j) << ";\n";
    }
    g << "}\n";
  }

  Function Map
  ::get_forward(s_t nfwd, const std::string& name,
                const std::vector<std::string>& inames,
                const std::vector<std::string>& onames,
                const Dict& opts) const {
    // Generate map of derivative
    Function df = f_.forward(nfwd);
    Function dm = df.map(n_, parallelization());

    // Input expressions
    vector<MX> arg = dm.mx_in();

    // Need to reorder sensitivity inputs
    vector<MX> res = arg;
    vector<MX>::iterator it=res.begin()+n_in_+n_out_;
    vector<s_t> ind;
    for (s_t i=0; i<n_in_; ++i, ++it) {
      s_t sz = f_.size2_in(i);
      ind.clear();
      for (s_t k=0; k<n_; ++k) {
        for (s_t d=0; d<nfwd; ++d) {
          for (s_t j=0; j<sz; ++j) {
            ind.push_back((d*n_ + k)*sz + j);
          }
        }
      }
      *it = (*it)(Slice(), ind);
    }

    // Get output expressions
    res = dm(res);

    // Reorder sensitivity outputs
    it = res.begin();
    for (s_t i=0; i<n_out_; ++i, ++it) {
      s_t sz = f_.size2_out(i);
      ind.clear();
      for (s_t d=0; d<nfwd; ++d) {
        for (s_t k=0; k<n_; ++k) {
          for (s_t j=0; j<sz; ++j) {
            ind.push_back((k*nfwd + d)*sz + j);
          }
        }
      }
      *it = (*it)(Slice(), ind);
    }

    // Construct return function
    return Function(name, arg, res, inames, onames, opts);
  }

  Function Map
  ::get_reverse(s_t nadj, const std::string& name,
                const std::vector<std::string>& inames,
                const std::vector<std::string>& onames,
                const Dict& opts) const {
    // Generate map of derivative
    Function df = f_.reverse(nadj);
    Function dm = df.map(n_, parallelization());

    // Input expressions
    vector<MX> arg = dm.mx_in();

    // Need to reorder sensitivity inputs
    vector<MX> res = arg;
    vector<MX>::iterator it=res.begin()+n_in_+n_out_;
    vector<s_t> ind;
    for (s_t i=0; i<n_out_; ++i, ++it) {
      s_t sz = f_.size2_out(i);
      ind.clear();
      for (s_t k=0; k<n_; ++k) {
        for (s_t d=0; d<nadj; ++d) {
          for (s_t j=0; j<sz; ++j) {
            ind.push_back((d*n_ + k)*sz + j);
          }
        }
      }
      *it = (*it)(Slice(), ind);
    }

    // Get output expressions
    res = dm(res);

    // Reorder sensitivity outputs
    it = res.begin();
    for (s_t i=0; i<n_in_; ++i, ++it) {
      s_t sz = f_.size2_in(i);
      ind.clear();
      for (s_t d=0; d<nadj; ++d) {
        for (s_t k=0; k<n_; ++k) {
          for (s_t j=0; j<sz; ++j) {
            ind.push_back((k*nadj + d)*sz + j);
          }
        }
      }
      *it = (*it)(Slice(), ind);
    }

    // Construct return function
    return Function(name, arg, res, inames, onames, opts);
  }

  r_t Map::eval(const double** arg, double** res, s_t* iw, double* w, void* mem) const {
    return eval_gen(arg, res, iw, w);
  }

  MapOmp::~MapOmp() {
  }

  r_t MapOmp::eval(const double** arg, double** res, s_t* iw, double* w, void* mem) const {
#ifndef WITH_OPENMP
    return Map::eval(arg, res, iw, w, mem);
#else // WITH_OPENMP
    size_t sz_arg, sz_res, sz_iw, sz_w;
    f_.sz_work(sz_arg, sz_res, sz_iw, sz_w);

    // Error flag
    s_t flag = 0;

    // Checkout memory objects
    s_t* ind = iw; iw += n_;
    for (s_t i=0; i<n_; ++i) ind[i] = f_.checkout();

    // Evaluate in parallel
#pragma omp parallel for reduction(||:flag)
    for (s_t i=0; i<n_; ++i) {
      // Input buffers
      const double** arg1 = arg + n_in_ + i*sz_arg;
      for (s_t j=0; j<n_in_; ++j) {
        arg1[j] = arg[j] ? arg[j] + i*f_.nnz_in(j) : 0;
      }

      // Output buffers
      double** res1 = res + n_out_ + i*sz_res;
      for (s_t j=0; j<n_out_; ++j) {
        res1[j] = res[j] ? res[j] + i*f_.nnz_out(j) : 0;
      }

      // Evaluation
      flag = f_(arg1, res1, iw + i*sz_iw, w + i*sz_w, ind[i]) || flag;
    }
    // Release memory objects
    for (s_t i=0; i<n_; ++i) f_.release(ind[i]);
    // Return error flag
    return flag;
#endif  // WITH_OPENMP
  }

  void MapOmp::codegen_body(CodeGenerator& g) const {
    size_t sz_arg, sz_res, sz_iw, sz_w;
    f_.sz_work(sz_arg, sz_res, sz_iw, sz_w);
    g << "s_t i;\n"
      << "const double** arg1;\n"
      << "double** res1;\n"
      << "s_t flag = 0;\n"
      << "#pragma omp parallel for private(i,arg1,res1) reduction(||:flag)\n"
      << "for (i=0; i<" << n_ << "; ++i) {\n"
      << "arg1 = arg + " << n_in_ << "+i*" << sz_arg << ";\n";
    for (s_t j=0; j<n_in_; ++j) {
      g << "arg1[" << j << "] = arg[" << j << "] ? "
        << "arg[" << j << "]+i*" << f_.nnz_in(j) << ": 0;\n";
    }
    g << "res1 = res + " <<  n_out_ << "+i*" <<  sz_res << ";\n";
    for (s_t j=0; j<n_out_; ++j) {
      g << "res1[" << j << "] = res[" << j << "] ?"
        << "res[" << j << "]+i*" << f_.nnz_out(j) << ": 0;\n";
    }
    g << "flag = "
      << g(f_, "arg1", "res1", "iw+i*" + str(sz_iw), "w+i*" + str(sz_w)) << " || flag;\n"
      << "}\n"
      << "if (flag) return 1;\n";
  }

  void MapOmp::init(const Dict& opts) {
    // Call the initialization method of the base class
    Map::init(opts);

    // Allocate memory for holding memory object references
    alloc_iw(n_, true);

    // Allocate sufficient memory for parallel evaluation
    alloc_arg(f_.sz_arg() * n_);
    alloc_res(f_.sz_res() * n_);
    alloc_w(f_.sz_w() * n_);
    alloc_iw(f_.sz_iw() * n_);
  }

} // namespace casadi
