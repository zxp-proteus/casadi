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


#ifndef CASADI_TRANSPOSE_HPP
#define CASADI_TRANSPOSE_HPP

#include "mx_node.hpp"
#include <map>
#include <stack>

/// \cond INTERNAL

namespace casadi {
  /** \brief Matrix transpose
      \author Joel Andersson
      \date 2013
  */
  class CASADI_EXPORT Transpose : public MXNode {
  public:

    /// Constructor
    Transpose(const MX& x);

    /// Destructor
    ~Transpose() override {}

    /// Evaluate the function (template)
    template<typename T>
    r_t eval_gen(const T* const* arg, T* const* res, int* iw, T* w) const;

    /// Evaluate the function numerically
    r_t eval(const double** arg, double** res, int* iw, double* w) const override;

    /// Evaluate the function symbolically (SX)
    r_t eval_sx(const SXElem** arg, SXElem** res, int* iw, SXElem* w) const override;

    /** \brief  Evaluate symbolically (MX) */
    void eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const override;

    /** \brief Calculate forward mode directional derivatives */
    void ad_forward(const std::vector<std::vector<MX> >& fseed,
                         std::vector<std::vector<MX> >& fsens) const override;

    /** \brief Calculate reverse mode directional derivatives */
    void ad_reverse(const std::vector<std::vector<MX> >& aseed,
                         std::vector<std::vector<MX> >& asens) const override;

    /** \brief  Propagate sparsity forward */
    r_t sp_forward(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const override;

    /** \brief  Propagate sparsity backwards */
    r_t sp_reverse(bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const override;

    /** \brief  Print expression */
    std::string disp(const std::vector<std::string>& arg) const override;

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g,
                          const std::vector<int>& arg, const std::vector<int>& res) const override;

    /** \brief Get the operation */
    e_t op() const override { return OP_TRANSPOSE;}

    /** \brief Get required length of iw field */
    size_t sz_iw() const override { return size2()+1;}

    /// Transpose
    MX get_transpose() const override { return dep();}

    /// Solve for square linear system
    //virtual MX get_solve(const MX& r, bool tr, const Linsol& linear_solver) const {
    // return dep()->get_solve(r, !tr, linear_solver);} // FIXME #1001

    /** \brief Check if two nodes are equivalent up to a given depth */
    bool is_equal(const MXNode* node, int depth) const override {
      return sameOpAndDeps(node, depth);
    }
  };

  /** \brief Matrix transpose (dense)
      \author Joel Andersson
      \date 2013
  */
  class CASADI_EXPORT DenseTranspose : public Transpose {
  public:

    /// Constructor
    DenseTranspose(const MX& x) : Transpose(x) {}

    /// Destructor
    ~DenseTranspose() override {}

    /// Evaluate the function (template)
    template<typename T>
    r_t eval_gen(const T* const* arg, T* const* res, int* iw, T* w) const;

    /// Evaluate the function numerically
    r_t eval(const double** arg, double** res, int* iw, double* w) const override;

    /// Evaluate the function symbolically (SX)
    r_t eval_sx(const SXElem** arg, SXElem** res, int* iw, SXElem* w) const override;

    /** \brief  Propagate sparsity forward */
    r_t sp_forward(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const override;

    /** \brief  Propagate sparsity backwards */
    r_t sp_reverse(bvec_t** arg, bvec_t** res, int* iw, bvec_t* w) const override;

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g,
                          const std::vector<int>& arg, const std::vector<int>& res) const override;

    /** \brief Get required length of iw field */
    size_t sz_iw() const override { return 0;}
  };



} // namespace casadi

/// \endcond

#endif // CASADI_TRANSPOSE_HPP
