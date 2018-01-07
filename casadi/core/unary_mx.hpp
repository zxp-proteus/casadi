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


#ifndef CASADI_UNARY_MX_HPP
#define CASADI_UNARY_MX_HPP

#include "mx_node.hpp"
/// \cond INTERNAL

namespace casadi {
  /** \brief Represents a general unary operation on an MX
      \author Joel Andersson
      \date 2010
  */
  class CASADI_EXPORT UnaryMX : public MXNode {
  public:

    /** \brief  Constructor is private, use "create" below */
    UnaryMX(Operation op, MX x);

    /** \brief  Destructor */
    ~UnaryMX() override {}

    /** \brief  Print expression */
    std::string disp(const std::vector<std::string>& arg) const override;

    /// Evaluate the function numerically
    r_t eval(const double** arg, double** res, s_t* iw, double* w) const override;

    /// Evaluate the function symbolically (SX)
    r_t eval_sx(const SXElem** arg, SXElem** res, s_t* iw, SXElem* w) const override;

    /** \brief  Evaluate symbolically (MX) */
    void eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const override;

    /** \brief Calculate forward mode directional derivatives */
    void ad_forward(const std::vector<std::vector<MX> >& fseed,
                         std::vector<std::vector<MX> >& fsens) const override;

    /** \brief Calculate reverse mode directional derivatives */
    void ad_reverse(const std::vector<std::vector<MX> >& aseed,
                         std::vector<std::vector<MX> >& asens) const override;

    /** \brief  Propagate sparsity forward */
    r_t sp_forward(const bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w) const override;

    /** \brief  Propagate sparsity backwards */
    r_t sp_reverse(bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w) const override;

    /** \brief Check if unary operation */
    bool is_unary() const override { return true;}

    /** \brief Get the operation */
    e_t op() const override { return op_;}

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g,
                          const std::vector<s_t>& arg, const std::vector<s_t>& res) const override;

    /// Can the operation be performed inplace (i.e. overwrite the result)
    s_t n_inplace() const override { return 1;}

    /// Get a unary operation
    MX get_unary(e_t op) const override;

    /// Get a binary operation operation
    MX _get_binary(e_t op, const MX& y, bool scX, bool scY) const override;

    /** \brief Check if two nodes are equivalent up to a given depth */
    bool is_equal(const MXNode* node, s_t depth) const override {
      return sameOpAndDeps(node, depth);
    }

    //! \brief operation
    Operation op_;
  };

} // namespace casadi

/// \endcond

#endif // CASADI_UNARY_MX_HPP
