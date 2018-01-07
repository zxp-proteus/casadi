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


#ifndef CASADI_RESHAPE_HPP
#define CASADI_RESHAPE_HPP

#include "mx_node.hpp"
#include <map>
#include <stack>

/// \cond INTERNAL

namespace casadi {
  /** \brief Reshape an expression
      \author Joel Andersson
      \date 2013
  */
  class CASADI_EXPORT Reshape : public MXNode {
  public:

    /// Constructor
    Reshape(const MX& x, Sparsity sp);

    /// Destructor
    ~Reshape() override {}

    /// Evaluate the function (template)
    template<typename T>
    r_t eval_gen(const T** arg, T** res, s_t* iw, T* w) const;

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

    /** \brief  Print expression */
    std::string disp(const std::vector<std::string>& arg) const override;

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g,
                          const std::vector<s_t>& arg, const std::vector<s_t>& res) const override;

    /** \brief Get the operation */
    e_t op() const override { return OP_RESHAPE;}

    /// Can the operation be performed inplace (i.e. overwrite the result)
    s_t n_inplace() const override { return 1;}

    /// Reshape
    MX get_reshape(const Sparsity& sp) const override;

    /** \brief Check if two nodes are equivalent up to a given depth */
    bool is_equal(const MXNode* node, s_t depth) const override
    { return sameOpAndDeps(node, depth) && sparsity()==node->sparsity();}

    /// Transpose (if a dimension is one)
    MX get_transpose() const override;

    /** \brief  Check if valid function input */
    bool is_valid_input() const override;

    /** \brief Get the number of symbolic primitives */
    s_t n_primitives() const override;

    /** \brief Get symbolic primitives */
    void primitives(std::vector<MX>::iterator& it) const override;

    /** \brief Split up an expression along symbolic primitives */
    void split_primitives(const MX& x, std::vector<MX>::iterator& it) const override;

    /** \brief Join an expression along symbolic primitives */
    MX join_primitives(std::vector<MX>::const_iterator& it) const override;

    /** \brief Detect duplicate symbolic expressions */
    bool has_duplicates() const override;

    /** \brief Reset the marker for an input expression */
    void reset_input() const override;
  };

} // namespace casadi
/// \endcond

#endif // CASADI_RESHAPE_HPP
