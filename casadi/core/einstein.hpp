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


#ifndef CASADI_EINSTEIN_HPP
#define CASADI_EINSTEIN_HPP

#include "mx_node.hpp"

/// \cond INTERNAL

namespace casadi {
  /** \brief An MX atomic for an Einstein product,
      \author Joris Gillis
      \date 2016
  */
  class CASADI_EXPORT Einstein : public MXNode {
  public:

    /** \brief  Constructor */
    Einstein(const MX& C, const MX& A, const MX& B,
      const std::vector<int>& dim_c, const std::vector<int>& dim_a, const std::vector<int>& dim_b,
      const std::vector<int>& c, const std::vector<int>& a, const std::vector<int>& b);

    /** \brief  Destructor */
    ~Einstein() override {}

    /** \brief  Print expression */
    std::string disp(const std::vector<std::string>& arg) const override;

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g,
                          const std::vector<int>& arg, const std::vector<int>& res) const override;

    /// Evaluate the function (template)
    template<typename T>
    r_t eval_gen(const T** arg, T** res, int* iw, T* w) const;

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

    /** \brief Get the operation */
    e_t op() const override { return OP_EINSTEIN;}

    /// Can the operation be performed inplace (i.e. overwrite the result)
    int n_inplace() const override { return 1;}

    /** \brief Check if two nodes are equivalent up to a given depth */
    bool is_equal(const MXNode* node, int depth) const override {
      return sameOpAndDeps(node, depth) && dynamic_cast<const Einstein*>(node)!=0;
    }

    /** \brief Get required length of w field */
    size_t sz_w() const override { return sparsity().size1();}

    /** Obtain information about node */
    Dict info() const override {
      return {{"dim_a", dim_a_}, {"dim_b", dim_b_}, {"dim_c", dim_c_},
              {"a", a_}, {"b", b_}, {"c", c_},
              {"iter_dims", iter_dims_},
              {"strides_a", strides_a_}, {"strides_b", strides_b_}, {"strides_c", strides_c_},
              {"n_iter", n_iter_}};
    }

    /// Dimensions of tensors A B C
    std::vector<int> dim_c_, dim_a_, dim_b_;
    /// Einstein indices
    std::vector<int> c_, a_, b_;

    std::vector<int> iter_dims_;

    std::vector<int> strides_a_;
    std::vector<int> strides_b_;
    std::vector<int> strides_c_;

    int n_iter_;

  };


} // namespace casadi
/// \endcond

#endif // CASADI_EINSTEIN_HPP
