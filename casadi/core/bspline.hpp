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


#ifndef CASADI_BSPLINE_HPP
#define CASADI_BSPLINE_HPP

#include "function_internal.hpp"

namespace casadi {

  /** Base class for BSpline evaluators
  *
  *
  *
  */
  class BSplineCommon : public FunctionInternal {
  public:
    BSplineCommon(const std::string &name, const std::vector<double>& knots,
      const std::vector<s_t>& offset, const std::vector<s_t>& degree, s_t m);

    /** \brief  Initialize */
    void init(const Dict& opts) override;

    ///@{
    /** \brief Number of function inputs and outputs */
    size_t get_n_in() override { return 1; }
    size_t get_n_out() override { return 1; }
    ///@}

    ///@{
    /** \brief Options */
    static Options options_;
    const Options& get_options() const override { return options_;}
    ///@}

    static void from_knots(const std::vector< std::vector<double> >& knots,
      std::vector<s_t>& offset, std::vector<double>& stacked);

    std::vector<s_t> lookup_mode_;
    std::vector<double> knots_;
    std::vector<s_t> offset_;
    std::vector<s_t> degree_;
    std::vector<s_t> strides_;

    std::vector<s_t> coeffs_dims_;
    s_t coeffs_size_;
    s_t m_;
  };

  class BSpline : public BSplineCommon {
  public:
    static Function create(const std::string &name,
      const std::vector< std::vector<double> >& knots,
      const std::vector<double>& coeffs, const std::vector<s_t>& degree, s_t m=1,
      const Dict& opts=Dict());

    BSpline(const std::string &name, const std::vector<double>& knots,
      const std::vector<s_t>& offset, const std::vector<double>& coeffs,
      const std::vector<s_t>& degree, s_t m);

    /** \brief  Destructor */
    ~BSpline() override {}

    /// @{
    /** \brief Sparsities of function inputs and outputs */
    Sparsity get_sparsity_in(s_t i) override { return Sparsity::dense(offset_.size()-1); }
    Sparsity get_sparsity_out(s_t i) override { return Sparsity::dense(m_, 1); }
    /// @}

    /** \brief  Initialize */
    void init(const Dict& opts) override;

    /** \brief  Evaluate numerically, work vectors given */
    r_t eval(const double** arg, double** res, s_t* iw, double* w, void* mem) const override;

    ///@{
    /** \brief Generate a function that calculates \a nfwd forward derivatives */
    bool has_forward(s_t nfwd) const override { return true;}
    Function get_forward(s_t nfwd, const std::string& name,
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames,
                         const Dict& opts) const override;
    ///@}

    ///@{
    /** \brief Generate a function that calculates \a nadj adjoint derivatives */
    bool has_reverse(s_t nadj) const override { return true;}
    Function get_reverse(s_t nadj, const std::string& name,
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames,
                         const Dict& opts) const override;
    ///@}

    ///@{
    /** \brief Return Jacobian of all input elements with respect to all output elements */
    bool has_jacobian() const override { return true;}
    Function get_jacobian(const std::string& name,
                          const std::vector<std::string>& inames,
                          const std::vector<std::string>& onames,
                          const Dict& opts) const override;
    ///@}

    /** \brief Is codegen supported? */
    bool has_codegen() const override { return true;}

    /** \brief Generate code for the body of the C function */
    void codegen_body(CodeGenerator& g) const override;
    void codegen_declarations(CodeGenerator& g) const override {};

    std::string class_name() const override { return "BSpline"; }

    std::vector<double> coeffs_;

  private:
    std::vector<double> derivative_coeff(s_t i) const;
    MX jac(const MX& x) const;
  };


  class BSplineDual : public BSplineCommon {
  public:
    static Function create(const std::string &name,
      const std::vector< std::vector<double> >& knots,
      const std::vector<double>& x, const std::vector<s_t>& degree, s_t m=1, bool reverse=false,
      const Dict& opts=Dict());

    BSplineDual(const std::string &name, const std::vector<double>& knots,
      const std::vector<s_t>& offset, const std::vector<double>& x,
      const std::vector<s_t>& degree, s_t m, bool reverse);

    /** \brief  Destructor */
    ~BSplineDual() override {}

    ///@{
    /** \brief Number of function inputs and outputs */
    size_t get_n_in() override { return 1; }
    size_t get_n_out() override { return 1; }
    ///@}

    /// @{
    /** \brief Sparsities of function inputs and outputs */
    Sparsity get_sparsity_in(s_t i) override;
    Sparsity get_sparsity_out(s_t i) override;
    /// @}

    /** \brief  Initialize */
    void init(const Dict& opts) override;

    /** \brief  Evaluate numerically, work vectors given */
    r_t eval(const double** arg, double** res, s_t* iw, double* w, void* mem) const override;

    ///@{
    /** \brief Generate a function that calculates \a nfwd forward derivatives */
    bool has_forward(s_t nfwd) const override { return true;}
    Function get_forward(s_t nfwd, const std::string& name,
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames,
                         const Dict& opts) const override;
    ///@}

    ///@{
    /** \brief Generate a function that calculates \a nadj adjoint derivatives */
    bool has_reverse(s_t nadj) const override { return true;}
    Function get_reverse(s_t nadj, const std::string& name,
                         const std::vector<std::string>& inames,
                         const std::vector<std::string>& onames,
                         const Dict& opts) const override;
    ///@}


    /** \brief  Propagate sparsity forward */
    r_t sp_forward(const bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w, void* mem) const override;

    /** \brief  Propagate sparsity backwards */
    r_t sp_reverse(bvec_t** arg, bvec_t** res, s_t* iw, bvec_t* w, void* mem) const override;

    ///@{
    /// Is the class able to propagate seeds through the algorithm?
    bool has_spfwd() const override { return true;}
    bool has_sprev() const override { return true;}
    ///@}

    /** \brief Is codegen supported? */
    bool has_codegen() const override { return true;}

    /** \brief Generate code for the body of the C function */
    void codegen_body(CodeGenerator& g) const override;
    void codegen_declarations(CodeGenerator& g) const override {};

    std::string class_name() const override { return "BSplineDual"; }

    std::vector<double> x_;
    bool reverse_;
    s_t N_;
  };

} // namespace casadi

#endif // CASADI_BSPLINE_HPP
