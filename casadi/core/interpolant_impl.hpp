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


#ifndef CASADI_INTERPOLANT_IMPL_HPP
#define CASADI_INTERPOLANT_IMPL_HPP

#include "interpolant.hpp"
#include "function_internal.hpp"
#include "plugin_interface.hpp"

/// \cond INTERNAL

namespace casadi {

  /** Internal class
      @copydoc Interpolant_doc
  */
  class CASADI_EXPORT Interpolant
  : public FunctionInternal, public PluginInterface<Interpolant> {
  public:
    /// Constructor
    Interpolant(const std::string& name,
                const std::vector<double>& grid,
                const std::vector<s_t>& offset,
                const std::vector<double>& values,
                s_t m);

    /// Destructor
    ~Interpolant() override;

    ///@{
    /** \brief Number of function inputs and outputs */
    size_t get_n_in() override { return 1;}
    size_t get_n_out() override { return 1;}
    ///@}

    /// @{
    /** \brief Sparsities of function inputs and outputs */
    Sparsity get_sparsity_in(s_t i) override;
    Sparsity get_sparsity_out(s_t i) override;
    /// @}

    ///@{
    /** \brief Names of function input and outputs */
    std::string get_name_in(s_t i) override;
    std::string get_name_out(s_t i) override;
    /// @}

    // Creator function for internal class
    typedef Interpolant* (*Creator)(const std::string& name,
                                    const std::vector<double>& grid,
                                    const std::vector<s_t>& offset,
                                    const std::vector<double>& values,
                                    s_t m);

    // No static functions exposed
    struct Exposed{ };

    /// Collection of solvers
    static std::map<std::string, Plugin> solvers_;

    /// Infix
    static const std::string infix_;

    // Number of dimensions
    s_t ndim_;

    // Number of outputs
    s_t m_;

    // Input grid
    std::vector<double> grid_;

    // Offset for each dimension
    std::vector<s_t> offset_;

    // Values at gridpoints
    std::vector<double> values_;
  };

} // namespace casadi
/// \endcond

#endif // CASADI_INTERPOLANT_IMPL_HPP
