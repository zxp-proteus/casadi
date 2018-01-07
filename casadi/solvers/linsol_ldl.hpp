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


#ifndef CASADI_LINSOL_LDL_HPP
#define CASADI_LINSOL_LDL_HPP

/** \defgroup plugin_Linsol_ldl
  * Linear solver using sparse direct LDL factorization
*/

/** \pluginsection{Linsol,ldl} */

/// \cond INTERNAL
#include "casadi/core/linsol_internal.hpp"
#include <casadi/solvers/casadi_linsol_ldl_export.h>

namespace casadi {
  struct CASADI_LINSOL_LDL_EXPORT LinsolLdlMemory : public LinsolMemory {
    std::vector<s_t> iw;
    std::vector<double> l, d, w;
  };

  /** \brief \pluginbrief{LinsolInternal,ldl}
   * @copydoc LinsolInternal_doc
   * @copydoc plugin_LinsolInternal_ldl
   */
  class CASADI_LINSOL_LDL_EXPORT LinsolLdl : public LinsolInternal {
  public:

    // Create a linear solver given a sparsity pattern and a number of right hand sides
    LinsolLdl(const std::string& name, const Sparsity& sp);

    /** \brief  Create a new LinsolInternal */
    static LinsolInternal* creator(const std::string& name, const Sparsity& sp) {
      return new LinsolLdl(name, sp);
    }

    // Destructor
    ~LinsolLdl() override;

    // Initialize the solver
    void init(const Dict& opts) override;

    /** \brief Create memory block */
    void* alloc_mem() const override { return new LinsolLdlMemory();}

    /** \brief Initalize memory block */
    r_t init_mem(void* mem) const override;

    /** \brief Free memory block */
    void free_mem(void *mem) const override { delete static_cast<LinsolLdlMemory*>(mem);}

    // Symbolic factorization
    r_t sfact(void* mem, const double* A) const override;

    // Factorize the linear system
    r_t nfact(void* mem, const double* A) const override;

    // Solve the linear system
    r_t solve(void* mem, const double* A, double* x, s_t nrhs, bool tr) const override;

    /// Number of negative eigenvalues
    s_t neig(void* mem, const double* A) const override;

    /// Matrix rank
    s_t rank(void* mem, const double* A) const override;

    /// A documentation string
    static const std::string meta_doc;

    // Get name of the plugin
    const char* plugin_name() const override { return "ldl";}

    // Get name of the class
    std::string class_name() const override { return "LinsolLdl";}

    // Symbolic factorization
    std::vector<s_t> parent_;
    Sparsity sp_L_;
  };

} // namespace casadi

/// \endcond

#endif // CASADI_LINSOL_LDL_HPP
