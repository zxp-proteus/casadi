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


#ifndef CASADI_GUROBI_INTERFACE_HPP
#define CASADI_GUROBI_INTERFACE_HPP

#include "casadi/core/function/qp_solver_internal.hpp"
#include <casadi/interfaces/gurobi/casadi_qpsolver_gurobi_export.h>

// GUROBI header
extern "C" {
#include "gurobi_c.h" // NOLINT(build/include)
}

/** \defgroup plugin_QpSolver_gurobi
    Interface to the GUROBI Solver for quadratic programming
*/

/** \pluginsection{QpSolver,gurobi} */

/// \cond INTERNAL
namespace casadi {

  /** \brief \pluginbrief{QpSolver,gurobi}

      @copydoc QpSolver_doc
      @copydoc plugin_QpSolver_gurobi

  */
  class CASADI_QPSOLVER_GUROBI_EXPORT GurobiInterface : public QpSolverInternal {
  public:

    /** \brief  Constructor */
    explicit GurobiInterface();

    /** \brief  Create a new Solver */
    explicit GurobiInterface(const std::map<std::string, Sparsity>& st);

    /** \brief  Create a new QP Solver */
    static QpSolverInternal* creator(const std::map<std::string, Sparsity>& st) {
      return new GurobiInterface(st);
    }

    /** \brief  Clone */
    virtual GurobiInterface* clone() const;

    /** \brief  Destructor */
    virtual ~GurobiInterface();

    /** \brief  Initialize */
    virtual void init();

    virtual void evaluate();

    /// A documentation string
    static const std::string meta_doc;

    // Variable types
    std::vector<char> vtype_;

  protected:
    // Gurobi environment
    GRBenv *env_;

    std::vector<double> val_;
    std::vector<int> ind_;
    std::vector<int> ind2_;
    std::vector<int> tr_ind_;

  };

} // namespace casadi

/// \endcond
#endif // CASADI_GUROBI_INTERFACE_HPP
