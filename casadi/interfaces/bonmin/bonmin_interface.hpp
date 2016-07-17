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


#ifndef CASADI_BONMIN_INTERFACE_HPP
#define CASADI_BONMIN_INTERFACE_HPP

#include <IpIpoptApplication.hpp>
#include "BonOsiTMINLPInterface.hpp"
#include "BonIpoptSolver.hpp"
#include "BonCbc.hpp"
#include "BonBonminSetup.hpp"
#include "BonOACutGenerator2.hpp"
#include "BonEcpCuts.hpp"
#include "BonOaNlpOptim.hpp"

#include <casadi/interfaces/bonmin/casadi_nlpsolver_bonmin_export.h>
#include "casadi/core/function/nlp_solver_internal.hpp"

/** \defgroup plugin_NlpSolver_bonmin
*
* Interface to BONMIN solver
*
* On demand by Deltares, July 2016
*
*/

/** \pluginsection{NlpSolver,bonmin} **/

  typedef struct {
    double proc;
    double wall;
  } timer;

  typedef struct {
    double proc;
    double wall;
  } diffTime;

/// \cond INTERNAL
namespace casadi {

/** \brief \pluginbrief{NlpSolver,bonmin}

@copydoc NlpSolver_doc
@copydoc plugin_NlpSolver_bonmin
*/
class CASADI_NLPSOLVER_BONMIN_EXPORT BonminInterface : public NlpSolverInternal {
friend class BonminUserClass;

public:
  explicit BonminInterface(const Function& nlp);
  virtual ~BonminInterface();
  virtual BonminInterface* clone() const { return new BonminInterface(*this);}

  /** \brief  Create a new NLP Solver */
  static NlpSolverInternal* creator(const Function& nlp)
  { return new BonminInterface(nlp);}

  virtual void init();
  virtual void evaluate();

  /// Set default options for a given recipe
  virtual void setDefaultOptions(const std::vector<std::string>& recipes);

  // Get reduced Hessian
  virtual DMatrix getReducedHessian();

  /// Exact Hessian?
  bool exact_hessian_;

  /// All BONMIN options
  std::map<std::string, TypeID> ops_;

  // Bonmin callback functions
  bool eval_f(int n, const double* x, bool new_x, double& obj_value);
  bool eval_grad_f(int n, const double* x, bool new_x, double* grad_f);
  bool eval_g(int n, const double* x, bool new_x, int m, double* g);
  bool eval_jac_g(int n, const double* x, bool new_x, int m, int nele_jac, int* iRow, int *jCol,
                  double* values);
  bool eval_h(const double* x, bool new_x, double obj_factor, const double* lambda,
              bool new_lambda, int nele_hess, int* iRow, int* jCol, double* values);
  void finalize_solution(Bonmin::TMINLP::SolverReturn status, const double* x, double obj_value);
  bool get_bounds_info(int n, double* x_l, double* x_u, int m, double* g_l, double* g_u);
  bool get_starting_point(int n, bool init_x, double* x, bool init_z, double* z_L, double* z_U,
                          int m, bool init_lambda, double* lambda);
  void get_nlp_info(int& n, int& m, int& nnz_jac_g, int& nnz_h_lag);
  int get_number_of_nonlinear_variables();
  bool get_list_of_nonlinear_variables(int num_nonlin_vars, int* pos_nonlin_vars);
  bool intermediate_callback(const double* x, const double* z_L,
                             const double* z_U, const double* g,
                             const double* lambda, double obj_value, int iter,
                             double inf_pr, double inf_du, double mu, double d_norm,
                             double regularization_size, double alpha_du, double alpha_pr,
                             int ls_trials, bool full_callback);

  static timer getTimerTime(void);
  static diffTime diffTimers(const timer t1, const timer t0);
  static void timerPlusEq(diffTime & t, const diffTime diff);
  static void timingSummary(
    std::vector<std::tuple<std::string, int, diffTime> >& xs);

  // Accumulated time since last reset:
  diffTime t_eval_f_; // time spent in eval_f
  diffTime t_eval_grad_f_; // time spent in eval_grad_f
  diffTime t_eval_g_; // time spent in eval_g
  diffTime t_eval_jac_g_; // time spent in eval_jac_g
  diffTime t_eval_h_; // time spent in eval_h
  diffTime t_callback_fun_;  // time spent in callback function
  diffTime t_callback_prepare_; // time spent in callback preparation
  diffTime t_mainloop_; // time spent in the main loop of the solver

  // Accumulated counts since last reset:
  int n_eval_f_; // number of calls to eval_f
  int n_eval_grad_f_; // number of calls to eval_grad_f
  int n_eval_g_; // number of calls to eval_g
  int n_eval_jac_g_; // number of calls to eval_jac_g
  int n_eval_h_; // number of calls to eval_h
  int n_eval_callback_; // number of calls to callback
  int n_iter_; // number of iterations

  /// A documentation string
  static const std::string meta_doc;

  std::vector<bool> discrete_;

  // Options
  bool pass_nonlinear_variables_;
  std::vector<bool> nl_ex_;
  Dict var_string_md_, var_integer_md_, var_numeric_md_,
    con_string_md_, con_integer_md_, con_numeric_md_;
};

} // namespace casadi
/// \endcond

#endif // CASADI_BONMIN_INTERFACE_HPP
