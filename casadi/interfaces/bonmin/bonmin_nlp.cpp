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


#include "bonmin_nlp.hpp"
#include "bonmin_interface.hpp"
#include <ctime>

namespace casadi {


BonminUserClass::BonminUserClass(BonminInterface* solver) {
  this->solver = solver;
  n_ = solver->nx_;
  m_ = solver->ng_;
}

BonminUserClass::~BonminUserClass() {

}

// returns the size of the problem
bool BonminUserClass::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, TNLP::IndexStyleEnum& index_style) {
  solver->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag);

  // use the C style indexing (0-based)
  index_style = TNLP::C_STYLE;

  return true;
}

// returns the variable bounds
bool BonminUserClass::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                Index m, Number* g_l, Number* g_u) {
  return solver->get_bounds_info(n, x_l, x_u, m, g_l, g_u);
}

// returns the initial point for the problem
bool BonminUserClass::get_starting_point(Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Index m, bool init_lambda,
                                   Number* lambda) {
  return solver->get_starting_point(n, init_x, x, init_z, z_L, z_U, m, init_lambda, lambda);
}

// returns the value of the objective function
bool BonminUserClass::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
  return solver->eval_f(n, x, new_x, obj_value);
}

// return the gradient of the objective function grad_ {x} f(x)
bool BonminUserClass::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
  return solver->eval_grad_f(n, x, new_x, grad_f);
}

// return the value of the constraints: g(x)
bool BonminUserClass::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
  return solver->eval_g(n, x, new_x, m, g);
}

// return the structure or values of the jacobian
bool BonminUserClass::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values) {
  return solver->eval_jac_g(n, x, new_x, m, nele_jac, iRow, jCol, values);
}


bool BonminUserClass::eval_h(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values) {
  return solver->eval_h(x, new_x, obj_factor, lambda, new_lambda, nele_hess, iRow, jCol, values);
}


void BonminUserClass::finalize_solution(TMINLP::SolverReturn status,
        Ipopt::Index n, const Ipopt::Number* x, Ipopt::Number obj_value) {
    solver->finalize_solution(status, x, obj_value);
}

bool BonminUserClass::intermediate_callback(AlgorithmMode mode, Index iter, Number obj_value,
                                             Number inf_pr, Number inf_du,
                                             Number mu, Number d_norm,
                                             Number regularization_size,
                                             Number alpha_du, Number alpha_pr,
                                             Index ls_trials,
                                             const IpoptData* ip_data,
                                             IpoptCalculatedQuantities* ip_cq) {

    // Only do the callback every few iterations
    if (iter % solver->callback_step_!=0) return true;

    /// Code copied from TNLPAdapter::FinalizeSolution
    /// See also: http://list.coin-or.org/pipermail/ipopt/2010-July/002078.html
    // http://list.coin-or.org/pipermail/ipopt/2010-April/001965.html

    bool full_callback = false;

    return solver->intermediate_callback(x_, z_L_, z_U_, g_, lambda_, obj_value, iter,
                                         inf_pr, inf_du, mu, d_norm, regularization_size,
                                         alpha_du, alpha_pr, ls_trials, full_callback);
}


Index BonminUserClass::get_number_of_nonlinear_variables() {
  return solver->get_number_of_nonlinear_variables();
}

bool BonminUserClass::get_list_of_nonlinear_variables(Index num_nonlin_vars,
                                                     Index* pos_nonlin_vars) {
  return solver->get_list_of_nonlinear_variables(num_nonlin_vars, pos_nonlin_vars);
}

bool BonminUserClass::get_variables_types(Index n, VariableType* var_types) {
   if (solver->discrete_.empty()) {
     std::fill_n(var_types, n, CONTINUOUS);
   } else {
     if (solver->discrete_.size()!=n) return false;
     for (auto&& d : solver->discrete_) {
       *var_types++ = d ? INTEGER : CONTINUOUS;
     }
   }
   return true;
}

bool BonminUserClass::get_variables_linearity(Index n, Ipopt::TNLP::LinearityType* var_types) {
   for (int i=0; i<n; ++i) var_types[i] = Ipopt::TNLP::NON_LINEAR;
   // Ipopt::TNLP::NON_LINEAR / Ipopt::TNLP::LINEAR
   return true;
}

bool BonminUserClass::get_constraints_linearity(Index m, Ipopt::TNLP::LinearityType* const_types) {
   for (int i=0; i<m; ++i) const_types[i] = Ipopt::TNLP::NON_LINEAR;
   // Ipopt::TNLP::NON_LINEAR / Ipopt::TNLP::LINEAR
   return true;
}

} // namespace casadi
