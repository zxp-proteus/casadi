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


#include "gurobi_interface.hpp"
#include "casadi/core/std_vector_tools.hpp"

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_QPSOLVER_GUROBI_EXPORT
  casadi_register_qpsolver_gurobi(QpSolverInternal::Plugin* plugin) {
    plugin->creator = GurobiInterface::creator;
    plugin->name = "gurobi";
    plugin->doc = GurobiInterface::meta_doc.c_str();
    plugin->version = 23;
    return 0;
  }

  extern "C"
  void CASADI_QPSOLVER_GUROBI_EXPORT casadi_load_qpsolver_gurobi() {
    QpSolverInternal::registerPlugin(casadi_register_qpsolver_gurobi);
  }

  GurobiInterface* GurobiInterface::clone() const {
    // Return a deep copy
    GurobiInterface* node =
      new GurobiInterface(make_map("h", st_[QP_STRUCT_H], "a", st_[QP_STRUCT_A]));
    if (!node->is_init_)
      node->init();
    return node;
  }

  GurobiInterface::~GurobiInterface() {
    if (env_) GRBfreeenv(env_);
  }

  GurobiInterface::GurobiInterface(const std::map<std::string, Sparsity>& st)
    : QpSolverInternal(st) {
    addOption("vtype",                   OT_STRINGVECTOR,     GenericType(),
              "Type of variables: [CONTINUOUS|binary|integer|semicont|semiint]");
    env_ = 0;
  }

  void GurobiInterface::init() {
    QpSolverInternal::init();

    // Read options
    if (hasSetOption("vtype")) {
      std::vector<std::string> vtype = getOption("vtype");
      userOut() << vtype << ":" << n_ << std::endl;
      casadi_assert_message(vtype.size()==n_, "Option 'vtype' has wrong length");
      vtype_.resize(n_);
      for (int i=0; i<n_; ++i) {
        if (vtype[i]=="continuous") {
          vtype_[i] = GRB_CONTINUOUS;
        } else if (vtype[i]=="binary") {
          vtype_[i] = GRB_BINARY;
        } else if (vtype[i]=="integer") {
          vtype_[i] = GRB_INTEGER;
        } else if (vtype[i]=="semicont") {
          vtype_[i] = GRB_SEMICONT;
        } else if (vtype[i]=="semiint") {
          vtype_[i] = GRB_SEMIINT;
        } else {
          casadi_error("No such variable type: " + vtype[i]);
        }
      }
    } else {
      vtype_.resize(n_, GRB_CONTINUOUS);
    }

    val_.resize(n_);
    ind_.resize(n_);
    ind2_.resize(n_);
    tr_ind_.resize(n_);

    // Load environment
    int flag = GRBloadenv(&env_, 0); // no log file
    casadi_assert_message(!flag && env_, "Failed to create GUROBI environment");
  }

  void GurobiInterface::evaluate() {
    if (inputs_check_) checkInputs();


    // Inputs
    const double *h=getPtr(input(QP_SOLVER_H)),
      *g=getPtr(input(QP_SOLVER_G)),
      *a=getPtr(input(QP_SOLVER_A)),
      *lba=getPtr(input(QP_SOLVER_LBA)),
      *uba=getPtr(input(QP_SOLVER_UBA)),
      *lbx=getPtr(input(QP_SOLVER_LBX)),
      *ubx=getPtr(input(QP_SOLVER_UBX));
      //*x0=getPtr(input(QP_SOLVER_X0)),
      //*lam_x0=getPtr(input(QP_SOLVER_LAM_X0));

    // Outputs
    double *x=getPtr(output(QP_SOLVER_X)),
      *cost=getPtr(output(QP_SOLVER_COST));
  //    *lam_a=getPtr(output(QP_SOLVER_LAM_A));
  //    *lam_x=getPtr(output(QP_SOLVER_LAM_X));

    // Temporary memory
    double *val=getPtr(val_);
    int *ind=getPtr(ind_);
    int *ind2=getPtr(ind2_);
    int *tr_ind=getPtr(tr_ind_);

    // Greate an empty model
    GRBmodel *model = 0;
    try {
      int flag = GRBnewmodel(env_, &model, "mysolver", 0, 0, 0, 0, 0, 0);
      casadi_assert_message(!flag, GRBgeterrormsg(env_));

      // Add variables
      for (int i=0; i<n_; ++i) {
        // Get bounds
        double lb = lbx ? lbx[i] : 0., ub = ubx ? ubx[i] : 0.;
        if (isinf(lb)) lb = -GRB_INFINITY;
        if (isinf(ub)) ub =  GRB_INFINITY;

        // Pass to model
        flag = GRBaddvar(model, 0, 0, 0, g ? g[i] : 0., lb, ub, vtype_.at(i), 0);
        casadi_assert_message(!flag, GRBgeterrormsg(env_));
      }
      flag = GRBupdatemodel(model);
      casadi_assert_message(!flag, GRBgeterrormsg(env_));

      // Add quadratic terms
      const int *H_colind=input(QP_SOLVER_H).sparsity().colind(),
        *H_row=input(QP_SOLVER_H).sparsity().row();
      for (int i=0; i<n_; ++i) {

        // Quadratic term nonzero indices
        int numqnz = H_colind[1]-H_colind[0];
        casadi_copy_n(H_row, numqnz, ind);
        H_colind++;
        H_row += numqnz;

        // Corresponding column
        casadi_fill_n(ind2, numqnz, i);

        // Quadratic term nonzeros
        if (h) {
          casadi_copy_n(h, numqnz, val);
          casadi_scal(numqnz, 0.5, val, 1);
          h += numqnz;
        } else {
          casadi_fill_n(val, numqnz, 0.);
        }

        // Pass to model
        flag = GRBaddqpterms(model, numqnz, ind, ind2, val);
        casadi_assert_message(!flag, GRBgeterrormsg(env_));
      }

      // Add constraints
      const int *A_colind=input(QP_SOLVER_A).sparsity().colind(),
        *A_row=input(QP_SOLVER_A).sparsity().row();
      casadi_copy_n(A_colind, n_, tr_ind);
      for (int i=0; i<nc_; ++i) {
        // Get bounds
        double lb = lba ? lba[i] : 0., ub = uba ? uba[i] : 0.;
//        if (isinf(lb)) lb = -GRB_INFINITY;
//        if (isinf(ub)) ub =  GRB_INFINITY;

        // Constraint nonzeros
        int numnz = 0;
        for (int j=0; j<n_; ++j) {
          if (tr_ind[j]<A_colind[j+1] && A_row[tr_ind[j]]==i) {
            ind[numnz] = j;
            val[numnz] = a ? a[tr_ind[j]] : 0;
            numnz++;
            tr_ind[j]++;
          }
        }

        // Pass to model
        if (isinf(lb)) {
          if (isinf(ub)) {
            // Neither upper or lower bounds, skip
          } else {
            // Only upper bound
            flag = GRBaddconstr(model, numnz, ind, val, GRB_LESS_EQUAL, ub, 0);
            casadi_assert_message(!flag, GRBgeterrormsg(env_));
          }
        } else {
          if (isinf(ub)) {
            // Only lower bound
            flag = GRBaddconstr(model, numnz, ind, val, GRB_GREATER_EQUAL, lb, 0);
            casadi_assert_message(!flag, GRBgeterrormsg(env_));
          } else if (lb==ub) {
            // Upper and lower bounds equal
            flag = GRBaddconstr(model, numnz, ind, val, GRB_EQUAL, lb, 0);
            casadi_assert_message(!flag, GRBgeterrormsg(env_));
          } else {
            // Both upper and lower bounds
            flag = GRBaddrangeconstr(model, numnz, ind, val, lb, ub, 0);
            casadi_assert_message(!flag, GRBgeterrormsg(env_));
          }
        }
      }

      // Solve the optimization problem
      flag = GRBoptimize(model);
      casadi_assert_message(!flag, GRBgeterrormsg(env_));
      int optimstatus;
      flag = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
      casadi_assert_message(!flag, GRBgeterrormsg(env_));

      // Get the objective value, if requested
      if (cost) {
        flag = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, cost);
        if (flag) cost[0] = numeric_limits<double>::quiet_NaN();
      }

      // Get the optimal solution, if requested
      if (x) {
        flag = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, n_, x);
        if (flag) casadi_fill_n(x, n_, numeric_limits<double>::quiet_NaN());
      }

      //if (lam_a) {
      //  flag = GRBgetdblattrarray(model, GRB_DBL_ATTR_PI, 0, nc_, lam_a);
      //  casadi_assert_message(!flag, GRBgeterrormsg(env_));
      //  casadi_scal(nc_, -1.0, lam_a, 1);
      //}

      // Free memory
      GRBfreemodel(model);

    } catch (...) {
      // Free memory
      if (model) GRBfreemodel(model);
      throw;
    }
  }

} // namespace casadi
