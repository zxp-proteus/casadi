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


#ifndef CASADI_BINARY_SX_HPP
#define CASADI_BINARY_SX_HPP

#include "sx_node.hpp"
#include <stack>


/// \cond INTERNAL
namespace casadi {

/** \brief Represents a basic binary operation on two SXElem nodes
  \author Joel Andersson
  \date 2010
*/
class BinarySX : public SXNode {
  private:

    /** \brief  Constructor is private, use "create" below */
    BinarySX(unsigned char op, const SXElem& dep0, const SXElem& dep1) :
        op_(op), dep0_(dep0), dep1_(dep1) {}

  public:

    /** \brief  Create a binary expression */
    inline static SXElem create(unsigned char op, const SXElem& dep0, const SXElem& dep1) {
      if (dep0.is_constant() && dep1.is_constant()) {
        // Evaluate constant
        double dep0_val(dep0);
        double dep1_val(dep1);
        double ret_val;
        casadi_math<double>::fun(op, dep0_val, dep1_val, ret_val);
        return ret_val;
      } else {
        // Expression containing free variables
        return SXElem::create(new BinarySX(op, dep0, dep1));
      }
    }

    /** \brief Destructor
    This is a rather complex destructor which is necessary since the default destructor
    can cause stack overflow due to recursive calling.
    */
    ~BinarySX() override {
      // Start destruction method if any of the dependencies has dependencies
      for (s_t c1=0; c1<2; ++c1) {
        // Get the node of the dependency and remove it from the smart pointer
        SXNode* n1 = dep(c1).assignNoDelete(casadi_limits<SXElem>::nan);

        // Check if this was the last reference
        if (n1->count==0) {

          // Check if binary
          if (!n1->n_dep()) { // n1 is not binary

            delete n1; // Delete straight away

          } else { // n1 is binary

            // Stack of expressions to be deleted
            std::stack<SXNode*> deletion_stack;

            // Add the node to the deletion stack
            deletion_stack.push(n1);

            // Process stack
            while (!deletion_stack.empty()) {

              // Top element
              SXNode *t = deletion_stack.top();

              // Check if the top element has dependencies with dependencies
              bool added_to_stack = false;
              for (s_t c2=0; c2<t->n_dep(); ++c2) { // for all dependencies of the dependency

                // Get the node of the dependency of the top element
                // and remove it from the smart pointer
                SXNode *n2 = t->dep(c2).assignNoDelete(casadi_limits<SXElem>::nan);

                // Check if this is the only reference to the element
                if (n2->count == 0) {

                  // Check if binary
                  if (!n2->n_dep()) {

                    // Delete straight away if not binary
                    delete n2;

                  } else {

                    // Add to deletion stack
                    deletion_stack.push(n2);
                    added_to_stack = true;
                  }
                }
              }

              // Delete and pop from stack if nothing added to the stack
              if (!added_to_stack) {
                delete deletion_stack.top();
                deletion_stack.pop();
              }
            } // while
          }
        }
      }
    }

    // Class name
    std::string class_name() const override {return "BinarySX";}

    bool is_smooth() const override { return operation_checker<SmoothChecker>(op_);}

    bool is_op(e_t op) const override { return op_==op; }

    /** \brief Check if two nodes are equivalent up to a given depth */
    bool is_equal(const SXNode* node, s_t depth) const override {
      const BinarySX* n = dynamic_cast<const BinarySX*>(node);
      if (n==0) return false;
      if (n->op_ != op_) return false;
      if (SXElem::is_equal(n->dep0_, dep0_, depth-1)
          && SXElem::is_equal(n->dep1_, dep1_, depth-1)) return true;
      if (operation_checker<CommChecker>(op_)
          && SXElem::is_equal(n->dep1_, dep0_, depth-1)
          && SXElem::is_equal(n->dep0_, dep1_, depth-1)) return true;
      return false;
    }

    /** \brief  Number of dependencies */
    s_t n_dep() const override { return 2;}

    /** \brief  get the reference of a dependency */
    const SXElem& dep(s_t i) const override { return i==0 ? dep0_ : dep1_;}
    SXElem& dep(s_t i) override { return i==0 ? dep0_ : dep1_;}

    /** \brief  Get the operation */
    e_t op() const override { return op_;}

    /** \brief  Print expression */
    std::string print(const std::string& arg1, const std::string& arg2) const override {
      return casadi_math<double>::print(op_, arg1, arg2);
    }

    /** \brief  The binary operation as an 1 byte integer (allows 256 values) */
    unsigned char op_;

    /** \brief  The dependencies of the node */
    SXElem dep0_, dep1_;
};

} // namespace casadi
/// \endcond

#endif // CASADI_BINARY_SX_HPP
