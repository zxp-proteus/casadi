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


#ifndef CASADI_X_FUNCTION_HPP
#define CASADI_X_FUNCTION_HPP

#include <stack>
#include "function_internal.hpp"
#include "factory.hpp"

// To reuse variables we need to be able to sort by sparsity pattern
#include <unordered_map>
#define SPARSITY_MAP std::unordered_map

/// \cond INTERNAL

namespace casadi {

  /** \brief  Internal node class for the base class of SXFunction and MXFunction
      (lacks a public counterpart)
      The design of the class uses the curiously recurring template pattern (CRTP) idiom
      \author Joel Andersson
      \date 2011
  */
  template<typename DerivedType, typename MatType, typename NodeType>
  class CASADI_EXPORT XFunction : public FunctionInternal {
  public:

    /** \brief  Constructor  */
    XFunction(const std::string& name,
              const std::vector<MatType>& ex_in,
              const std::vector<MatType>& ex_out,
              const std::vector<std::string>& name_in,
              const std::vector<std::string>& name_out);

    /** \brief  Destructor */
    ~XFunction() override {}

    /** \brief  Initialize */
    void init(const Dict& opts) override;

    ///@{
    /// Is the class able to propagate seeds through the algorithm?
    bool has_spfwd() const override { return true;}
    bool has_sprev() const override { return true;}
    ///@}

    /** \brief  Topological sorting of the nodes based on Depth-First Search (DFS) */
    static void sort_depth_first(std::stack<NodeType*>& s, std::vector<NodeType*>& nodes);

    /** \brief  Construct a complete Jacobian by compression */
    MatType jac(s_t iind, s_t oind, const Dict& opts) const;

    /** \brief Check if the function is of a particular type */
    bool is_a(const std::string& type, bool recursive) const override {
      return type=="xfunction" || (recursive && FunctionInternal::is_a(type, recursive));
    }

    // Factory
    Function factory(const std::string& name,
                             const std::vector<std::string>& s_in,
                             const std::vector<std::string>& s_out,
                             const Function::AuxOut& aux,
                             const Dict& opts) const override;

    /** \brief Which variables enter with some order
    *
    * \param[in] order Only 1 (linear) and 2 (nonlinear) allowed
    * \param[in] tr   Flip the relationship. Return which expressions contain the variables
    */
    std::vector<bool> which_depends(const std::string& s_in,
                                            const std::vector<std::string>& s_out,
                                            s_t order, bool tr=false) const override;

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

    /** \brief Get Jacobian sparsity */
    Sparsity get_jacobian_sparsity() const override;

    /** \brief returns a new function with a selection of inputs/outputs of the original */
    Function slice(const std::string& name, const std::vector<s_t>& order_in,
                   const std::vector<s_t>& order_out, const Dict& opts) const override;

    /** \brief Generate code for the declarations of the C function */
    void codegen_declarations(CodeGenerator& g) const override = 0;

    /** \brief Generate code for the body of the C function */
    void codegen_body(CodeGenerator& g) const override = 0;

    /** \brief Export function in a specific language */
    void export_code(const std::string& lang,
      std::ostream &stream, const Dict& options) const override;

    /** \brief Export function body in a specific language */
    virtual void export_code_body(const std::string& lang,
        std::ostream &stream, const Dict& options) const = 0;

    /** \brief Is codegen supported? */
    bool has_codegen() const override { return true;}

    /** \brief Helper function: Check if a vector equals ex_in */
    virtual bool isInput(const std::vector<MatType>& arg) const;

    /** Inline calls? */
    virtual bool should_inline(bool always_inline, bool never_inline) const = 0;

    /** \brief Create call to (cached) derivative function, forward mode  */
    void call_forward(const std::vector<MatType>& arg,
                              const std::vector<MatType>& res,
                              const std::vector<std::vector<MatType> >& fseed,
                              std::vector<std::vector<MatType> >& fsens,
                              bool always_inline, bool never_inline) const override;

    /** \brief Create call to (cached) derivative function, reverse mode  */
    void call_reverse(const std::vector<MatType>& arg,
                              const std::vector<MatType>& res,
                              const std::vector<std::vector<MatType> >& aseed,
                              std::vector<std::vector<MatType> >& asens,
                              bool always_inline, bool never_inline) const override;

    ///@{
    /** \brief Number of function inputs and outputs */
    size_t get_n_in() override { return in_.size(); }
    size_t get_n_out() override { return out_.size(); }
    ///@}

    /// @{
    /** \brief Sparsities of function inputs and outputs */
    Sparsity get_sparsity_in(s_t i) override { return in_.at(i).sparsity();}
    Sparsity get_sparsity_out(s_t i) override { return out_.at(i).sparsity();}
    /// @}

    // Data members (all public)

    /** \brief  Inputs of the function (needed for symbolic calculations) */
    std::vector<MatType> in_;

    /** \brief  Outputs of the function (needed for symbolic calculations) */
    std::vector<MatType> out_;
  };

  // Template implementations

  template<typename DerivedType, typename MatType, typename NodeType>
  XFunction<DerivedType, MatType, NodeType>::
  XFunction(const std::string& name,
            const std::vector<MatType>& ex_in,
            const std::vector<MatType>& ex_out,
            const std::vector<std::string>& name_in,
            const std::vector<std::string>& name_out)
    : FunctionInternal(name), in_(ex_in),  out_(ex_out) {
    // Names of inputs
    if (!name_in.empty()) {
      casadi_assert(ex_in.size()==name_in.size(),
      "Mismatching number of input names");
      name_in_ = name_in;
    }
    // Names of outputs
    if (!name_out.empty()) {
      casadi_assert(ex_out.size()==name_out.size(),
      "Mismatching number of output names");
      name_out_ = name_out;
    }
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  void XFunction<DerivedType, MatType, NodeType>::init(const Dict& opts) {
    // Call the init function of the base class
    FunctionInternal::init(opts);
    if (verbose_) casadi_message(name_ + "::init");

    // Make sure that inputs are symbolic
    for (s_t i=0; i<n_in_; ++i) {
      if (in_.at(i).nnz()>0 && !in_.at(i).is_valid_input()) {
        casadi_error("Xfunction input arguments must be purely symbolic. \n"
                     "Argument " + str(i) + "(" + name_in_[i] + ") is not symbolic.");
      }
    }

    // Check for duplicate entries among the input expressions
    bool has_duplicates = false;
    for (auto&& i : in_) {
      if (i.has_duplicates()) {
        has_duplicates = true;
        break;
      }
    }

    // Reset temporaries
    for (auto&& i : in_) i.reset_input();

    if (has_duplicates) {
      std::stringstream s;
      s << "The input expressions are not independent:\n";
      for (s_t iind=0; iind<in_.size(); ++iind) {
        s << iind << ": " << in_[iind] << "\n";
      }
      casadi_error(s.str());
    }
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  void XFunction<DerivedType, MatType, NodeType>::sort_depth_first(
      std::stack<NodeType*>& s, std::vector<NodeType*>& nodes) {
    while (!s.empty()) {
      // Get the topmost element
      NodeType* t = s.top();
      // If the last element on the stack has not yet been added
      if (t && t->temp>=0) {
        // Get the index of the next dependency
        s_t next_dep = t->temp++;
        // If there is any dependency which has not yet been added
        if (next_dep < t->n_dep()) {
          // Add dependency to stack
          s.push(static_cast<NodeType*>(t->dep(next_dep).get()));
        } else {
          // if no dependencies need to be added, we can add the node to the algorithm
          nodes.push_back(t);
          // Mark the node as found
          t->temp = -1;
          // Remove from stack
          s.pop();
        }
      } else {
        // If the last element on the stack has already been added
        s.pop();
      }
    }
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  MatType XFunction<DerivedType, MatType, NodeType>
  ::jac(s_t iind, s_t oind, const Dict& opts) const {
    using namespace std;
    if (verbose_) casadi_message(name_ + "::jac");

    // Read options
    bool compact = false;
    bool symmetric = false;
    bool allow_forward = true;
    bool allow_reverse = true;
    for (auto&& op : opts) {
      if (op.first=="compact") {
        compact = op.second;
      } else if (op.first=="symmetric") {
        symmetric = op.second;
      } else if (op.first=="allow_forward") {
        allow_forward = op.second;
      } else if (op.first=="allow_reverse") {
        allow_reverse = op.second;
      } else if (op.first=="verbose") {
        continue;
      } else {
        casadi_error("No such Jacobian option: " + string(op.second));
      }
    }

    // Quick return if trivially empty
    if (nnz_in(iind)==0 || nnz_out(oind)==0) {
      std::pair<s_t, s_t> jac_shape;
      jac_shape.first = compact ? nnz_out(oind) : numel_out(oind);
      jac_shape.second = compact ? nnz_in(iind) : numel_in(iind);
      return MatType(jac_shape);
    }

    if (symmetric) {
      casadi_assert_dev(sparsity_out_.at(oind).is_dense());
    }

    // Create return object
    MatType ret = MatType::zeros(sparsity_jac(iind, oind, compact, symmetric).T());
    if (verbose_) casadi_message("Allocated return value");

    // Quick return if empty
    if (ret.nnz()==0) {
      return ret.T();
    }

    // Get a bidirectional partition
    Sparsity D1, D2;
    get_partition(iind, oind, D1, D2, true, symmetric, allow_forward, allow_reverse);
    if (verbose_) casadi_message("Graph coloring completed");

    // Get the number of forward and adjoint sweeps
    s_t nfdir = D1.is_null() ? 0 : D1.size2();
    s_t nadir = D2.is_null() ? 0 : D2.size2();

    // Number of derivative directions supported by the function
    s_t max_nfdir = max_num_dir_;
    s_t max_nadir = max_num_dir_;

    // Current forward and adjoint direction
    s_t offset_nfdir = 0, offset_nadir = 0;

    // Evaluation result (known)
    std::vector<MatType> res(out_);

    // Forward and adjoint seeds and sensitivities
    std::vector<std::vector<MatType> > fseed, aseed, fsens, asens;

    // Get the sparsity of the Jacobian block
    Sparsity jsp = sparsity_jac(iind, oind, true, symmetric).T();
    const s_t* jsp_colind = jsp.colind();
    const s_t* jsp_row = jsp.row();

    // Input sparsity
    std::vector<s_t> input_col = sparsity_in_.at(iind).get_col();
    const s_t* input_row = sparsity_in_.at(iind).row();

    // Output sparsity
    std::vector<s_t> output_col = sparsity_out_.at(oind).get_col();
    const s_t* output_row = sparsity_out_.at(oind).row();

    // Get transposes and mappings for jacobian sparsity pattern if we are using forward mode
    if (verbose_) casadi_message("jac transposes and mapping");
    std::vector<s_t> mapping;
    Sparsity jsp_trans;
    if (nfdir>0) {
      jsp_trans = jsp.transpose(mapping);
    }

    // The nonzeros of the sensitivity matrix
    std::vector<s_t> nzmap, nzmap2;

    // Additions to the jacobian matrix
    std::vector<s_t> adds, adds2;

    // Temporary vector
    std::vector<s_t> tmp;

    // Progress
    s_t progress = -10;

    // Number of sweeps
    s_t nsweep_fwd = nfdir/max_nfdir;   // Number of sweeps needed for the forward mode
    if (nfdir%max_nfdir>0) nsweep_fwd++;
    s_t nsweep_adj = nadir/max_nadir;   // Number of sweeps needed for the adjoint mode
    if (nadir%max_nadir>0) nsweep_adj++;
    s_t nsweep = std::max(nsweep_fwd, nsweep_adj);
    if (verbose_) {
      casadi_message(str(nsweep) + " sweeps needed for " + str(nfdir) + " forward and "
                     + str(nadir) + " reverse directions");
    }

    // Sparsity of the seeds
    vector<s_t> seed_col, seed_row;

    // Evaluate until everything has been determined
    for (s_t s=0; s<nsweep; ++s) {
      // Print progress
      if (verbose_) {
        s_t progress_new = (s*100)/nsweep;
        // Print when entering a new decade
        if (progress_new / 10 > progress / 10) {
          progress = progress_new;
          casadi_message(str(progress) + " %");
        }
      }

      // Number of forward and adjoint directions in the current "batch"
      s_t nfdir_batch = std::min(nfdir - offset_nfdir, max_nfdir);
      s_t nadir_batch = std::min(nadir - offset_nadir, max_nadir);

      // Forward seeds
      fseed.resize(nfdir_batch);
      for (s_t d=0; d<nfdir_batch; ++d) {
        // Nonzeros of the seed matrix
        seed_col.clear();
        seed_row.clear();

        // For all the directions
        for (s_t el = D1.colind(offset_nfdir+d); el<D1.colind(offset_nfdir+d+1); ++el) {

          // Get the direction
          s_t c = D1.row(el);

          // Give a seed in the direction
          seed_col.push_back(input_col[c]);
          seed_row.push_back(input_row[c]);
        }

        // initialize to zero
        fseed[d].resize(n_in_);
        for (s_t ind=0; ind<fseed[d].size(); ++ind) {
          s_t nrow = size1_in(ind), ncol = size2_in(ind); // Input dimensions
          if (ind==iind) {
            fseed[d][ind] = MatType::ones(Sparsity::triplet(nrow, ncol, seed_row, seed_col));
          } else {
            fseed[d][ind] = MatType(nrow, ncol);
          }
        }
      }

      // Adjoint seeds
      aseed.resize(nadir_batch);
      for (s_t d=0; d<nadir_batch; ++d) {
        // Nonzeros of the seed matrix
        seed_col.clear();
        seed_row.clear();

        // For all the directions
        for (s_t el = D2.colind(offset_nadir+d); el<D2.colind(offset_nadir+d+1); ++el) {

          // Get the direction
          s_t c = D2.row(el);

          // Give a seed in the direction
          seed_col.push_back(output_col[c]);
          seed_row.push_back(output_row[c]);
        }

        //initialize to zero
        aseed[d].resize(n_out_);
        for (s_t ind=0; ind<aseed[d].size(); ++ind) {
          s_t nrow = size1_out(ind), ncol = size2_out(ind); // Output dimensions
          if (ind==oind) {
            aseed[d][ind] = MatType::ones(Sparsity::triplet(nrow, ncol, seed_row, seed_col));
          } else {
            aseed[d][ind] = MatType(nrow, ncol);
          }
        }
      }

      // Forward sensitivities
      fsens.resize(nfdir_batch);
      for (s_t d=0; d<nfdir_batch; ++d) {
        // initialize to zero
        fsens[d].resize(n_out_);
        for (s_t oind=0; oind<fsens[d].size(); ++oind) {
          fsens[d][oind] = MatType::zeros(sparsity_out_.at(oind));
        }
      }

      // Adjoint sensitivities
      asens.resize(nadir_batch);
      for (s_t d=0; d<nadir_batch; ++d) {
        // initialize to zero
        asens[d].resize(n_in_);
        for (s_t ind=0; ind<asens[d].size(); ++ind) {
          asens[d][ind] = MatType::zeros(sparsity_in_.at(ind));
        }
      }

      // Evaluate symbolically
      if (fseed.size()>0) {
        casadi_assert_dev(aseed.size()==0);
        if (verbose_) casadi_message("Calling 'ad_forward'");
        static_cast<const DerivedType*>(this)->ad_forward(fseed, fsens);
        if (verbose_) casadi_message("Back from 'ad_forward'");
      } else if (aseed.size()>0) {
        casadi_assert_dev(fseed.size()==0);
        if (verbose_) casadi_message("Calling 'ad_reverse'");
        static_cast<const DerivedType*>(this)->ad_reverse(aseed, asens);
        if (verbose_) casadi_message("Back from 'ad_reverse'");
      }

      // Carry out the forward sweeps
      for (s_t d=0; d<nfdir_batch; ++d) {
        // Skip if nothing to add
        if (fsens[d][oind].nnz()==0) {
          continue;
        }

        // If symmetric, see how many times each output appears
        if (symmetric) {
          // Initialize to zero
          tmp.resize(nnz_out(oind));
          fill(tmp.begin(), tmp.end(), 0);

          // "Multiply" Jacobian sparsity by seed vector
          for (s_t el = D1.colind(offset_nfdir+d); el<D1.colind(offset_nfdir+d+1); ++el) {

            // Get the input nonzero
            s_t c = D1.row(el);

            // Propagate dependencies
            for (s_t el_jsp=jsp_colind[c]; el_jsp<jsp_colind[c+1]; ++el_jsp) {
              tmp[jsp_row[el_jsp]]++;
            }
          }
        }

        // Locate the nonzeros of the forward sensitivity matrix
        sparsity_out_.at(oind).find(nzmap);
        fsens[d][oind].sparsity().get_nz(nzmap);

        if (symmetric) {
          sparsity_in_.at(iind).find(nzmap2);
          fsens[d][oind].sparsity().get_nz(nzmap2);
        }

        // Assignments to the Jacobian
        adds.resize(fsens[d][oind].nnz());
        fill(adds.begin(), adds.end(), -1);
        if (symmetric) {
          adds2.resize(adds.size());
          fill(adds2.begin(), adds2.end(), -1);
        }

        // For all the input nonzeros treated in the sweep
        for (s_t el = D1.colind(offset_nfdir+d); el<D1.colind(offset_nfdir+d+1); ++el) {

          // Get the input nonzero
          s_t c = D1.row(el);
          //s_t f2_out;
          //if (symmetric) {
          //  f2_out = nzmap2[c];
          //}

          // Loop over the output nonzeros corresponding to this input nonzero
          for (s_t el_out = jsp_trans.colind(c); el_out<jsp_trans.colind(c+1); ++el_out) {

            // Get the output nonzero
            s_t r_out = jsp_trans.row(el_out);

            // Get the forward sensitivity nonzero
            s_t f_out = nzmap[r_out];
            if (f_out<0) continue; // Skip if structurally zero

            // The nonzero of the Jacobian now treated
            s_t elJ = mapping[el_out];

            if (symmetric) {
              if (tmp[r_out]==1) {
                adds[f_out] = el_out;
                adds2[f_out] = elJ;
              }
            } else {
              // Get the output seed
              adds[f_out] = elJ;
            }
          }
        }

        // Get entries in fsens[d][oind] with nonnegative indices
        tmp.resize(adds.size());
        s_t sz = 0;
        for (s_t i=0; i<adds.size(); ++i) {
          if (adds[i]>=0) {
            adds[sz] = adds[i];
            tmp[sz++] = i;
          }
        }
        adds.resize(sz);
        tmp.resize(sz);

        // Add contribution to the Jacobian
        ret.nz(adds) = fsens[d][oind].nz(tmp);

        if (symmetric) {
          // Get entries in fsens[d][oind] with nonnegative indices
          tmp.resize(adds2.size());
          sz = 0;
          for (s_t i=0; i<adds2.size(); ++i) {
            if (adds2[i]>=0) {
              adds2[sz] = adds2[i];
              tmp[sz++] = i;
            }
          }
          adds2.resize(sz);
          tmp.resize(sz);

          // Add contribution to the Jacobian
          ret.nz(adds2) = fsens[d][oind].nz(tmp);
        }
      }

      // Add elements to the Jacobian matrix
      for (s_t d=0; d<nadir_batch; ++d) {
        // Skip if nothing to add
        if (asens[d][iind].nnz()==0) {
          continue;
        }

        // Locate the nonzeros of the adjoint sensitivity matrix
        sparsity_in_.at(iind).find(nzmap);
        asens[d][iind].sparsity().get_nz(nzmap);

        // For all the output nonzeros treated in the sweep
        for (s_t el = D2.colind(offset_nadir+d); el<D2.colind(offset_nadir+d+1); ++el) {

          // Get the output nonzero
          s_t r = D2.row(el);

          // Loop over the input nonzeros that influences this output nonzero
          for (s_t elJ = jsp.colind(r); elJ<jsp.colind(r+1); ++elJ) {

            // Get the input nonzero
            s_t inz = jsp.row(elJ);

            // Get the corresponding adjoint sensitivity nonzero
            s_t anz = nzmap[inz];
            if (anz<0) continue;

            // Get the input seed
            ret.nz(elJ) = asens[d][iind].nz(anz);
          }
        }
      }

      // Update direction offsets
      offset_nfdir += nfdir_batch;
      offset_nadir += nadir_batch;
    }

    // Return
    return ret.T();
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Function XFunction<DerivedType, MatType, NodeType>
  ::get_forward(s_t nfwd, const std::string& name,
                const std::vector<std::string>& inames,
                const std::vector<std::string>& onames,
                const Dict& opts) const {
    // Seeds
    std::vector<std::vector<MatType> > fseed = fwd_seed<MatType>(nfwd), fsens;

    // Evaluate symbolically
    static_cast<const DerivedType*>(this)->ad_forward(fseed, fsens);
    casadi_assert_dev(fsens.size()==fseed.size());

    // All inputs of the return function
    std::vector<MatType> ret_in(inames.size());
    copy(in_.begin(), in_.end(), ret_in.begin());
    for (s_t i=0; i<n_out_; ++i) {
      ret_in.at(n_in_+i) = MatType::sym(inames[n_in_+i], Sparsity(out_.at(i).size()));
    }
    std::vector<MatType> v(nfwd);
    for (s_t i=0; i<n_in_; ++i) {
      for (s_t d=0; d<nfwd; ++d) v[d] = fseed[d][i];
      ret_in.at(n_in_ + n_out_ + i) = horzcat(v);
    }

    // All outputs of the return function
    std::vector<MatType> ret_out(onames.size());
    for (s_t i=0; i<n_out_; ++i) {
      for (s_t d=0; d<nfwd; ++d) v[d] = fsens[d][i];
      ret_out.at(i) = horzcat(v);
    }

    // Assemble function and return
    return Function(name, ret_in, ret_out, inames, onames, opts);
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Function XFunction<DerivedType, MatType, NodeType>
  ::get_reverse(s_t nadj, const std::string& name,
                const std::vector<std::string>& inames,
                const std::vector<std::string>& onames,
                const Dict& opts) const {
    // Seeds
    std::vector<std::vector<MatType> > aseed = symbolicAdjSeed(nadj, out_), asens;

    // Evaluate symbolically
    static_cast<const DerivedType*>(this)->ad_reverse(aseed, asens);

    // All inputs of the return function
    std::vector<MatType> ret_in(inames.size());
    copy(in_.begin(), in_.end(), ret_in.begin());
    for (s_t i=0; i<n_out_; ++i) {
      ret_in.at(n_in_ + i) = MatType::sym(inames[n_in_+i], Sparsity(out_.at(i).size()));
    }
    std::vector<MatType> v(nadj);
    for (s_t i=0; i<n_out_; ++i) {
      for (s_t d=0; d<nadj; ++d) v[d] = aseed[d][i];
      ret_in.at(n_in_ + n_out_ + i)  = horzcat(v);
    }

    // All outputs of the return function
    std::vector<MatType> ret_out(onames.size());
    for (s_t i=0; i<n_in_; ++i) {
      for (s_t d=0; d<nadj; ++d) v[d] = asens[d][i];
      ret_out.at(i) = horzcat(v);
    }

    // Assemble function and return
    return Function(name, ret_in, ret_out, inames, onames, opts);
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Function XFunction<DerivedType, MatType, NodeType>
  ::get_jacobian(const std::string& name,
                 const std::vector<std::string>& inames,
                 const std::vector<std::string>& onames,
                 const Dict& opts) const {
    // Temporary single-input, single-output function FIXME(@jaeandersson)
    Function tmp("tmp", {veccat(in_)}, {veccat(out_)},
                 {{"ad_weight", ad_weight()}, {"ad_weight_sp", sp_weight()}});

    // Jacobian expression
    MatType J = tmp.get<DerivedType>()->jac(0, 0, Dict());

    // All inputs of the return function
    std::vector<MatType> ret_in(inames.size());
    copy(in_.begin(), in_.end(), ret_in.begin());
    for (s_t i=0; i<n_out_; ++i) {
      ret_in.at(n_in_+i) = MatType::sym(inames[n_in_+i], Sparsity(out_.at(i).size()));
    }

    // Assemble function and return
    return Function(name, ret_in, {J}, inames, onames, opts);
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Sparsity XFunction<DerivedType, MatType, NodeType>
  ::get_jacobian_sparsity() const {
    // Temporary single-input, single-output function FIXME(@jaeandersson)
    Function tmp("tmp", {veccat(in_)}, {veccat(out_)},
                 {{"ad_weight", ad_weight()}, {"ad_weight_sp", sp_weight()}});
    return tmp.sparsity_jac(0, 0);
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Function XFunction<DerivedType, MatType, NodeType>
  ::slice(const std::string& name, const std::vector<s_t>& order_in,
          const std::vector<s_t>& order_out, const Dict& opts) const {
    // Return expressions
    std::vector<MatType> ret_in, ret_out;
    std::vector<std::string> ret_in_name, ret_out_name;

    // Reorder inputs
    for (s_t k : order_in) {
      ret_in.push_back(in_.at(k));
      ret_in_name.push_back(name_in_.at(k));
    }

    // Reorder outputs
    for (s_t k : order_out) {
      ret_out.push_back(out_.at(k));
      ret_out_name.push_back(name_out_.at(k));
    }

    // Assembe function
    return Function(name, ret_in, ret_out,
                    ret_in_name, ret_out_name, opts);
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  void XFunction<DerivedType, MatType, NodeType>
  ::export_code(const std::string& lang, std::ostream &ss, const Dict& options) const {

    casadi_assert(lang=="matlab", "Only matlab language supported for now.");

    // start function
    ss << "function [varargout] = " << name_ << "(varargin)" << std::endl;

    // Allocate space for output argument (segments)
    for (s_t i=0;i<n_out_;++i) {
      ss << "  argout_" << i <<  " = cell(" << nnz_out(i) << ",1);" << std::endl;
    }

    Dict opts;
    opts["indent_level"] = 1;
    export_code_body(lang, ss, opts);

    // Process the outputs
    for (s_t i=0;i<n_out_;++i) {
      const Sparsity& out = sparsity_out_.at(i);
      if (out.is_dense()) {
        // Special case if dense
        ss << "  varargout{" << i+1 <<  "} = reshape(vertcat(argout_" << i << "{:}), ";
        ss << out.size1() << ", " << out.size2() << ");" << std::endl;
      } else {
        // For sparse outputs, export sparsity and call 'sparse'
        Dict opts;
        opts["name"] = "sp";
        opts["indent_level"] = 1;
        opts["as_matrix"] = false;
        out.export_code("matlab", ss, opts);
        ss << "  varargout{" << i+1 <<  "} = ";
        ss << "sparse(sp_i, sp_j, vertcat(argout_" << i << "{:}), sp_m, sp_n);" << std::endl;
      }
    }

    // end function
    ss << "end" << std::endl;
    ss << "function y=nonzeros_gen(x)" << std::endl;
    ss << "  if isa(x,'casadi.SX') || isa(x,'casadi.MX') || isa(x,'casadi.DM')" << std::endl;
    ss << "    y = x{:};" << std::endl;
    ss << "  elseif isa(x,'sdpvar')" << std::endl;
    ss << "    b = getbase(x);" << std::endl;
    ss << "    f = find(sum(b~=0,2));" << std::endl;
    ss << "    y = sdpvar(length(f),1,[],getvariables(x),b(f,:));" << std::endl;
    ss << "  else" << std::endl;
    ss << "    y = nonzeros(x);" << std::endl;
    ss << "  end" << std::endl;
    ss << "end" << std::endl;
    ss << "function y=if_else_zero_gen(c,e)" << std::endl;
    ss << "  if isa(c+e,'casadi.SX') || isa(c+e,'casadi.MX') || isa(c+e,'casadi.DM')" << std::endl;
    ss << "    y = if_else(c, e, 0);" << std::endl;
    ss << "  else" << std::endl;
    ss << "    if c" << std::endl;
    ss << "        y = x;" << std::endl;
    ss << "    else" << std::endl;
    ss << "        y = 0;" << std::endl;
    ss << "    end" << std::endl;
    ss << "  end" << std::endl;
    ss << "end" << std::endl;


  }

  template<typename DerivedType, typename MatType, typename NodeType>
  bool XFunction<DerivedType, MatType, NodeType>
  ::isInput(const std::vector<MatType>& arg) const {
    // Check if arguments matches the input expressions, in which case
    // the output is known to be the output expressions
    const s_t checking_depth = 2;
    for (s_t i=0; i<arg.size(); ++i) {
      if (!is_equal(arg[i], in_[i], checking_depth)) {
        return false;
      }
    }
    return true;
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  void XFunction<DerivedType, MatType, NodeType>::
  call_forward(const std::vector<MatType>& arg,
               const std::vector<MatType>& res,
               const std::vector<std::vector<MatType> >& fseed,
               std::vector<std::vector<MatType> >& fsens,
               bool always_inline, bool never_inline) const {
    casadi_assert(!(always_inline && never_inline), "Inconsistent options");
    if (!should_inline(always_inline, never_inline)) {
      // The non-inlining version is implemented in the base class
      return FunctionInternal::call_forward(arg, res, fseed, fsens,
                                            always_inline, never_inline);
    }

    // Quick return if no seeds
    if (fseed.empty()) {
      fsens.clear();
      return;
    }

    // Call inlining
    if (isInput(arg)) {
      // Argument agrees with in_, call ad_forward directly
      static_cast<const DerivedType*>(this)->ad_forward(fseed, fsens);
    } else {
      // Need to create a temporary function
      Function f("tmp", arg, res);
      static_cast<DerivedType *>(f.get())->ad_forward(fseed, fsens);
    }
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  void XFunction<DerivedType, MatType, NodeType>::
  call_reverse(const std::vector<MatType>& arg,
               const std::vector<MatType>& res,
               const std::vector<std::vector<MatType> >& aseed,
               std::vector<std::vector<MatType> >& asens,
               bool always_inline, bool never_inline) const {
    casadi_assert(!(always_inline && never_inline), "Inconsistent options");
    if (!should_inline(always_inline, never_inline)) {
      // The non-inlining version is implemented in the base class
      return FunctionInternal::call_reverse(arg, res, aseed, asens,
                                            always_inline, never_inline);
    }

    // Quick return if no seeds
    if (aseed.empty()) {
      asens.clear();
      return;
    }

    // Call inlining
    if (isInput(arg)) {
      // Argument agrees with in_, call ad_reverse directly
      static_cast<const DerivedType*>(this)->ad_reverse(aseed, asens);
    } else {
      // Need to create a temporary function
      Function f("tmp", arg, res);
      static_cast<DerivedType *>(f.get())->ad_reverse(aseed, asens);
    }
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  Function XFunction<DerivedType, MatType, NodeType>::
  factory(const std::string& name,
          const std::vector<std::string>& s_in,
          const std::vector<std::string>& s_out,
          const Function::AuxOut& aux,
          const Dict& opts) const {
    using namespace std;

    // Create an expression factory
    Factory<MatType> f(aux);
    for (s_t i=0; i<in_.size(); ++i) f.add_input(name_in_[i], in_[i]);
    for (s_t i=0; i<out_.size(); ++i) f.add_output(name_out_[i], out_[i]);

    // Specify input expressions to be calculated
    vector<string> ret_iname;
    for (const string& s : s_in) {
      try {
        ret_iname.push_back(f.request_input(s));
      } catch (CasadiException& ex) {
        casadi_error("Cannot process factory input \"" + s + "\":" + ex.what());
      }
    }

    // Specify output expressions to be calculated
    vector<string> ret_oname;
    for (const string& s : s_out) {
      try {
        ret_oname.push_back(f.request_output(s));
      } catch (CasadiException& ex) {
        casadi_error("Cannot process factory output \"" + s + "\":" + ex.what());
      }
    }

    // Calculate expressions
    f.calculate();

    // Get input expressions
    vector<MatType> ret_in;
    ret_in.reserve(s_in.size());
    for (const string& s : s_in) ret_in.push_back(f.get_input(s));

    // Get output expressions
    vector<MatType> ret_out;
    ret_out.reserve(s_out.size());
    for (const string& s : s_out) ret_out.push_back(f.get_output(s));

    // Create function and return
    Function ret(name, ret_in, ret_out, ret_iname, ret_oname, opts);
    if (ret.has_free()) {
      // Substitute free variables with zeros
      // We assume that the free variables are caused by false positive dependencies
      vector<MatType> free_in = MatType::get_free(ret);
      vector<MatType> free_sub = free_in;
      for (auto&& e : free_sub) e = MatType::zeros(e.sparsity());
      ret_out = substitute(ret_out, free_in, free_sub);
      ret = Function(name, ret_in, ret_out, ret_iname, ret_oname, opts);
    }
    return ret;
  }

  template<typename DerivedType, typename MatType, typename NodeType>
  std::vector<bool> XFunction<DerivedType, MatType, NodeType>::
  which_depends(const std::string& s_in, const std::vector<std::string>& s_out,
      s_t order, bool tr) const {
    using namespace std;

    // Input arguments
    auto it = find(name_in_.begin(), name_in_.end(), s_in);
    casadi_assert_dev(it!=name_in_.end());
    MatType arg = in_.at(it-name_in_.begin());

    // Output arguments
    vector<MatType> res;
    for (auto&& s : s_out) {
      it = find(name_out_.begin(), name_out_.end(), s);
      casadi_assert_dev(it!=name_out_.end());
      res.push_back(out_.at(it-name_out_.begin()));
    }

    // Extract variables entering nonlinearly
    return MatType::which_depends(veccat(res), arg, order, tr);
  }

  template<typename MatType>
  std::vector<bool> _which_depends(const MatType &expr, const MatType &var, s_t order, bool tr) {
    // Short-circuit
    if (expr.is_empty() || var.is_empty()) {
      return std::vector<bool>(tr? expr.numel() : var.numel(), false);
    }

    MatType e = expr;

    // Create a function for calculating a forward-mode derivative
    casadi_assert(order==1 || order==2,
      "which_depends: order argument must be 1 or 2, got " + str(order) + " instead.");

    MatType v = MatType::sym("v", var.sparsity());
    for (s_t i=1;i<order;++i) {
      e = jtimes(e, var, v);
    }

    Function f = Function("tmp", {var}, {e});
    // Propagate sparsities backwards seeding all outputs
    std::vector<bvec_t> seed(tr? f.nnz_in(0) : f.nnz_out(0), 1);
    std::vector<bvec_t> sens(tr? f.nnz_out(0) : f.nnz_in(0), 0);

    if (tr)
      f({get_ptr(seed)}, {get_ptr(sens)});
    else
      f.rev({get_ptr(sens)}, {get_ptr(seed)});
    // Temporaries for evaluation
    std::vector<bool> ret(sens.size());
    std::copy(sens.begin(), sens.end(), ret.begin());

    // Project the result back on the original sparsity
    if (tr && e.sparsity()!=expr.sparsity()) {
      // std::vector<bool> is not accessible as bool*
      // bool -> s_t
      std::vector<s_t> source(sens.size());
      std::copy(ret.begin(), ret.end(), source.begin());
      std::vector<s_t> target(expr.nnz());

      // project
      std::vector<s_t> scratch(expr.size1());
      casadi_project(get_ptr(source), e.sparsity(), get_ptr(target), expr.sparsity(),
        get_ptr(scratch));

      // s_t -> bool
      ret.resize(expr.nnz());
      std::copy(target.begin(), target.end(), ret.begin());
    }

    return ret;
  }

} // namespace casadi
/// \endcond

#endif // CASADI_X_FUNCTION_HPP
