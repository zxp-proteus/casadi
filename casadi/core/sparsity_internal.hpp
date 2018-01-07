/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *    Copyright (C) 2006-2009 Timothy A. Davis
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


#ifndef CASADI_SPARSITY_INTERNAL_HPP
#define CASADI_SPARSITY_INTERNAL_HPP

#include "sparsity.hpp"
#include "shared_object_internal.hpp"
/// \cond INTERNAL

namespace casadi {

  class CASADI_EXPORT SparsityInternal : public SharedObjectInternal {
  private:
    /* \brief Sparsity pattern in compressed column storage (CCS) format
       The first two entries are the number of rows (nrow) and columns (ncol).
       The next (ncol+1) entries are the column offsets (colind). This means that
       the number of nonzeros (nnz) is given as sp_[sp_[1]+2].
       The last nnz entries are the rows of the nonzeros (row). See public class
       for more info about the CCS format used in CasADi. */
    std::vector<s_t> sp_;

    /** \brief Structure to hold the block triangular form */
    struct Btf {
      s_t nb;
      std::vector<s_t> rowperm, colperm;
      std::vector<s_t> rowblock, colblock;
      std::vector<s_t> coarse_rowblock, coarse_colblock;
    };

    /* \brief The block-triangular factorization for the sparsity
      Calculated on first call, then cached
    */
    mutable Btf* btf_;

  public:
    /// Construct a sparsity pattern from arrays
    SparsityInternal(s_t nrow, s_t ncol, const s_t* colind, const s_t* row);

    /// Destructor
    ~SparsityInternal() override;

    /** \brief Get number of rows (see public class) */
    inline const std::vector<s_t>& sp() const { return sp_;}

    /** \brief Get number of rows (see public class) */
    inline s_t size1() const { return sp_[0];}

    /** \brief Get number of columns (see public class) */
    inline s_t size2() const { return sp_[1];}

    /** \brief Get column offsets (see public class) */
    inline const s_t* colind() const { return &sp_.front()+2;}

    /** \brief Get row indices (see public class) */
    inline const s_t* row() const { return colind()+size2()+1;}

    /// Number of structural non-zeros
    inline s_t nnz() const { return colind()[size2()];}

    /// Check if the dimensions and colind, row vectors are compatible
    void sanity_check(bool complete=false) const;

    /** \brief Get the diagonal of the matrix/create a diagonal matrix
     *
     * \param[out] mapping will contain the nonzero mapping
     */
    Sparsity get_diag(std::vector<s_t>& mapping) const;

    /// has diagonal entries?
    bool has_diag() const;

    /// Drop diagonal entries
    Sparsity drop_diag() const;

    /// Find strongly connected components: See cs_dfs in CSparse
    s_t dfs(s_t j, s_t top, std::vector<s_t>& xi, std::vector<s_t>& pstack,
                         const std::vector<s_t>& pinv, std::vector<bool>& marked) const;

    /// Find the strongly connected components of a square matrix: See cs_scc in CSparse
    s_t scc(std::vector<s_t>& p, std::vector<s_t>& r) const;

    /** \brief Approximate minimal degree preordering
      * The implementation is a modified version of cs_amd in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    std::vector<s_t> amd() const;

    /** \brief Calculate the elimination tree for a matrix
      * len[w] >= ata ? ncol + nrow : ncol
      * len[parent] == ncol
      * Ref: Chapter 4, Direct Methods for Sparse Linear Systems by Tim Davis
      * Modified version of cs_etree in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static void etree(const s_t* sp, s_t* parent, s_t *w, s_t ata);

    /** \brief Traverse an elimination tree using depth first search
      * Ref: Chapter 4, Direct Methods for Sparse Linear Systems by Tim Davis
      * Modified version of cs_tdfs in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static s_t postorder_dfs(s_t j, s_t k, s_t* head, s_t* next,
                             s_t* post, s_t* stack);

    /** \brief Calculate the postorder permuation
      * Ref: Chapter 4, Direct Methods for Sparse Linear Systems by Tim Davis
      * len[w] >= 3*n
      * len[post] == n
      * Modified version of cs_post in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static void postorder(const s_t* parent, s_t n, s_t* post, s_t* w);

    /** \brief Needed by casadi_qr_colind
      * Ref: Chapter 4, Direct Methods for Sparse Linear Systems by Tim Davis
      * Modified version of cs_leaf in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static s_t leaf(s_t i, s_t j, const s_t* first, s_t* maxfirst,
                    s_t* prevleaf, s_t* ancestor, s_t* jleaf);

    /** \brief Calculate the column offsets for the QR R matrix
      * Ref: Chapter 4, Direct Methods for Sparse Linear Systems by Tim Davis
      * len[counts] = ncol
      * len[w] >= 5*ncol + nrow + 1
      * Modified version of cs_counts in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static s_t qr_counts(const s_t* tr_sp, const s_t* parent,
                         const s_t* post, s_t* counts, s_t* w);

    /** \brief Calculate the number of nonzeros in the QR V matrix
      * Ref: Chapter 5, Direct Methods for Sparse Linear Systems by Tim Davis
      * len[w] >= nrow + 3*ncol
      * len[pinv] == nrow + ncol
      * len[leftmost] == nrow
      * Modified version of cs_sqr in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static s_t qr_nnz(const s_t* sp, s_t* pinv, s_t* leftmost,
                      const s_t* parent, s_t* nrow_ext, s_t* w);

    /** \brief Setup QP solver
      * Ref: Chapter 5, Direct Methods for Sparse Linear Systems by Tim Davis
      * len[w] >= nrow + 7*ncol + 1
      * len[pinv] == nrow + ncol
      * len[leftmost] == nrow
      */
    static void qr_init(const s_t* sp, const s_t* sp_tr,
                        s_t* leftmost, s_t* parent, s_t* pinv,
                        s_t* nrow_ext, s_t* v_nnz, s_t* r_nnz, s_t* w);

    /** \brief Get the row indices for V and R in QR factorization
      * Ref: Chapter 5, Direct Methods for Sparse Linear Systems by Tim Davis
      * Note: nrow <= nrow_ext <= nrow+ncol
      * len[iw] = nrow_ext + ncol
      * len[x] = nrow_ext
      * sp_v = [nrow_ext, ncol, 0, 0, ...] len[3 + ncol + nnz_v]
      * len[v] nnz_v
      * sp_r = [nrow_ext, ncol, 0, 0, ...] len[3 + ncol + nnz_r]
      * len[r] nnz_r
      * len[beta] ncol
      * Modified version of cs_qr in CSparse
      * Copyright(c) Timothy A. Davis, 2006-2009
      * Licensed as a derivative work under the GNU LGPL
      */
    static void qr_sparsities(const s_t* sp_a, s_t nrow_ext, s_t* sp_v, s_t* sp_r,
                              const s_t* leftmost, const s_t* parent, const s_t* pinv,
                              s_t* iw);

    /// Transpose the matrix
    Sparsity T() const;

    /** \brief Transpose the matrix and get the reordering of the non-zero entries,
     *
     * \param[out] mapping the non-zeros of the original matrix for each non-zero of the new matrix
     */
    Sparsity transpose(std::vector<s_t>& mapping, bool invert_mapping=false) const;

    /// Check if the sparsity is the transpose of another
    bool is_transpose(const SparsityInternal& y) const;

    /// Check if the sparsity is a reshape of another
    bool is_reshape(const SparsityInternal& y) const;

    /// Breadth-first search for coarse decomposition: see cs_bfs in CSparse
    void breadthFirstSearch(s_t n, std::vector<s_t>& wi, std::vector<s_t>& wj,
                            std::vector<s_t>& queue, const std::vector<s_t>& imatch,
                            const std::vector<s_t>& jmatch, s_t mark) const;

    /// Collect matched cols and rows into p and q: see cs_matched in CSparse
    static void matched(s_t n, const std::vector<s_t>& wj, const std::vector<s_t>& imatch,
                        std::vector<s_t>& p, std::vector<s_t>& q, std::vector<s_t>& cc,
                        std::vector<s_t>& rr, s_t set, s_t mark);

    /// Collect unmatched cols into the permutation vector p : see cs_unmatched in CSparse
    static void unmatched(s_t m, const std::vector<s_t>& wi, std::vector<s_t>& p,
                          std::vector<s_t>& rr, s_t set);

    /// return 1 if col i is in R2 : see cs_rprune in CSparse
    static s_t rprune(s_t i, s_t j, double aij, void *other);

    /** \brief drop entries for which fkeep(A(i, j)) is false; return nz if OK, else -1: :
     * see cs_fkeep in CSparse
     */
    static s_t drop(s_t (*fkeep)(s_t, s_t, double, void *), void *other,
                    s_t nrow, s_t ncol,
                    std::vector<s_t>& colind, std::vector<s_t>& row);

    /// Compute the Dulmage-Mendelsohn decomposition : see cs_dmperm in CSparse
    s_t btf(std::vector<s_t>& rowperm, std::vector<s_t>& colperm,
                          std::vector<s_t>& rowblock, std::vector<s_t>& colblock,
                          std::vector<s_t>& coarse_rowblock,
                          std::vector<s_t>& coarse_colblock) const {
      T()->dmperm(colperm, rowperm, colblock, rowblock,
                  coarse_colblock, coarse_rowblock);
      return rowblock.size()-1;
    }

    /// Get cached block triangular form
    const Btf& btf() const;


    /** \brief Compute the Dulmage-Mendelsohn decomposition
     *
     * -- upper triangular TODO: refactor and merge with the above
     */
    void dmperm(std::vector<s_t>& rowperm, std::vector<s_t>& colperm,
                std::vector<s_t>& rowblock, std::vector<s_t>& colblock,
                std::vector<s_t>& coarse_rowblock,
                std::vector<s_t>& coarse_colblock) const;

    /// Compute the maximum transversal (maximum matching): see cs_maxtrans in CSparse
    void maxTransversal(std::vector<s_t>& imatch,
                        std::vector<s_t>& jmatch, Sparsity& trans, s_t seed) const;

    /// Find an augmenting path: see cs_augment in CSparse
    void augmentingPath(s_t k, std::vector<s_t>& jmatch,
                        s_t *cheap, std::vector<s_t>& w, s_t *js, s_t *is, s_t *ps) const;

    /**
     * return a random permutation vector, the identity perm, or p = n-1:-1:0.
     * seed = -1 means p = n-1:-1:0.  seed = 0 means p = identity.
     * otherwise p = random permutation. See cs_randperm in CSparse
     */
    static std::vector<s_t> randomPermutation(s_t n, s_t seed);

    /// Invert a permutation matrix: see cs_pinv in CSparse
    static std::vector<s_t> invertPermutation(const std::vector<s_t>& p);

    /// C = A(p, q) where p and q are permutations of 0..m-1 and 0..n-1.: see cs_permute in CSparse
    Sparsity permute(const std::vector<s_t>& pinv, const std::vector<s_t>& q, s_t values) const;

    /// C = A(p, q) where p and q are permutations of 0..m-1 and 0..n-1.: see cs_permute in CSparse
    void permute(const std::vector<s_t>& pinv,
                 const std::vector<s_t>& q, s_t values,
                 std::vector<s_t>& colind_C,
                 std::vector<s_t>& row_C) const;

    /// clear w: cs_wclear in CSparse
    static s_t wclear(s_t mark, s_t lemax, s_t *w, s_t n);

    /// keep off-diagonal entries; drop diagonal entries: See cs_diag in CSparse
    static s_t diag(s_t i, s_t j, double aij, void *other);

    /// C = A*B: See cs_multiply in CSparse
    Sparsity multiply(const Sparsity& B) const;

    /** x = x + beta * A(:, j), where x is a dense vector and A(:, j) is sparse:
     * See cs_scatter in CSparse
     */
    s_t scatter(s_t j, std::vector<s_t>& w, s_t mark, s_t* Ci, s_t nz) const;

    /// Get row() as a vector
    std::vector<s_t> get_row() const;

    /// Get colind() as a vector
    std::vector<s_t> get_colind() const;

    /// Get the column for each nonzero
    std::vector<s_t> get_col() const;

    /// Resize
    Sparsity _resize(s_t nrow, s_t ncol) const;

    /// Reshape a sparsity, order of nonzeros remains the same
    Sparsity _reshape(s_t nrow, s_t ncol) const;

    /// Number of elements
    s_t numel() const;

    /// Number of non-zeros in the lower triangular half
    s_t nnz_lower(bool strictly=false) const;

    /// Number of non-zeros in the upper triangular half
    s_t nnz_upper(bool strictly=false) const;

    /// Number of non-zeros on the diagonal
    s_t nnz_diag() const;

    /** \brief Upper half-bandwidth */
    s_t bw_upper() const;

    /** \brief Lower half-bandwidth */
    s_t bw_lower() const;

    /// Shape
    std::pair<s_t, s_t> size() const;

    /// Is scalar?
    bool is_scalar(bool scalar_and_dense) const;

    /** \brief Check if the sparsity is empty
     *
     * A sparsity is considered empty if one of the dimensions is zero
     * (or optionally both dimensions)
     */
    bool is_empty(bool both=false) const;

    /// Is dense?
    bool is_dense() const;

    /** \brief  Check if the pattern is a row vector (i.e. size1()==1) */
    bool is_row() const;

    /** \brief  Check if the pattern is a column vector (i.e. size2()==1) */
    bool is_column() const;

    /** \brief  Check if the pattern is a row or column vector */
    bool is_vector() const;

    /// Is diagonal?
    bool is_diag() const;

    /// Is square?
    bool is_square() const;

    /// Is symmetric?
    bool is_symmetric() const;

    /// Is lower triangular?
    bool is_tril() const;

    /// is upper triangular?
    bool is_triu() const;

    /// Get upper triangular part
    Sparsity _triu(bool includeDiagonal) const;

    /// Get lower triangular part
    Sparsity _tril(bool includeDiagonal) const;

    /// Get nonzeros in lower triangular part
    std::vector<s_t> get_lower() const;

    /// Get nonzeros in upper triangular part
    std::vector<s_t> get_upper() const;

    /// Get the dimension as a string
    std::string dim(bool with_nz=false) const;

    /// Describe the nonzero location k as a string
    std::string repr_el(s_t k) const;

    /// Sparsity pattern for a matrix-matrix product (details in public class)
    Sparsity _mtimes(const Sparsity& y) const;

    ///@{
    /// Union of two sparsity patterns
    Sparsity combine(const Sparsity& y, bool f0x_is_zero, bool function0_is_zero,
                            std::vector<unsigned char>& mapping) const;
    Sparsity combine(const Sparsity& y, bool f0x_is_zero, bool function0_is_zero) const;

    template<bool with_mapping>
    Sparsity combineGen1(const Sparsity& y, bool f0x_is_zero, bool function0_is_zero,
                                std::vector<unsigned char>& mapping) const;

    template<bool with_mapping, bool f0x_is_zero, bool function0_is_zero>
    Sparsity combineGen(const Sparsity& y, std::vector<unsigned char>& mapping) const;
    ///@}

    /// Take the inverse of a sparsity pattern; flip zeros and non-zeros
    Sparsity pattern_inverse() const;

    /// Check if two sparsity patterns are the same
    bool is_equal(const Sparsity& y) const;

    /// Check if two sparsity patterns are the same
    bool is_equal(s_t y_nrow, s_t y_ncol, const std::vector<s_t>& y_colind,
                 const std::vector<s_t>& y_row) const;

    /// Check if two sparsity patterns are the same
    bool is_equal(s_t y_nrow, s_t y_ncol, const s_t* y_colind, const s_t* y_row) const;

    /// Enlarge the matrix along the first dimension (i.e. insert rows)
    Sparsity _enlargeRows(s_t nrow, const std::vector<s_t>& rr, bool ind1) const;

    /// Enlarge the matrix along the second dimension (i.e. insert columns)
    Sparsity _enlargeColumns(s_t ncol, const std::vector<s_t>& cc, bool ind1) const;

    /// Make a patten dense
    Sparsity makeDense(std::vector<s_t>& mapping) const;

    /// Erase rows and/or columns - does bounds checking
    Sparsity _erase(const std::vector<s_t>& rr, const std::vector<s_t>& cc,
                      bool ind1, std::vector<s_t>& mapping) const;

    /// Erase elements
    Sparsity _erase(const std::vector<s_t>& rr, bool ind1,
                      std::vector<s_t>& mapping) const;

    /// Append another sparsity patten vertically (vectors only)
    Sparsity _appendVector(const SparsityInternal& sp) const;

    /// Append another sparsity patten horizontally
    Sparsity _appendColumns(const SparsityInternal& sp) const;

    /** \brief Get a submatrix
    * Does bounds checking
    * rr and rr are not required to be monotonous
    */
    Sparsity sub(const std::vector<s_t>& rr, const std::vector<s_t>& cc,
                 std::vector<s_t>& mapping, bool ind1) const;

    /** \brief Get a set of elements
    * Does bounds checking
    * rr is not required to be monotonous
    */
    Sparsity sub(const std::vector<s_t>& rr, const SparsityInternal& sp,
                 std::vector<s_t>& mapping, bool ind1) const;

    /// Get the index of an existing non-zero element
    s_t get_nz(s_t rr, s_t cc) const;

    /// Get a set of non-zero element - does bounds checking
    std::vector<s_t> get_nz(const std::vector<s_t>& rr, const std::vector<s_t>& cc) const;

    /// Get the nonzero index for a set of elements (see description in public class)
    void get_nz(std::vector<s_t>& indices) const;

    /// Does the rows appear sequentially on each col
    bool rowsSequential(bool strictly) const;

    /** \brief Remove duplicate entries
     *
     * The same indices will be removed from the mapping vector,
     * which must have the same length as the number of nonzeros
     */
    Sparsity _removeDuplicates(std::vector<s_t>& mapping) const;

    /// Get element index for each nonzero
    void find(std::vector<s_t>& loc, bool ind1) const;

    /// Hash the sparsity pattern
    std::size_t hash() const;

    /// Readable name of the internal class
    std::string class_name() const override {return "SparsityInternal";}

    /// Print description
    void disp(std::ostream& stream, bool more) const override;

    /** \brief Perform a unidirectional coloring
     *
     * A greedy distance-2 coloring algorithm
     * (Algorithm 3.1 in A. H. GEBREMEDHIN, F. MANNE, A. POTHEN)
     */
    Sparsity uni_coloring(const Sparsity& AT, s_t cutoff) const;

    /** \brief A greedy distance-2 coloring algorithm
     * See description in public class.
     */
    Sparsity star_coloring(s_t ordering, s_t cutoff) const;

    /** \brief An improved distance-2 coloring algorithm
     * See description in public class.
     */
    Sparsity star_coloring2(s_t ordering, s_t cutoff) const;

    /// Order the columns by decreasing degree
    std::vector<s_t> largest_first() const;

    /// Permute rows and/or columns
    Sparsity pmult(const std::vector<s_t>& p, bool permute_rows=true, bool permute_cols=true,
                   bool invert_permutation=false) const;

    /** \brief Print a textual representation of sparsity */
    void spy(std::ostream &stream) const;

    /// Generate a script for Matlab or Octave which visualizes the sparsity using the spy command
    void spy_matlab(const std::string& mfile) const;

    /** \brief Export sparsity in Matlab format */
    void export_code(const std::string& lang, std::ostream &stream,
       const Dict& options) const;

    /// Propagate sparsity through a linear solve
    void spsolve(bvec_t* X, const bvec_t* B, bool tr) const;
};

} // namespace casadi
/// \endcond

#endif // CASADI_SPARSITY_INTERNAL_HPP
