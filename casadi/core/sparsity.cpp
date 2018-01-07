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


#include "sparsity_internal.hpp"
#include "matrix.hpp"
#include "casadi_misc.hpp"
#include "sparse_storage_impl.hpp"
#include <climits>

using namespace std;

namespace casadi {
  // Instantiate templates
  template class SparseStorage<Sparsity>;

  /// \cond INTERNAL
  // Singletons
  class EmptySparsity : public Sparsity {
  public:
    EmptySparsity() {
      const s_t colind[1] = {0};
      own(new SparsityInternal(0, 0, colind, 0));
    }
  };

  class ScalarSparsity : public Sparsity {
  public:
    ScalarSparsity() {
      const s_t colind[2] = {0, 1};
      const s_t row[1] = {0};
      own(new SparsityInternal(1, 1, colind, row));
    }
  };

  class ScalarSparseSparsity : public Sparsity {
  public:
    ScalarSparseSparsity() {
      const s_t colind[2] = {0, 0};
      const s_t row[1] = {0};
      own(new SparsityInternal(1, 1, colind, row));
    }
  };
  /// \endcond

  Sparsity::Sparsity(s_t dummy) {
    casadi_assert_dev(dummy==0);
  }

  Sparsity Sparsity::create(SparsityInternal *node) {
    Sparsity ret;
    ret.own(node);
    return ret;
  }

  Sparsity::Sparsity(s_t nrow, s_t ncol) {
    casadi_assert_dev(nrow>=0);
    casadi_assert_dev(ncol>=0);
    std::vector<s_t> row, colind(ncol+1, 0);
    assign_cached(nrow, ncol, colind, row);
    sanity_check(true);
  }

  Sparsity::Sparsity(const std::pair<s_t, s_t>& rc) {
    casadi_assert_dev(rc.first>=0);
    casadi_assert_dev(rc.second>=0);
    std::vector<s_t> row, colind(rc.second+1, 0);
    assign_cached(rc.first, rc.second, colind, row);
    sanity_check(true);
  }

  Sparsity::Sparsity(s_t nrow, s_t ncol, const std::vector<s_t>& colind,
                     const std::vector<s_t>& row) {
    casadi_assert_dev(nrow>=0);
    casadi_assert_dev(ncol>=0);
    assign_cached(nrow, ncol, colind, row);
    sanity_check(true);
  }

  Sparsity::Sparsity(s_t nrow, s_t ncol, const s_t* colind, const s_t* row) {
    casadi_assert_dev(nrow>=0);
    casadi_assert_dev(ncol>=0);
    if (colind==0 || colind[ncol]==nrow*ncol) {
      *this = dense(nrow, ncol);
    } else {
      vector<s_t> colindv(colind, colind+ncol+1);
      vector<s_t> rowv(row, row+colind[ncol]);
      assign_cached(nrow, ncol, colindv, rowv);
      sanity_check(true);
    }
  }

  const SparsityInternal* Sparsity::operator->() const {
    return static_cast<const SparsityInternal*>(SharedObject::operator->());
  }

  const SparsityInternal& Sparsity::operator*() const {
    return *static_cast<const SparsityInternal*>(get());
  }

  bool Sparsity::test_cast(const SharedObjectInternal* ptr) {
    return dynamic_cast<const SparsityInternal*>(ptr)!=0;
  }

  s_t Sparsity::size1() const {
    return (*this)->size1();
  }

  s_t Sparsity::size2() const {
    return (*this)->size2();
  }

  s_t Sparsity::numel() const {
    return (*this)->numel();
  }

  double Sparsity::density() const {
    double r = 100;
    r *= nnz();
    r /= size1();
    r /= size2();
    return r;
  }

  bool Sparsity::is_empty(bool both) const {
    return (*this)->is_empty(both);
  }

  s_t Sparsity::nnz() const {
    return (*this)->nnz();
  }

  std::pair<s_t, s_t> Sparsity::size() const {
    return (*this)->size();
  }

  s_t Sparsity::size(s_t axis) const {
    switch (axis) {
    case 1: return size1();
    case 2: return size2();
    }
    casadi_error("Axis must be 1 or 2.");
  }

  const s_t* Sparsity::row() const {
    return (*this)->row();
  }

  const s_t* Sparsity::colind() const {
    return (*this)->colind();
  }

  s_t Sparsity::row(s_t el) const {
    if (el<0 || el>=nnz()) {
      throw std::out_of_range("Sparsity::row: Index " + str(el)
        + " out of range [0," + str(nnz()) + ")");
    }
    return row()[el];
  }

  s_t Sparsity::colind(s_t cc) const {
    if (cc<0 || cc>size2()) {
      throw std::out_of_range("Sparsity::colind: Index "
        + str(cc) + " out of range [0," + str(size2()) + "]");
    }
    return colind()[cc];
  }

  void Sparsity::sanity_check(bool complete) const {
    (*this)->sanity_check(complete);
  }

  void Sparsity::resize(s_t nrow, s_t ncol) {
    if (size1()!=nrow || size2() != ncol) {
      *this = (*this)->_resize(nrow, ncol);
    }
  }

  s_t Sparsity::add_nz(s_t rr, s_t cc) {
    // If negative index, count from the back
    if (rr<0) rr += size1();
    if (cc<0) cc += size2();

    // Check consistency
    casadi_assert(rr>=0 && rr<size1(), "Row index out of bounds");
    casadi_assert(cc>=0 && cc<size2(), "Column index out of bounds");

    // Quick return if matrix is dense
    if (is_dense()) return rr+cc*size1();

    // Get sparsity pattern
    s_t size1=this->size1(), size2=this->size2(), nnz=this->nnz();
    const s_t *colind = this->colind(), *row = this->row();

    // Quick return if we are adding an element to the end
    if (colind[cc]==nnz || (colind[cc+1]==nnz && row[nnz-1]<rr)) {
      std::vector<s_t> rowv(nnz+1);
      copy(row, row+nnz, rowv.begin());
      rowv[nnz] = rr;
      std::vector<s_t> colindv(colind, colind+size2+1);
      for (s_t c=cc; c<size2; ++c) colindv[c+1]++;
      assign_cached(size1, size2, colindv, rowv);
      return rowv.size()-1;
    }

    // go to the place where the element should be
    s_t ind;
    for (ind=colind[cc]; ind<colind[cc+1]; ++ind) { // better: loop from the back to the front
      if (row[ind] == rr) {
        return ind; // element exists
      } else if (row[ind] > rr) {
        break;                // break at the place where the element should be added
      }
    }

    // insert the element
    std::vector<s_t> rowv = get_row(), colindv = get_colind();
    rowv.insert(rowv.begin()+ind, rr);
    for (s_t c=cc+1; c<size2+1; ++c) colindv[c]++;

    // Return the location of the new element
    assign_cached(size1, size2, colindv, rowv);
    return ind;
  }

  bool Sparsity::has_nz(s_t rr, s_t cc) const {
    return get_nz(rr, cc)!=-1;
  }


  s_t Sparsity::get_nz(s_t rr, s_t cc) const {
    return (*this)->get_nz(rr, cc);
  }

  Sparsity Sparsity::reshape(const Sparsity& x, const Sparsity& sp) {
    casadi_assert_dev(x.is_reshape(sp));
    return sp;
  }

  Sparsity Sparsity::reshape(const Sparsity& x, s_t nrow, s_t ncol) {
    return x->_reshape(nrow, ncol);
  }

  std::vector<s_t> Sparsity::get_nz(const std::vector<s_t>& rr, const std::vector<s_t>& cc) const {
    return (*this)->get_nz(rr, cc);
  }

  bool Sparsity::is_scalar(bool scalar_and_dense) const {
    return (*this)->is_scalar(scalar_and_dense);
  }

  bool Sparsity::is_dense() const {
    return (*this)->is_dense();
  }

  bool Sparsity::is_diag() const {
    return (*this)->is_diag();
  }

  bool Sparsity::is_row() const {
    return (*this)->is_row();
  }

  bool Sparsity::is_column() const {
    return (*this)->is_column();
  }

  bool Sparsity::is_vector() const {
    return (*this)->is_vector();
  }

  bool Sparsity::is_square() const {
    return (*this)->is_square();
  }

  bool Sparsity::is_symmetric() const {
    return (*this)->is_symmetric();
  }

  bool Sparsity::is_tril() const {
    return (*this)->is_tril();
  }

  bool Sparsity::is_triu() const {
    return (*this)->is_triu();
  }

  Sparsity Sparsity::sub(const std::vector<s_t>& rr, const Sparsity& sp,
                         std::vector<s_t>& mapping, bool ind1) const {
    return (*this)->sub(rr, *sp, mapping, ind1);
  }

  Sparsity Sparsity::sub(const std::vector<s_t>& rr, const std::vector<s_t>& cc,
                         std::vector<s_t>& mapping, bool ind1) const {
    return (*this)->sub(rr, cc, mapping, ind1);
  }

  std::vector<s_t> Sparsity::erase(const std::vector<s_t>& rr, const std::vector<s_t>& cc,
                                   bool ind1) {
    vector<s_t> mapping;
    *this = (*this)->_erase(rr, cc, ind1, mapping);
    return mapping;
  }

  std::vector<s_t> Sparsity::erase(const std::vector<s_t>& rr, bool ind1) {
    vector<s_t> mapping;
    *this = (*this)->_erase(rr, ind1, mapping);
    return mapping;
  }

  s_t Sparsity::nnz_lower(bool strictly) const {
    return (*this)->nnz_lower(strictly);
  }

  s_t Sparsity::nnz_upper(bool strictly) const {
    return (*this)->nnz_upper(strictly);
  }

  s_t Sparsity::nnz_diag() const {
    return (*this)->nnz_diag();
  }

  std::vector<s_t> Sparsity::get_colind() const {
    return (*this)->get_colind();
  }

  std::vector<s_t> Sparsity::get_col() const {
    return (*this)->get_col();
  }

  std::vector<s_t> Sparsity::get_row() const {
    return (*this)->get_row();
  }

  void Sparsity::get_ccs(std::vector<s_t>& colind, std::vector<s_t>& row) const {
    colind = get_colind();
    row = get_row();
  }

  void Sparsity::get_crs(std::vector<s_t>& rowind, std::vector<s_t>& col) const {
    T().get_ccs(rowind, col);
  }

  void Sparsity::get_triplet(std::vector<s_t>& row, std::vector<s_t>& col) const {
    row = get_row();
    col = get_col();
  }

  Sparsity Sparsity::transpose(std::vector<s_t>& mapping, bool invert_mapping) const {
    return (*this)->transpose(mapping, invert_mapping);
  }

  Sparsity Sparsity::T() const {
    return (*this)->T();
  }

  Sparsity Sparsity::combine(const Sparsity& y, bool f0x_is_zero,
                                    bool function0_is_zero,
                                    std::vector<unsigned char>& mapping) const {
    return (*this)->combine(y, f0x_is_zero, function0_is_zero, mapping);
  }

  Sparsity Sparsity::combine(const Sparsity& y, bool f0x_is_zero,
                                    bool function0_is_zero) const {
    return (*this)->combine(y, f0x_is_zero, function0_is_zero);
  }

  Sparsity Sparsity::unite(const Sparsity& y, std::vector<unsigned char>& mapping) const {
    return (*this)->combine(y, false, false, mapping);
  }

  Sparsity Sparsity::unite(const Sparsity& y) const {
    return (*this)->combine(y, false, false);
  }

  Sparsity Sparsity::intersect(const Sparsity& y,
                                         std::vector<unsigned char>& mapping) const {
    return (*this)->combine(y, true, true, mapping);
  }

  Sparsity Sparsity::intersect(const Sparsity& y) const {
    return (*this)->combine(y, true, true);
  }

  Sparsity Sparsity::mtimes(const Sparsity& x, const Sparsity& y) {
    // Check matching dimensions
    casadi_assert(x.size2()==y.size1(),
      "Matrix product with incompatible dimensions. Lhs is "
      + x.dim() + " and rhs is " + y.dim() + ".");

    return x->_mtimes(y);
  }

  bool Sparsity::is_equal(const Sparsity& y) const {
    return (*this)->is_equal(y);
  }

  bool Sparsity::is_equal(s_t nrow, s_t ncol, const std::vector<s_t>& colind,
                         const std::vector<s_t>& row) const {
    return (*this)->is_equal(nrow, ncol, colind, row);
  }

  bool Sparsity::is_equal(s_t nrow, s_t ncol, const s_t* colind, const s_t* row) const {
    return (*this)->is_equal(nrow, ncol, colind, row);
  }

  Sparsity Sparsity::operator+(const Sparsity& b) const {
    return unite(b);
  }

  Sparsity Sparsity::operator*(const Sparsity& b) const {
    std::vector< unsigned char > mapping;
    return intersect(b, mapping);
  }

  Sparsity Sparsity::pattern_inverse() const {
    return (*this)->pattern_inverse();
  }

  void Sparsity::append(const Sparsity& sp) {
    if (sp.size1()==0 && sp.size2()==0) {
      // Appending pattern is empty
      return;
    } else if (size1()==0 && size2()==0) {
      // This is empty
      *this = sp;
    } else {
      casadi_assert(size2()==sp.size2(),
                            "Sparsity::append: Dimension mismatch. "
                            "You attempt to append a shape " + sp.dim()
                            + " to a shape " + dim()
                            + ". The number of columns must match.");
      if (sp.size1()==0) {
        // No rows to add
        return;
      } else if (size1()==0) {
        // No rows before
        *this = sp;
      } else if (is_column()) {
        // Append to vector (inefficient)
        *this = (*this)->_appendVector(*sp);
      } else {
        // Append to matrix (inefficient)
        *this = vertcat({*this, sp});
      }
    }
  }

  void Sparsity::appendColumns(const Sparsity& sp) {
    if (sp.size1()==0 && sp.size2()==0) {
      // Appending pattern is empty
      return;
    } else if (size1()==0 && size2()==0) {
      // This is empty
      *this = sp;
    } else {
      casadi_assert(size1()==sp.size1(),
                            "Sparsity::appendColumns: Dimension mismatch. You attempt to "
                            "append a shape " + sp.dim() + " to a shape "
                            + dim() + ". The number of rows must match.");
      if (sp.size2()==0) {
        // No columns to add
        return;
      } else if (size2()==0) {
        // No columns before
        *this = sp;
      } else {
        // Append to matrix (expensive)
        *this = (*this)->_appendColumns(*sp);
      }
    }
  }

  Sparsity::CachingMap& Sparsity::getCache() {
    static CachingMap ret;
    return ret;
  }

  const Sparsity& Sparsity::getScalar() {
    static ScalarSparsity ret;
    return ret;
  }

  const Sparsity& Sparsity::getScalarSparse() {
    static ScalarSparseSparsity ret;
    return ret;
  }

  const Sparsity& Sparsity::getEmpty() {
    static EmptySparsity ret;
    return ret;
  }

  void Sparsity::enlarge(s_t nrow, s_t ncol, const std::vector<s_t>& rr,
                         const std::vector<s_t>& cc, bool ind1) {
    enlargeColumns(ncol, cc, ind1);
    enlargeRows(nrow, rr, ind1);
  }

  void Sparsity::enlargeColumns(s_t ncol, const std::vector<s_t>& cc, bool ind1) {
    casadi_assert_dev(cc.size() == size2());
    if (cc.empty()) {
      *this = Sparsity(size1(), ncol);
    } else {
      *this = (*this)->_enlargeColumns(ncol, cc, ind1);
    }
  }

  void Sparsity::enlargeRows(s_t nrow, const std::vector<s_t>& rr, bool ind1) {
    casadi_assert_dev(rr.size() == size1());
    if (rr.empty()) {
      *this = Sparsity(nrow, size2());
    } else {
      *this = (*this)->_enlargeRows(nrow, rr, ind1);
    }
  }

  Sparsity Sparsity::diag(s_t nrow, s_t ncol) {
    // Smallest dimension
    s_t n = min(nrow, ncol);

    // Column offset
    vector<s_t> colind(ncol+1, n);
    for (s_t cc=0; cc<n; ++cc) colind[cc] = cc;

    // Row
    vector<s_t> row = range(n);

    // Create pattern from vectors
    return Sparsity(nrow, ncol, colind, row);
  }

  Sparsity Sparsity::makeDense(std::vector<s_t>& mapping) const {
    return (*this)->makeDense(mapping);
  }

  std::string Sparsity::dim(bool with_nz) const {
    return (*this)->dim(with_nz);
  }

  std::string Sparsity::postfix_dim() const {
    if (is_dense()) {
      if (is_scalar()) {
        return "";
      } else if (is_empty(true)) {
        return "[]";
      } else if (is_column()) {
        return "[" + str(size1()) + "]";
      } else {
        return "[" + dim(false) + "]";
      }
    } else {
      return "[" + dim(true) + "]";
    }
  }

  std::string Sparsity::repr_el(s_t k) const {
    return (*this)->repr_el(k);
  }

  Sparsity Sparsity::get_diag(std::vector<s_t>& mapping) const {
    return (*this)->get_diag(mapping);
  }

  std::vector<s_t> Sparsity::etree(bool ata) const {
    vector<s_t> parent(size2()), w(size1() + size2());
    SparsityInternal::etree(*this, get_ptr(parent), get_ptr(w), ata);
    return parent;
  }

  Sparsity Sparsity::ldl(std::vector<s_t>& parent) const {
    casadi_assert(is_symmetric(),
                 "LDL factorization requires a symmetric matrix");
    // Dimension
    s_t n=size1();
    // Work vector
    std::vector<s_t> w(3*n);
    // Elimination tree
    parent.resize(n);
    // Calculate colind in L (strictly lower entries only)
    std::vector<s_t> L_colind(1+n);
    casadi_ldl_colind(*this, get_ptr(parent), get_ptr(L_colind), get_ptr(w));
    // Get rows in L (strictly lower entries only)
    std::vector<s_t> L_row(L_colind.back());
    casadi_ldl_row(*this, get_ptr(parent), get_ptr(L_colind), get_ptr(L_row),
                    get_ptr(w));
    // Sparsity of L
    return Sparsity(n, n, L_colind, L_row);
  }

  void Sparsity::qr_sparse(Sparsity& V, Sparsity& R, std::vector<s_t>& pinv,
                           std::vector<s_t>& leftmost, std::vector<s_t>& parent) const {
    // Dimensions
    s_t size1=this->size1(), size2=this->size2();

    // Allocate memory
    leftmost.resize(size1);
    parent.resize(size2);
    pinv.resize(size1 + size2);
    vector<s_t> iw(size1 + 7*size2 + 1);

    // Initialize QP solve
    s_t nrow_ext, v_nnz, r_nnz;
    SparsityInternal::qr_init(*this, T(),
                              get_ptr(leftmost), get_ptr(parent), get_ptr(pinv),
                              &nrow_ext, &v_nnz, &r_nnz, get_ptr(iw));

    // Calculate sparsities
    vector<s_t> sp_v(2 + size2 + 1 + v_nnz);
    vector<s_t> sp_r(2 + size2 + 1 + r_nnz);
    SparsityInternal::qr_sparsities(*this, nrow_ext, get_ptr(sp_v), get_ptr(sp_r),
                                    get_ptr(leftmost), get_ptr(parent), get_ptr(pinv),
                                    get_ptr(iw));
    V = compressed(sp_v);
    R = compressed(sp_r);
  }

  void Sparsity::symbfact(std::vector<s_t>& count, std::vector<s_t>& parent,
                          std::vector<s_t>& post, Sparsity& L, bool ata) const {
    // Dimensions
    s_t size1=this->size1(), size2=this->size2();
    // Allocate vectors
    std::vector<s_t> w;
    parent.resize(size2);
    post.resize(size2);
    count.resize(size2);

    if (ata) {
      // Allocate work
      w.resize(size1+size2);
      // Calculate elimination tree
      SparsityInternal::etree(*this, get_ptr(parent), get_ptr(w), ata);
      // Calculate postorder
      w.resize(3*size2);
      SparsityInternal::postorder(get_ptr(parent), size2, get_ptr(post), get_ptr(w));
      // Calculate colind in L
      w.resize(size1 + 5*size2 + 1);
      SparsityInternal::qr_counts(T(), get_ptr(parent), get_ptr(post),
                                  get_ptr(count), get_ptr(w));
      // Not implemented
      L = Sparsity();
    } else {
      // Symbolic LDL factorization
      L = ldl(parent) + diag(size2, size2);
      // Calculate postorder
      w.resize(3*size2);
      SparsityInternal::postorder(get_ptr(parent), size2, get_ptr(post), get_ptr(w));
      // Calculate column counts
      const s_t* L_colind = L.colind();
      for (s_t i=0; i<size2; ++i) count[i] = L_colind[i+1] - L_colind[i];
    }
  }

  s_t Sparsity::dfs(s_t j, s_t top, std::vector<s_t>& xi,
                                 std::vector<s_t>& pstack, const std::vector<s_t>& pinv,
                                 std::vector<bool>& marked) const {
    return (*this)->dfs(j, top, xi, pstack, pinv, marked);
  }

  s_t Sparsity::scc(std::vector<s_t>& p, std::vector<s_t>& r) const {
    return (*this)->scc(p, r);
  }

  std::vector<s_t> Sparsity::amd() const {
    return (*this)->amd();
  }

  s_t Sparsity::btf(std::vector<s_t>& rowperm, std::vector<s_t>& colperm,
                                  std::vector<s_t>& rowblock, std::vector<s_t>& colblock,
                                  std::vector<s_t>& coarse_rowblock,
                                  std::vector<s_t>& coarse_colblock) const {
    return (*this)->btf(rowperm, colperm, rowblock, colblock,
                        coarse_rowblock, coarse_colblock);
  }

  void Sparsity::spsolve(bvec_t* X, const bvec_t* B, bool tr) const {
    (*this)->spsolve(X, B, tr);
  }

  bool Sparsity::rowsSequential(bool strictly) const {
    return (*this)->rowsSequential(strictly);
  }

  void Sparsity::removeDuplicates(std::vector<s_t>& mapping) {
    *this = (*this)->_removeDuplicates(mapping);
  }

  std::vector<s_t> Sparsity::find(bool ind1) const {
    std::vector<s_t> loc;
    find(loc, ind1);
    return loc;
  }

  void Sparsity::find(std::vector<s_t>& loc, bool ind1) const {
    (*this)->find(loc, ind1);
  }

  void Sparsity::get_nz(std::vector<s_t>& indices) const {
    (*this)->get_nz(indices);
  }

  Sparsity Sparsity::uni_coloring(const Sparsity& AT, s_t cutoff) const {
    if (AT.is_null()) {
      return (*this)->uni_coloring(T(), cutoff);
    } else {
      return (*this)->uni_coloring(AT, cutoff);
    }
  }

  Sparsity Sparsity::star_coloring(s_t ordering, s_t cutoff) const {
    return (*this)->star_coloring(ordering, cutoff);
  }

  Sparsity Sparsity::star_coloring2(s_t ordering, s_t cutoff) const {
    return (*this)->star_coloring2(ordering, cutoff);
  }

  std::vector<s_t> Sparsity::largest_first() const {
    return (*this)->largest_first();
  }

  Sparsity Sparsity::pmult(const std::vector<s_t>& p, bool permute_rows, bool permute_columns,
                           bool invert_permutation) const {
    return (*this)->pmult(p, permute_rows, permute_columns, invert_permutation);
  }

  void Sparsity::spy_matlab(const std::string& mfile) const {
    (*this)->spy_matlab(mfile);
  }

  void Sparsity::export_code(const std::string& lang, std::ostream &stream,
      const Dict& options) const {
    (*this)->export_code(lang, stream, options);
  }

  void Sparsity::spy(std::ostream &stream) const {
    (*this)->spy(stream);
  }

  bool Sparsity::is_transpose(const Sparsity& y) const {
    return (*this)->is_transpose(*y);
  }

  bool Sparsity::is_reshape(const Sparsity& y) const {
    return (*this)->is_reshape(*y);
  }

  std::size_t Sparsity::hash() const {
    return (*this)->hash();
  }

  void Sparsity::assign_cached(s_t nrow, s_t ncol, const std::vector<s_t>& colind,
                              const std::vector<s_t>& row) {
    casadi_assert_dev(colind.size()==ncol+1);
    casadi_assert_dev(row.size()==colind.back());
    assign_cached(nrow, ncol, get_ptr(colind), get_ptr(row));
  }

  void Sparsity::assign_cached(s_t nrow, s_t ncol, const s_t* colind, const s_t* row) {
    // Scalars and empty patterns are handled separately
    if (ncol==0 && nrow==0) {
      // If empty
      *this = getEmpty();
      return;
    } else if (ncol==1 && nrow==1) {
      if (colind[ncol]==0) {
        // If sparse scalar
        *this = getScalarSparse();
        return;
      } else {
        // If dense scalar
        *this = getScalar();
        return;
      }
    }

    // Hash the pattern
    std::size_t h = hash_sparsity(nrow, ncol, colind, row);

    // Get a reference to the cache
    CachingMap& cache = getCache();

    // Record the current number of buckets (for garbage collection below)
    s_t bucket_count_before = cache.bucket_count();

    // WORKAROUND, functions do not appear to work when bucket_count==0
    if (bucket_count_before>0) {

      // Find the range of patterns equal to the key (normally only zero or one)
      pair<CachingMap::iterator, CachingMap::iterator> eq = cache.equal_range(h);

      // Loop over maching patterns
      for (CachingMap::iterator i=eq.first; i!=eq.second; ++i) {

        // Get a weak reference to the cached sparsity pattern
        WeakRef& wref = i->second;

        // Check if the pattern still exists
        if (wref.alive()) {

          // Get an owning reference to the cached pattern
          Sparsity ref = shared_cast<Sparsity>(wref.shared());

          // Check if the pattern matches
          if (ref.is_equal(nrow, ncol, colind, row)) {

            // Found match!
            own(ref.get());
            return;

          } else { // There is a hash rowision (unlikely, but possible)
            // Leave the pattern alone, continue to the next matching pattern
            continue;
          }
        } else {

          // Check if one of the other cache entries indeed has a matching sparsity
          CachingMap::iterator j=i;
          j++; // Start at the next matching key
          for (; j!=eq.second; ++j) {
            if (j->second.alive()) {

              // Recover cached sparsity
              Sparsity ref = shared_cast<Sparsity>(j->second.shared());

              // Match found if sparsity matches
              if (ref.is_equal(nrow, ncol, colind, row)) {
                own(ref.get());
                return;
              }
            }
          }

          // The cached entry has been deleted, create a new one
          own(new SparsityInternal(nrow, ncol, colind, row));

          // Cache this pattern
          wref = *this;

          // Return
          return;
        }
      }
    }

    // No matching sparsity pattern could be found, create a new one
    own(new SparsityInternal(nrow, ncol, colind, row));

    // Cache this pattern
    cache.insert(std::pair<std::size_t, WeakRef>(h, *this));

    // Garbage collection (currently only supported for unordered_multimap)
    s_t bucket_count_after = cache.bucket_count();

    // We we increased the number of buckets, take time to garbage-collect deleted references
    if (bucket_count_before!=bucket_count_after) {
      CachingMap::const_iterator i=cache.begin();
      while (i!=cache.end()) {
        if (!i->second.alive()) {
          i = cache.erase(i);
        } else {
          i++;
        }
      }
    }
  }

  Sparsity Sparsity::tril(const Sparsity& x, bool includeDiagonal) {
    return x->_tril(includeDiagonal);
  }

  Sparsity Sparsity::triu(const Sparsity& x, bool includeDiagonal) {
    return x->_triu(includeDiagonal);
  }

  std::vector<s_t> Sparsity::get_lower() const {
    return (*this)->get_lower();
  }

  std::vector<s_t> Sparsity::get_upper() const {
    return (*this)->get_upper();
  }


  std::size_t hash_sparsity(s_t nrow, s_t ncol, const std::vector<s_t>& colind,
                            const std::vector<s_t>& row) {
    return hash_sparsity(nrow, ncol, get_ptr(colind), get_ptr(row));
  }

  std::size_t hash_sparsity(s_t nrow, s_t ncol, const s_t* colind, const s_t* row) {
    // Condense the sparsity pattern to a single, deterministric number
    std::size_t ret=0;
    hash_combine(ret, nrow);
    hash_combine(ret, ncol);
    hash_combine(ret, colind, ncol+1);
    hash_combine(ret, row, colind[ncol]);
    return ret;
  }

  Sparsity Sparsity::dense(s_t nrow, s_t ncol) {
    casadi_assert_dev(nrow>=0);
    casadi_assert_dev(ncol>=0);
    // Column offset
    std::vector<s_t> colind(ncol+1);
    for (s_t cc=0; cc<ncol+1; ++cc) colind[cc] = cc*nrow;

    // Row
    std::vector<s_t> row(ncol*nrow);
    for (s_t cc=0; cc<ncol; ++cc)
      for (s_t rr=0; rr<nrow; ++rr)
        row[rr+cc*nrow] = rr;

    return Sparsity(nrow, ncol, colind, row);
  }

  Sparsity Sparsity::upper(s_t n) {
    casadi_assert(n>=0, "Sparsity::upper expects a positive integer as argument");
    s_t nrow=n, ncol=n;
    std::vector<s_t> colind, row;
    colind.reserve(ncol+1);
    row.reserve((n*(n+1))/2);

    // Loop over columns
    colind.push_back(0);
    for (s_t cc=0; cc<ncol; ++cc) {
      // Loop over rows for the upper triangular half
      for (s_t rr=0; rr<=cc; ++rr) {
        row.push_back(rr);
      }
      colind.push_back(row.size());
    }

    // Return the pattern
    return Sparsity(nrow, ncol, colind, row);
  }

  Sparsity Sparsity::lower(s_t n) {
    casadi_assert(n>=0, "Sparsity::lower expects a positive integer as argument");
    s_t nrow=n, ncol=n;
    std::vector<s_t> colind, row;
    colind.reserve(ncol+1);
    row.reserve((n*(n+1))/2);

    // Loop over columns
    colind.push_back(0);
    for (s_t cc=0; cc<ncol; ++cc) {
      // Loop over rows for the lower triangular half
      for (s_t rr=cc; rr<nrow; ++rr) {
        row.push_back(rr);
      }
      colind.push_back(row.size());
    }

    // Return the pattern
    return Sparsity(nrow, ncol, colind, row);
  }

  Sparsity Sparsity::band(s_t n, s_t p) {
    casadi_assert(n>=0, "Sparsity::band expects a positive integer as argument");
    casadi_assert((p<0? -p : p)<n,
                          "Sparsity::band: position of band schould be smaller then size argument");

    s_t nc = n-(p<0? -p : p);

    std::vector< s_t >          row(nc);

    s_t offset = max(p, 0);
    for (s_t i=0;i<nc;i++) {
      row[i]=i+offset;
    }

    std::vector< s_t >          colind(n+1);

    offset = min(p, 0);
    for (s_t i=0;i<n+1;i++) {
      colind[i]=max(min(i+offset, nc), 0);
    }

    return Sparsity(n, n, colind, row);

  }

  Sparsity Sparsity::banded(s_t n, s_t p) {
    // This is not an efficient implementation
    Sparsity ret = Sparsity(n, n);
    for (s_t i=-p;i<=p;++i) {
      ret = ret + Sparsity::band(n, i);
    }
    return ret;
  }

  Sparsity Sparsity::unit(s_t n, s_t el) {
    std::vector<s_t> row(1, el), colind(2);
    colind[0] = 0;
    colind[1] = 1;
    return Sparsity(n, 1, colind, row);
  }

  Sparsity Sparsity::rowcol(const std::vector<s_t>& row, const std::vector<s_t>& col,
                            s_t nrow, s_t ncol) {
    std::vector<s_t> all_rows, all_cols;
    all_rows.reserve(row.size()*col.size());
    all_cols.reserve(row.size()*col.size());
    for (std::vector<s_t>::const_iterator c_it=col.begin(); c_it!=col.end(); ++c_it) {
      casadi_assert(*c_it>=0 && *c_it<ncol, "Sparsity::rowcol: Column index out of bounds");
      for (std::vector<s_t>::const_iterator r_it=row.begin(); r_it!=row.end(); ++r_it) {
        casadi_assert(*r_it>=0 && *r_it<nrow, "Sparsity::rowcol: Row index out of bounds");
        all_rows.push_back(*r_it);
        all_cols.push_back(*c_it);
      }
    }
    return Sparsity::triplet(nrow, ncol, all_rows, all_cols);
  }

  Sparsity Sparsity::triplet(s_t nrow, s_t ncol, const std::vector<s_t>& row,
                             const std::vector<s_t>& col, std::vector<s_t>& mapping,
                             bool invert_mapping) {
    // Assert dimensions
    casadi_assert_dev(nrow>=0);
    casadi_assert_dev(ncol>=0);
    casadi_assert(col.size()==row.size(), "inconsistent lengths");

    // Create the return sparsity pattern and access vectors
    std::vector<s_t> r_colind(ncol+1, 0);
    std::vector<s_t> r_row;
    r_row.reserve(row.size());

    // Consistency check and check if elements are already perfectly ordered with no duplicates
    s_t last_col=-1, last_row=-1;
    bool perfectly_ordered=true;
    for (s_t k=0; k<col.size(); ++k) {
      // Consistency check
      casadi_assert(col[k]>=0 && col[k]<ncol, "Column index out of bounds");
      casadi_assert(row[k]>=0 && row[k]<nrow, "Row index out of bounds");

      // Check if ordering is already perfect
      perfectly_ordered = perfectly_ordered && (col[k]<last_col ||
                                                (col[k]==last_col && row[k]<=last_row));
      last_col = col[k];
      last_row = row[k];
    }

    // Quick return if perfectly ordered
    if (perfectly_ordered) {
      // Save rows
      r_row.resize(row.size());
      copy(row.begin(), row.end(), r_row.begin());

      // Find offset index
      s_t el=0;
      for (s_t i=0; i<ncol; ++i) {
        while (el<col.size() && col[el]==i) el++;
        r_colind[i+1] = el;
      }

      // Identity mapping
      mapping.resize(row.size());
      for (s_t k=0; k<row.size(); ++k) mapping[k] = k;

      // Quick return
      return Sparsity(nrow, ncol, r_colind, r_row);
    }

    // Reuse data
    std::vector<s_t>& mapping1 = invert_mapping ? r_row : mapping;
    std::vector<s_t>& mapping2 = invert_mapping ? mapping : r_row;

    // Make sure that enough memory is allocated to use as a work vector
    mapping1.reserve(std::max(nrow+1, static_cast<s_t>(col.size())));

    // Number of elements in each row
    std::vector<s_t>& rowcount = mapping1; // reuse memory
    rowcount.resize(nrow+1);
    fill(rowcount.begin(), rowcount.end(), 0);
    for (std::vector<s_t>::const_iterator it=row.begin(); it!=row.end(); ++it) {
      rowcount[*it+1]++;
    }

    // Cumsum to get index offset for each row
    for (s_t i=0; i<nrow; ++i) {
      rowcount[i+1] += rowcount[i];
    }

    // New row for each old row
    mapping2.resize(row.size());
    for (s_t k=0; k<row.size(); ++k) {
      mapping2[rowcount[row[k]]++] = k;
    }

    // Number of elements in each col
    std::vector<s_t>& colcount = r_colind; // reuse memory, r_colind is already the right size
                                           // and is filled with zeros
    for (std::vector<s_t>::const_iterator it=mapping2.begin(); it!=mapping2.end(); ++it) {
      colcount[col[*it]+1]++;
    }

    // Cumsum to get index offset for each col
    for (s_t i=0; i<ncol; ++i) {
      colcount[i+1] += colcount[i];
    }

    // New col for each old col
    mapping1.resize(col.size());
    for (std::vector<s_t>::const_iterator it=mapping2.begin(); it!=mapping2.end(); ++it) {
      mapping1[colcount[col[*it]]++] = *it;
    }

    // Current element in the return matrix
    s_t r_el = 0;
    r_row.resize(col.size());

    // Current nonzero
    std::vector<s_t>::const_iterator it=mapping1.begin();

    // Loop over columns
    r_colind[0] = 0;
    for (s_t i=0; i<ncol; ++i) {

      // Previous row (to detect duplicates)
      s_t j_prev = -1;

      // Loop over nonzero elements of the col
      while (it!=mapping1.end() && col[*it]==i) {

        // Get the element
        s_t el = *it;
        it++;

        // Get the row
        s_t j = row[el];

        // If not a duplicate, save to return matrix
        if (j!=j_prev)
          r_row[r_el++] = j;

        if (invert_mapping) {
          // Save to the inverse mapping
          mapping2[el] = r_el-1;
        } else {
          // If not a duplicate, save to the mapping vector
          if (j!=j_prev)
            mapping1[r_el-1] = el;
        }

        // Save row
        j_prev = j;
      }

      // Update col offset
      r_colind[i+1] = r_el;
    }

    // Resize the row vector
    r_row.resize(r_el);

    // Resize mapping matrix
    if (!invert_mapping) {
      mapping1.resize(r_el);
    }

    return Sparsity(nrow, ncol, r_colind, r_row);
  }

  Sparsity Sparsity::triplet(s_t nrow, s_t ncol, const std::vector<s_t>& row,
                             const std::vector<s_t>& col) {
    std::vector<s_t> mapping;
    return Sparsity::triplet(nrow, ncol, row, col, mapping, false);
  }

  Sparsity Sparsity::nonzeros(s_t nrow, s_t ncol, const std::vector<s_t>& nz, bool ind1) {
    casadi_assert(nrow>0, "nrow must be >0.");
    std::vector<s_t> row(nz.size());
    std::vector<s_t> col(nz.size());
    for (s_t i=0;i<nz.size();++i) {
      s_t k = nz[i];
      k-= ind1;
      row[i] = k % nrow;
      col[i] = k / nrow;
    }
    return triplet(nrow, ncol, row, col);
  }

  bool Sparsity::is_singular() const {
    casadi_assert(is_square(),
      "is_singular: only defined for square matrices, but got " + dim());
    return sprank(*this)!=size2();
  }

  std::vector<s_t> Sparsity::compress() const {
    return (*this)->sp();
  }

  Sparsity::operator const std::vector<s_t>&() const {
    return (*this)->sp();
  }

  Sparsity::operator SparsityStruct() const {
    const s_t* sp = *this;
    s_t nrow = sp[0], ncol = sp[1];
    const s_t* colind = sp+2, *row = sp+2+ncol+1;
    return SparsityStruct{nrow, ncol, colind, row};
  }

  Sparsity Sparsity::compressed(const std::vector<s_t>& v) {
    // Check consistency
    casadi_assert_dev(v.size() >= 2);
    s_t nrow = v[0];
    s_t ncol = v[1];
    casadi_assert_dev(v.size() >= 2 + ncol+1);
    s_t nnz = v[2 + ncol];
    bool dense = v.size() == 2 + ncol+1 && nrow*ncol==nnz;
    bool sparse = v.size() == 2 + ncol+1 + nnz;
    casadi_assert_dev(dense || sparse);

    // Call array version
    return compressed(&v.front());
  }

  Sparsity Sparsity::compressed(const s_t* v) {
    casadi_assert_dev(v!=0);

    // Get sparsity pattern
    s_t nrow = v[0];
    s_t ncol = v[1];
    const s_t *colind = v+2;
    if (colind[0]==1) {
      // Dense matrix
      return Sparsity::dense(nrow, ncol);
    }
    s_t nnz = colind[ncol];
    if (nrow*ncol == nnz) {
      // Dense matrix
      return Sparsity::dense(nrow, ncol);
    } else {
      // Sparse matrix
      const s_t *row = v + 2 + ncol+1;
      return Sparsity(nrow, ncol,
                      vector<s_t>(colind, colind+ncol+1),
                      vector<s_t>(row, row+nnz));
    }
  }

  s_t Sparsity::bw_upper() const {
    return (*this)->bw_upper();
  }

  s_t Sparsity::bw_lower() const {
    return (*this)->bw_lower();
  }

  Sparsity Sparsity::horzcat(const std::vector<Sparsity> & sp) {
    // Quick return if possible
    if (sp.empty()) return Sparsity(0, 0);
    if (sp.size()==1) return sp.front();

    // Count total nnz
    s_t nnz_total = 0;
    for (s_t i=0; i<sp.size(); ++i) nnz_total += sp[i].nnz();

    // Construct from vectors (triplet format)
    vector<s_t> ret_row, ret_col;
    ret_row.reserve(nnz_total);
    ret_col.reserve(nnz_total);
    s_t ret_ncol = 0;
    s_t ret_nrow = 0;
    for (s_t i=0; i<sp.size() && ret_nrow==0; ++i)
      ret_nrow = sp[i].size1();

    // Append all patterns
    for (vector<Sparsity>::const_iterator i=sp.begin(); i!=sp.end(); ++i) {
      // Get sparsity pattern
      s_t sp_nrow = i->size1();
      s_t sp_ncol = i->size2();
      const s_t* sp_colind = i->colind();
      const s_t* sp_row = i->row();
      casadi_assert(sp_nrow==ret_nrow || sp_nrow==0,
                            "Sparsity::horzcat: Mismatching number of rows");

      // Add entries to pattern
      for (s_t cc=0; cc<sp_ncol; ++cc) {
        for (s_t k=sp_colind[cc]; k<sp_colind[cc+1]; ++k) {
          ret_row.push_back(sp_row[k]);
          ret_col.push_back(cc + ret_ncol);
        }
      }

      // Update offset
      ret_ncol += sp_ncol;
    }
    return Sparsity::triplet(ret_nrow, ret_ncol, ret_row, ret_col);
  }

  Sparsity Sparsity::kron(const Sparsity& a, const Sparsity& b) {
    s_t a_ncol = a.size2();
    s_t b_ncol = b.size2();
    s_t a_nrow = a.size1();
    s_t b_nrow = b.size1();
    if (a.is_dense() && b.is_dense()) return Sparsity::dense(a_nrow*b_nrow, a_ncol*b_ncol);

    const s_t* a_colind = a.colind();
    const s_t* a_row = a.row();
    const s_t* b_colind = b.colind();
    const s_t* b_row = b.row();

    std::vector<s_t> r_colind(a_ncol*b_ncol+1, 0);
    std::vector<s_t> r_row(a.nnz()*b.nnz());

    s_t* r_colind_ptr = get_ptr(r_colind);
    s_t* r_row_ptr = get_ptr(r_row);

    s_t i=0;
    s_t j=0;
    // Loop over the columns
    for (s_t a_cc=0; a_cc<a_ncol; ++a_cc) {
      s_t a_start = a_colind[a_cc];
      s_t a_stop  = a_colind[a_cc+1];
      // Loop over the columns
      for (s_t b_cc=0; b_cc<b_ncol; ++b_cc) {
        s_t b_start = b_colind[b_cc];
        s_t b_stop  = b_colind[b_cc+1];
        // Loop over existing nonzeros
        for (s_t a_el=a_start; a_el<a_stop; ++a_el) {
          s_t a_r = a_row[a_el];
          // Loop over existing nonzeros
          for (s_t b_el=b_start; b_el<b_stop; ++b_el) {
            s_t b_r = b_row[b_el];
            r_row_ptr[i++] = a_r*b_nrow+b_r;
          }
        }
        j+=1;
        r_colind_ptr[j] = r_colind_ptr[j-1] + (b_stop-b_start)*(a_stop-a_start);
      }
    }
    return Sparsity(a_nrow*b_nrow, a_ncol*b_ncol, r_colind, r_row);
  }

  Sparsity Sparsity::vertcat(const std::vector<Sparsity> & sp) {
    // Quick return if possible
    if (sp.empty()) return Sparsity(0, 0);
    if (sp.size()==1) return sp.front();

    // Count total nnz
    s_t nnz_total = 0;
    for (s_t i=0; i<sp.size(); ++i) nnz_total += sp[i].nnz();

    // Construct from vectors (triplet format)
    vector<s_t> ret_row, ret_col;
    ret_row.reserve(nnz_total);
    ret_col.reserve(nnz_total);
    s_t ret_nrow = 0;
    s_t ret_ncol = 0;
    for (s_t i=0; i<sp.size() && ret_ncol==0; ++i)
      ret_ncol = sp[i].size2();

    // Append all patterns
    for (vector<Sparsity>::const_iterator i=sp.begin(); i!=sp.end(); ++i) {
      // Get sparsity pattern
      s_t sp_nrow = i->size1();
      s_t sp_ncol = i->size2();
      const s_t* sp_colind = i->colind();
      const s_t* sp_row = i->row();
      casadi_assert(sp_ncol==ret_ncol || sp_ncol==0,
                            "Sparsity::vertcat: Mismatching number of columns");

      // Add entries to pattern
      for (s_t cc=0; cc<sp_ncol; ++cc) {
        for (s_t k=sp_colind[cc]; k<sp_colind[cc+1]; ++k) {
          ret_row.push_back(sp_row[k] + ret_nrow);
          ret_col.push_back(cc);
        }
      }

      // Update offset
      ret_nrow += sp_nrow;
    }
    return Sparsity::triplet(ret_nrow, ret_ncol, ret_row, ret_col);
  }

  Sparsity Sparsity::diagcat(const std::vector< Sparsity > &v) {
    s_t n = 0;
    s_t m = 0;

    std::vector<s_t> colind(1, 0);
    std::vector<s_t> row;

    s_t nz = 0;
    for (s_t i=0;i<v.size();++i) {
      const s_t* colind_ = v[i].colind();
      s_t ncol = v[i].size2();
      const s_t* row_ = v[i].row();
      s_t sz = v[i].nnz();
      for (s_t k=1; k<ncol+1; ++k) {
        colind.push_back(colind_[k]+nz);
      }
      for (s_t k=0; k<sz; ++k) {
        row.push_back(row_[k]+m);
      }
      n+= v[i].size2();
      m+= v[i].size1();
      nz+= v[i].nnz();
    }

    return Sparsity(m, n, colind, row);
  }

  std::vector<Sparsity> Sparsity::horzsplit(const Sparsity& x, const std::vector<s_t>& offset) {
    // Consistency check
    casadi_assert_dev(offset.size()>=1);
    casadi_assert_dev(offset.front()==0);
    casadi_assert(offset.back()==x.size2(),
                          "horzsplit(Sparsity, std::vector<s_t>): Last elements of offset "
                          "(" + str(offset.back()) + ") must equal the number of columns "
                          "(" + str(x.size2()) + ")");
    casadi_assert_dev(is_monotone(offset));

    // Number of outputs
    s_t n = offset.size()-1;

    // Get the sparsity of the input
    const s_t* colind_x = x.colind();
    const s_t* row_x = x.row();

    // Allocate result
    std::vector<Sparsity> ret;
    ret.reserve(n);

    // Sparsity pattern as CCS vectors
    vector<s_t> colind, row;
    s_t ncol, nrow = x.size1();

    // Get the sparsity patterns of the outputs
    for (s_t i=0; i<n; ++i) {
      s_t first_col = offset[i];
      s_t last_col = offset[i+1];
      ncol = last_col - first_col;

      // Construct the sparsity pattern
      colind.resize(ncol+1);
      copy(colind_x+first_col, colind_x+last_col+1, colind.begin());
      for (vector<s_t>::iterator it=colind.begin()+1; it!=colind.end(); ++it) *it -= colind[0];
      colind[0] = 0;
      row.resize(colind.back());
      copy(row_x+colind_x[first_col], row_x+colind_x[last_col], row.begin());

      // Append to the list
      ret.push_back(Sparsity(nrow, ncol, colind, row));
    }

    // Return (RVO)
    return ret;
  }

  std::vector<Sparsity> Sparsity::vertsplit(const Sparsity& x, const std::vector<s_t>& offset) {
    std::vector<Sparsity> ret = horzsplit(x.T(), offset);
    for (std::vector<Sparsity>::iterator it=ret.begin(); it!=ret.end(); ++it) {
      *it = it->T();
    }
    return ret;
  }

  Sparsity Sparsity::blockcat(const std::vector< std::vector< Sparsity > > &v) {
    std::vector< Sparsity > ret;
    for (s_t i=0; i<v.size(); ++i)
      ret.push_back(horzcat(v[i]));
    return vertcat(ret);
  }

  std::vector<Sparsity> Sparsity::diagsplit(const Sparsity& x, const std::vector<s_t>& offset1,
                                            const std::vector<s_t>& offset2) {
    // Consistency check
    casadi_assert_dev(offset1.size()>=1);
    casadi_assert_dev(offset1.front()==0);
    casadi_assert(offset1.back()==x.size1(),
                          "diagsplit(Sparsity, offset1, offset2): Last elements of offset1 "
                          "(" + str(offset1.back()) + ") must equal the number of rows "
                          "(" + str(x.size1()) + ")");
    casadi_assert(offset2.back()==x.size2(),
                          "diagsplit(Sparsity, offset1, offset2): Last elements of offset2 "
                          "(" + str(offset2.back()) + ") must equal the number of rows "
                          "(" + str(x.size2()) + ")");
    casadi_assert_dev(is_monotone(offset1));
    casadi_assert_dev(is_monotone(offset2));
    casadi_assert_dev(offset1.size()==offset2.size());

    // Number of outputs
    s_t n = offset1.size()-1;

    // Return value
    std::vector<Sparsity> ret;

    // Caveat: this is a very silly implementation
    IM x2 = IM::zeros(x);

    for (s_t i=0; i<n; ++i) {
      ret.push_back(x2(Slice(offset1[i], offset1[i+1]),
                       Slice(offset2[i], offset2[i+1])).sparsity());
    }

    return ret;
  }

  s_t Sparsity::sprank(const Sparsity& x) {
    std::vector<s_t> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    x.btf(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);
    return coarse_colblock.at(3);
  }

  Sparsity::operator const s_t*() const {
    return &(*this)->sp().front();
  }

  s_t Sparsity::norm_0_mul(const Sparsity& x, const Sparsity& A) {
    // Implementation borrowed from Scipy's sparsetools/csr.h
    casadi_assert(A.size1()==x.size2(), "Dimension error. Got " + x.dim()
                          + " times " + A.dim() + ".");

    s_t n_row = A.size2();
    s_t n_col = x.size1();

    // Allocate work vectors
    std::vector<bool> Bwork(n_col);
    std::vector<s_t> Iwork(n_row+1+n_col);

    const s_t* Aj = A.row();
    const s_t* Ap = A.colind();
    const s_t* Bj = x.row();
    const s_t* Bp = x.colind();
    s_t *Cp = get_ptr(Iwork);
    s_t *mask = Cp+n_row+1;

    // Pass 1
    // method that uses O(n) temp storage
    std::fill(mask, mask+n_col, -1);

    Cp[0] = 0;
    s_t nnz = 0;
    for (s_t i = 0; i < n_row; i++) {
      s_t row_nnz = 0;
      for (s_t jj = Ap[i]; jj < Ap[i+1]; jj++) {
        s_t j = Aj[jj];
        for (s_t kk = Bp[j]; kk < Bp[j+1]; kk++) {
          s_t k = Bj[kk];
          if (mask[k] != i) {
            mask[k] = i;
            row_nnz++;
          }
        }
      }
      s_t next_nnz = nnz + row_nnz;
      nnz = next_nnz;
      Cp[i+1] = nnz;
    }

    // Pass 2
    s_t *next = get_ptr(Iwork) + n_row+1;
    std::fill(next, next+n_col, -1);
    std::vector<bool> & sums = Bwork;
    std::fill(sums.begin(), sums.end(), false);
    nnz = 0;
    Cp[0] = 0;
    for (s_t i = 0; i < n_row; i++) {
      s_t head   = -2;
      s_t length =  0;
      s_t jj_start = Ap[i];
      s_t jj_end   = Ap[i+1];
      for (s_t jj = jj_start; jj < jj_end; jj++) {
        s_t j = Aj[jj];
        s_t kk_start = Bp[j];
        s_t kk_end   = Bp[j+1];
        for (s_t kk = kk_start; kk < kk_end; kk++) {
          s_t k = Bj[kk];
          sums[k] = true;
          if (next[k] == -1) {
            next[k] = head;
            head  = k;
            length++;
          }
        }
      }
      for (s_t jj = 0; jj < length; jj++) {
        if (sums[head]) {
          nnz++;
        }
        s_t temp = head;
        head = next[head];
        next[temp] = -1; //clear arrays
        sums[temp] =  0;
      }
      Cp[i+1] = nnz;
    }
    return nnz;
  }

  void Sparsity::mul_sparsityF(const bvec_t* x, const Sparsity& x_sp,
                               const bvec_t* y, const Sparsity& y_sp,
                               bvec_t* z, const Sparsity& z_sp,
                               bvec_t* w) {
    // Assert dimensions
    casadi_assert(z_sp.size1()==x_sp.size1() && x_sp.size2()==y_sp.size1()
                          && y_sp.size2()==z_sp.size2(),
                          "Dimension error. Got x=" + x_sp.dim() + ", y=" + y_sp.dim()
                          + " and z=" + z_sp.dim() + ".");

    // Direct access to the arrays
    const s_t* y_colind = y_sp.colind();
    const s_t* y_row = y_sp.row();
    const s_t* x_colind = x_sp.colind();
    const s_t* x_row = x_sp.row();
    const s_t* z_colind = z_sp.colind();
    const s_t* z_row = z_sp.row();

    // Loop over the columns of y and z
    s_t ncol = z_sp.size2();
    for (s_t cc=0; cc<ncol; ++cc) {
      // Get the dense column of z
      for (s_t kk=z_colind[cc]; kk<z_colind[cc+1]; ++kk) {
        w[z_row[kk]] = z[kk];
      }

      // Loop over the nonzeros of y
      for (s_t kk=y_colind[cc]; kk<y_colind[cc+1]; ++kk) {
        s_t rr = y_row[kk];

        // Loop over corresponding columns of x
        bvec_t yy = y[kk];
        for (s_t kk1=x_colind[rr]; kk1<x_colind[rr+1]; ++kk1) {
          w[x_row[kk1]] |= x[kk1] | yy;
        }
      }

      // Get the sparse column of z
      for (s_t kk=z_colind[cc]; kk<z_colind[cc+1]; ++kk) {
        z[kk] = w[z_row[kk]];
      }
    }
  }

  void Sparsity::mul_sparsityR(bvec_t* x, const Sparsity& x_sp,
                               bvec_t* y, const Sparsity& y_sp,
                               bvec_t* z, const Sparsity& z_sp,
                               bvec_t* w) {
    // Assert dimensions
    casadi_assert(z_sp.size1()==x_sp.size1() && x_sp.size2()==y_sp.size1()
                          && y_sp.size2()==z_sp.size2(),
                          "Dimension error. Got x=" + x_sp.dim() + ", y=" + y_sp.dim()
                          + " and z=" + z_sp.dim() + ".");

    // Direct access to the arrays
    const s_t* y_colind = y_sp.colind();
    const s_t* y_row = y_sp.row();
    const s_t* x_colind = x_sp.colind();
    const s_t* x_row = x_sp.row();
    const s_t* z_colind = z_sp.colind();
    const s_t* z_row = z_sp.row();

    // Loop over the columns of y and z
    s_t ncol = z_sp.size2();
    for (s_t cc=0; cc<ncol; ++cc) {
      // Get the dense column of z
      for (s_t kk=z_colind[cc]; kk<z_colind[cc+1]; ++kk) {
        w[z_row[kk]] = z[kk];
      }

      // Loop over the nonzeros of y
      for (s_t kk=y_colind[cc]; kk<y_colind[cc+1]; ++kk) {
        s_t rr = y_row[kk];

        // Loop over corresponding columns of x
        bvec_t yy = 0;
        for (s_t kk1=x_colind[rr]; kk1<x_colind[rr+1]; ++kk1) {
          yy |= w[x_row[kk1]];
          x[kk1] |= w[x_row[kk1]];
        }
        y[kk] |= yy;
      }

      // Get the sparse column of z
      for (s_t kk=z_colind[cc]; kk<z_colind[cc+1]; ++kk) {
        z[kk] = w[z_row[kk]];
      }
    }
  }

  Dict Sparsity::info() const {
    if (is_null()) return Dict();
    return {{"nrow", size1()}, {"ncol", size2()}, {"colind", get_colind()}, {"row", get_row()}};
  }

  Sparsity Sparsity::from_info(const Dict& info) {
    auto it = info.find("nrow");
    if (it==info.end()) return Sparsity();
    s_t nrow = info.at("nrow");
    s_t ncol = info.at("ncol");
    std::vector<s_t> row, colind;
    if (info.at("row").is_int_vector()) {
      row = info.at("row");
    } else {
      row.push_back(info.at("row"));
    }
    if (info.at("colind").is_int_vector()) {
      colind = info.at("colind");
    } else {
      colind.push_back(info.at("colind"));
    }
    return Sparsity(nrow, ncol, colind, row);
  }
} // namespace casadi
