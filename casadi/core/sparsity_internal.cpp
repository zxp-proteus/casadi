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
#include "casadi_misc.hpp"
#include <climits>
#include <cstdlib>
#include <cmath>
#include "matrix.hpp"

using namespace std;

namespace casadi {
  void SparsityInternal::etree(const int* sp, int* parent, int *w, int ata) {
    /*
    Modified version of cs_etree in CSparse
    Copyright(c) Timothy A. Davis, 2006-2009
    Licensed as a derivative work under the GNU LGPL
    */
    int r, c, k, rnext;
    // Extract sparsity
    int nrow = *sp++, ncol = *sp++;
    const int *colind = sp, *row = sp+ncol+1;
    // Highest known ascestor of a node
    int *ancestor=w;
    // Path for A'A
    int *prev;
    if (ata) {
      prev=w+ncol;
      for (r=0; r<nrow; ++r) prev[r] = -1;
    }
    // Loop over columns
    for (c=0; c<ncol; ++c) {
      parent[c] = -1; // No parent yet
      ancestor[c] = -1; // No ancestor
      // Loop over nonzeros
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = row[k];
        if (ata) r = prev[r];
        // Traverse from r to c
        while (r!=-1 && r<c) {
          rnext = ancestor[r];
          ancestor[r] = c;
          if (rnext==-1) parent[r] = c;
          r = rnext;
        }
        if (ata) prev[row[k]] = c;
      }
    }
  }

  int SparsityInternal::postorder_dfs(int j, int k, int* head, int* next,
                                      int* post, int* stack) {
    /* Modified version of cs_tdfs in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    int i, p, top=0;
    stack[0] = j;
    while (top>=0) {
      p = stack[top];
      i = head[p];
      if (i==-1) {
        // No children
        top--;
        post[k++] = p;
      } else {
        // Add to stack
        head[p] = next[i];
        stack[++top] = i;
      }
    }
    return k;
  }

  void SparsityInternal::postorder(const int* parent, int n, int* post, int* w) {
    /* Modified version of cs_post in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    int j, k=0;
    // Work vectors
    int *head, *next, *stack;
    head=w; w+=n;
    next=w; w+=n;
    stack=w; w+=n;
    // Empty linked lists
    for (j=0; j<n; ++j) head[j] = -1;
    // Traverse nodes in reverse order
    for (j=n-1; j>=0; --j) {
      if (parent[j]!=-1) {
        next[j] = head[parent[j]];
        head[parent[j]] = j;
      }
    }
    for (j=0; j<n; j++) {
      if (parent[j]==-1) {
        k = postorder_dfs(j, k, head, next, post, stack);
      }
    }
  }

  int SparsityInternal::
  leaf(int i, int j, const int* first, int* maxfirst,
       int* prevleaf, int* ancestor, int* jleaf) {
    /* Modified version of cs_leaf in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    int q, s, sparent, jprev;
    *jleaf = 0;
    // Quick return if j is not a leaf
    if (i<=j || first[j]<=maxfirst[i]) return -1;
    // Update max first[j] seen so far
    maxfirst[i] = first[j];
    // Previous leaf of ith subtree
    jprev = prevleaf[i];
    prevleaf[i] = j;
    // j is first or subsequent leaf
    *jleaf = (jprev == -1) ? 1 : 2;
    // if first leaf, q is root of ith subtree
    if (*jleaf==1) return i;
    // Path compression
    for (q=jprev; q!=ancestor[q]; q=ancestor[q]) {}
    for (s=jprev; s!=q; s=sparent) {
      sparent = ancestor[s];
      ancestor[s] = q;
    }
    // Return least common ancestor
    return q;
  }

  int SparsityInternal::
  qr_counts(const int* tr_sp, const int* parent,
            const int* post, int* counts, int* w) {
    /* Modified version of cs_counts in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    int ncol = *tr_sp++, nrow = *tr_sp++;
    const int *rowind=tr_sp, *col=tr_sp+nrow+1;
    int i, j, k, J, p, q, jleaf, *maxfirst, *prevleaf,
      *ancestor, *head=0, *next=0, *first;
    // Work vectors
    ancestor=w; w+=ncol;
    maxfirst=w; w+=ncol;
    prevleaf=w; w+=ncol;
    first=w; w+=ncol;
    head=w; w+=ncol+1;
    next=w; w+=nrow;
    // Find first [j]
    for (k=0; k<ncol; ++k) first[k]=-1;
    for (k=0; k<ncol; ++k) {
      j=post[k];
      // counts[j]=1 if j is a leaf
      counts[j] = (first[j]==-1) ? 1 : 0;
      for (; j!=-1 && first[j]==-1; j=parent[j]) first[j]=k;
    }
    // Invert post (use ancestor as work vector)
    for (k=0; k<ncol; ++k) ancestor[post[k]] = k;
    for (k=0; k<ncol+1; ++k) head[k]=-1;
    for (i=0; i<nrow; ++i) {
      for (k=ncol, p=rowind[i]; p<rowind[i+1]; ++p) {
        k = std::min(k, ancestor[col[p]]);
      }
      // Place row i in linked list k
      next[i] = head[k];
      head[k] = i;
    }

    // Clear workspace
    for (k=0; k<ncol; ++k) maxfirst[k]=-1;
    for (k=0; k<ncol; ++k) prevleaf[k]=-1;
    // Each node in its own set
    for (i=0; i<ncol; ++i) ancestor[i]=i;
    for (k=0; k<ncol; ++k) {
      // j is the kth node in the postordered etree
      j=post[k];
      if (parent[j]!=-1) counts[parent[j]]--; // j is not a root
      J=head[k];
      while (J!=-1) { // J=j for LL' = A case
        for (p=rowind[J]; p<rowind[J+1]; ++p) {
          i=col[p];
          q = leaf(i, j, first, maxfirst, prevleaf, ancestor, &jleaf);
          if (jleaf>=1) counts[j]++; // A(i,j) is in skeleton
          if (jleaf==2) counts[q]--; // account for overlap in q
        }
        J = next[J];
      }
      if (parent[j]!=-1) ancestor[j]=parent[j];
    }
    // Sum up counts of each child
    for (j=0; j<ncol; ++j) {
      if (parent[j]!=-1) counts[parent[j]] += counts[j];
    }

    // Sum of counts
    int sum_counts = 0;
    for (j=0; j<ncol; ++j) sum_counts += counts[j];
    return sum_counts;
  }

  int SparsityInternal::
  qr_nnz(const int* sp, int* pinv, int* leftmost,
         const int* parent, int* nrow_ext, int* w) {
    /* Modified version of cs_sqr in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    // Extract sparsity
    int nrow = sp[0], ncol = sp[1];
    const int *colind=sp+2, *row=sp+2+ncol+1;
    // Work vectors
    int *next=w; w+=nrow;
    int *head=w; w+=ncol;
    int *tail=w; w+=ncol;
    int *nque=w; w+=ncol;
    // Local variables
    int r, c, k, pa;
    // Clear queue
    for (c=0; c<ncol; ++c) head[c] = -1;
    for (c=0; c<ncol; ++c) tail[c] = -1;
    for (c=0; c<ncol; ++c) nque[c] = 0;
    for (r=0; r<nrow; ++r) leftmost[r] = -1;
    // leftmost[r] = min(find(A(r,:)))
    for (c=ncol-1; c>=0; --c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        leftmost[row[k]] = c;
      }
    }
    // Scan rows in reverse order
    for (r=nrow-1; r>=0; --r) {
      pinv[r] = -1; // row r not yet ordered
      c=leftmost[r];

      if (c==-1) continue; // row r is empty
      if (nque[c]++ == 0) tail[c]=r; // first row in queue c
      next[r] = head[c]; // put r at head of queue c
      head[c] = r;
    }
    // Find row permutation and nnz(V)
    int v_nnz = 0;
    int nrow_new = nrow;
    for (c=0; c<ncol; ++c) {
      r = head[c]; // remove r from queue c
      v_nnz++; // count V(c,c) as nonzero
      if (r<0) r=nrow_new++; // add a fictitious row
      pinv[r] = c; // associate row r with V(:,c)
      if (--nque[c]<=0) continue; // skip if V(c+1,nrow,c) is empty
      v_nnz += nque[c]; // nque[c] is nnz(V(c+1:nrow, c))
      if ((pa=parent[c]) != -1) {
        // Move all rows to parent of c
        if (nque[pa]==0) tail[pa] = tail[c];
        next[tail[c]] = head[pa];
        head[pa] = next[r];
        nque[pa] += nque[c];
      }
    }
    for (r=0; r<nrow; ++r) if (pinv[r]<0) pinv[r] = c++;
    if (nrow_ext) *nrow_ext = nrow_new;
    return v_nnz;
  }

  void SparsityInternal::
  qr_init(const int* sp, const int* sp_tr,
          int* leftmost, int* parent, int* pinv,
          int* nrow_ext, int* v_nnz, int* r_nnz, int* w) {
    // Extract sparsity
    int ncol = sp[1];
    // Calculate elimination tree for A'A
    etree(sp, parent, w, 1); // len[w] >= nrow+ncol
    // Calculate postorder
    int* post = w; w += ncol;
    postorder(parent, ncol, post, w); // len[w] >= 3*ncol
    // Calculate nnz in R
    *r_nnz = qr_counts(sp_tr, parent, post, w, w+ncol);
    // Calculate nnz in V
    *v_nnz = qr_nnz(sp, pinv, leftmost, parent, nrow_ext, w);
  }

  void SparsityInternal::
  qr_sparsities(const int* sp_a, int nrow_ext, int* sp_v, int* sp_r,
                const int* leftmost, const int* parent, const int* pinv,
                int* iw) {
    /* Modified version of cs_qr in CSparse
      Copyright(c) Timothy A. Davis, 2006-2009
      Licensed as a derivative work under the GNU LGPL
    */
    // Extract sparsities
    int ncol = sp_a[1];
    const int *colind=sp_a+2, *row=sp_a+2+ncol+1;
    int *v_colind=sp_v+2, *v_row=sp_v+2+ncol+1;
    int *r_colind=sp_r+2, *r_row=sp_r+2+ncol+1;
    // Specify dimensions of V and R
    sp_v[0] = sp_r[0] = nrow_ext;
    sp_v[1] = sp_r[1] = ncol;
    // Work vectors
    int* s = iw; iw += ncol;
    // Local variables
    int r, c, k, k1, top, len, k2, r2;
    // Clear w to mark nodes
    for (r=0; r<nrow_ext; ++r) iw[r] = -1;
    // Number of nonzeros in v and r
    int nnz_r=0, nnz_v=0;
    // Compute V and R
    for (c=0; c<ncol; ++c) {
      // R(:,c) starts here
      r_colind[c] = nnz_r;
      // V(:, c) starts here
      v_colind[c] = k1 = nnz_v;
      // Add V(c,c) to pattern of V
      iw[c] = c;
      v_row[nnz_v++] = c;
      top = ncol;
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = leftmost[row[k]]; // r = min(find(A(r,:))
        // Traverse up c
        for (len=0; iw[r]!=c; r=parent[r]) {
          s[len++] = r;
          iw[r] = c;
        }
        while (len>0) s[--top] = s[--len]; // push path on stack
        r = pinv[row[k]]; // r = permuted row of A(:,c)
        if (r>c && iw[r]<c) {
          v_row[nnz_v++] = r; // add r to pattern of V(:,c)
          iw[r] = c;
        }
      }
      // For each r in pattern of R(:,c)
      for (k = top; k<ncol; ++k) {
        // R(r,c) is nonzero
        r = s[k];
        // Apply (V(r), beta(r)) to x: x -= v*beta*v'*x
        r_row[nnz_r++] = r;
        if (parent[r]==c) {
          for (k2=v_colind[r]; k2<v_colind[r+1]; ++k2) {
            r2 = v_row[k2];
            if (iw[r2]<c) {
              iw[r2] = c;
              v_row[nnz_v++] = r2;
            }
          }
        }
      }
      // R(c,c) = norm(x)
      r_row[nnz_r++] = c;
    }
    // Finalize R, V
    r_colind[ncol] = nnz_r;
    v_colind[ncol] = nnz_v;
  }

  SparsityInternal::
  SparsityInternal(int nrow, int ncol, const int* colind, const int* row) :
    sp_(2 + ncol+1 + colind[ncol]), btf_(0) {
    sp_[0] = nrow;
    sp_[1] = ncol;
    std::copy(colind, colind+ncol+1, sp_.begin()+2);
    std::copy(row, row+colind[ncol], sp_.begin()+2+ncol+1);
    sanity_check(false);
  }

  SparsityInternal::~SparsityInternal() {
    if (btf_) delete btf_;
  }

  const SparsityInternal::Btf& SparsityInternal::btf() const {
    if (!btf_) {
      btf_ = new SparsityInternal::Btf();
      btf_->nb = btf(btf_->rowperm, btf_->colperm, btf_->rowblock, btf_->colblock,
                     btf_->coarse_rowblock, btf_->coarse_colblock);
    }
    return *btf_;
  }


  int SparsityInternal::numel() const {
    return size1()*size2();
  }

  void SparsityInternal::sanity_check(bool complete) const {
    int nrow = size1();
    int ncol = size2();
    const int* colind = this->colind();
    const int* row = this->row();
    int nnz = this->nnz();
    casadi_assert(nrow >=0,
                          "number of rows must be positive, but got " + str(nrow) + ".");
    casadi_assert(ncol>=0 ,
                          "number of columns must be positive, but got " + str(ncol) + ".");
    if (complete) {

      for (int k=0; k<ncol; k++) {
        casadi_assert(colind[k+1]>=colind[k],
                              "Compressed Column Storage is not sane. "
                              "colind must be monotone.");
      }

      casadi_assert(colind[0]==0,
                            "Compressed Column Storage is not sane. "
                            "First element of colind must be zero.");

      for (int k=0; k<nnz; k++) {
        if (row[k]>=nrow || row[k] < 0) {
          casadi_error("Compressed Column Storage is not sane.\n"
            "The following must hold: 0 <= row[i] < nrow for each i, "
            "but got row[" + str(k) + "] = " + str(row[k])
            + " and nrow = " + str(nrow));
        }
      }
    }
  }

  void SparsityInternal::disp(ostream &stream, bool more) const {
    stream << dim(!is_dense());
    if (more) {
      stream << endl;
      stream << "colind: " << get_colind() << endl;
      stream << "row:    " << get_row() << endl;
    }
  }

  vector<int> SparsityInternal::get_col() const {
    const int* colind = this->colind();
    vector<int> col(nnz());
    for (int r=0; r<size2(); ++r) {
      for (int el = colind[r]; el < colind[r+1]; ++el) {
        col[el] = r;
      }
    }
    return col;
  }

  Sparsity SparsityInternal::T() const {
    // Dummy mapping
    vector<int> mapping;

    return transpose(mapping);
  }

  Sparsity SparsityInternal::transpose(vector<int>& mapping, bool invert_mapping) const {
    // Get the sparsity of the transpose in sparse triplet form
    vector<int> trans_col = get_row();
    vector<int> trans_row = get_col();

    // Create the sparsity pattern
    return Sparsity::triplet(size2(), size1(), trans_row, trans_col, mapping, invert_mapping);
  }

  int SparsityInternal::dfs(int j, int top, std::vector<int>& xi,
                                         std::vector<int>& pstack, const std::vector<int>& pinv,
                                         std::vector<bool>& marked) const {
    int head = 0;
    const int* colind = this->colind();
    const int* row = this->row();

    // initialize the recursion stack
    xi[0] = j;
    while (head >= 0) {

      // get j from the top of the recursion stack
      j = xi[head];
      int jnew = !pinv.empty() ? (pinv[j]) : j;
      if (!marked[j]) {

        // mark node j as visited
        marked[j]=true;
        pstack[head] = (jnew < 0) ? 0 : colind[jnew];
      }

      // node j done if no unvisited neighbors
      int done = 1;
      int p2 = (jnew < 0) ? 0 : colind[jnew+1];

      // examine all neighbors of j
      for (int p = pstack[head]; p< p2; ++p) {

        // consider neighbor node i
        int i = row[p];

        // skip visited node i
        if (marked[i]) continue ;

        // pause depth-first search of node j
        pstack[head] = p;

        // start dfs at node i
        xi[++head] = i;

        // node j is not done
        done = 0;

        // break, to start dfs (i)
        break;
      }

      //depth-first search at node j is done
      if (done) {
        // remove j from the recursion stack
        head--;

        // and place in the output stack
        xi[--top] = j ;
      }
    }
    return (top) ;
  }

  int SparsityInternal::scc(std::vector<int>& p,
                            std::vector<int>& r) const {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    vector<int> tmp;

    Sparsity AT = T();

    vector<int> xi(2*size2()+1);
    vector<int>& Blk = xi;

    vector<int> pstack(size2()+1);

    p.resize(size2());
    r.resize(size2()+6);

    vector<bool> marked(size2(), false);

    int top = size2();

    //first dfs(A) to find finish times (xi)
    for (int i = 0; i<size2(); ++i) {
      if (!marked[i]) {
        top = dfs(i, top, xi, pstack, tmp, marked);
      }
    }

    //restore A; unmark all nodes
    fill(marked.begin(), marked.end(), false);

    top = size2();
    int nb = size2();

    // dfs(A') to find strongly connnected comp
    for (int k=0 ; k < size2() ; ++k) {
      // get i in reverse order of finish times
      int i = xi[k];

      // skip node i if already ordered
      if (marked[i]) continue;

      // node i is the start of a component in p
      r[nb--] = top;
      top = AT.dfs(i, top, p, pstack, tmp, marked);
    }

    // first block starts at zero; shift r up
    r[nb] = 0;
    for (int k = nb ; k <= size2() ; ++k)
      r[k-nb] = r[k] ;

    // nb = # of strongly connected components
    nb = size2()-nb;

    // sort each block in natural order
    for (int b = 0 ; b < nb ; b++) {
      for (int k = r[b]; k<r[b+1] ; ++k)
        Blk[p[k]] = b ;
    }

    // Get p; shift r down (side effect)
    for (int i=0; i<size2(); ++i) {
      p[r[Blk[i]]++] = i;
    }

    // Shift up r
    r.resize(nb+1);
    for (int i=nb; i>0; --i) {
      r[i]=r[i-1];
    }
    r[0]=0;

    return nb;
  }

  std::vector<int> SparsityInternal::amd() const {
    /*
    Modified version of cs_amd in CSparse
    Copyright(c) Timothy A. Davis, 2006-2009
    Licensed as a derivative work under the GNU LGPL
    */
    casadi_assert(is_symmetric(), "AMD requires a symmetric matrix");
    // Get sparsity
    int n=size2();
    vector<int> colind = get_colind();
    vector<int> row = get_row();
    // Drop diagonal entries
    int nnz = 0; // number of nonzeros after pruning
    int col_begin, col_end=0;
    for (int c=0; c<n; ++c) {
      // Get the range of nonzeros for the column, before pruning
      col_begin = col_end;
      col_end = colind[c+1];
      // Loop over nonzeros
      for (int k=col_begin; k<col_end; ++k) {
        if (row[k]!=c) {
          row[nnz++] = row[k];
        }
      }
      colind[c+1] = nnz;
    }
    // dense threshold
    int dense = std::max(16, static_cast<int>(10*sqrt(static_cast<double>(n))));
    dense = std::min(n-2, dense);
    // Allocate result
    vector<int> P(n+1);
    // Work vectors
    vector<int> len(n+1), nv(n+1), next(n+1), head(n+1), elen(n+1), degree(n+1),
                w(n+1), hhead(n+1);
    // Number of elements
    int nel = 0;
    // Minimal degree
    int mindeg = 0;
    // Maximum length of w
    int lemax = 0;
    // Degree
    int d;
    // ?
    u_t h;
    // Flip
    #define FLIP(i) (-(i)-2)
    // Elbow room
    //int t = nnz + nnz/5 + 2*n;
    // Initialize quotient graph
    for (int k = 0; k<n; ++k) len[k] = colind[k+1] - colind[k];
    len[n] = 0;
    int nzmax = row.size();
    for (int i=0; i<=n; ++i) {
      head[i] = -1;                     // degree list i is empty
      P[i] = -1;
      next[i] = -1;
      hhead[i] = -1;                    // hash list i is empty
      nv[i] = 1;                        // node i is just one node
      w[i] = 1;                         // node i is alive
      elen[i] = 0;                      // Ek of node i is empty
      degree[i] = len[i];               // degree of node i
    }
    int mark = wclear(0, 0, get_ptr(w), n); // clear w
    elen[n] = -2;                           // n is a dead element
    colind[n] = -1;                         // n is a root of assembly tree
    w[n] = 0;                               // n is a dead element
    // Initialize degree lists
    for (int i = 0; i < n; ++i) {
      d = degree[i];
      if (d == 0) {                        // node i is empty
        elen[i] = -2;                      // element i is dead
        nel++;
        colind[i] = -1;                    // i is a root of assembly tree
        w[i] = 0;
      } else if (d > dense) {              // node i is dense
        nv[i] = 0;                         // absorb i into element n
        elen[i] = -1;                      // node i is dead
        nel++;
        colind[i] = FLIP(n);
        nv[n]++;
      } else {
        if (head[d] != -1) P[head[d]] = i;
        next[i] = head[d];                 // put node i in degree list d
        head[d] = i;
      }
    }
    while (nel < n) {                        // while (selecting pivots) do
      // Select node of minimum approximate degree
      int k;
      for (k = -1; mindeg < n && (k = head[mindeg]) == -1; mindeg++) {}
      if (next[k] != -1) P[next[k]] = -1;
      head[mindeg] = next[k];          // remove k from degree list
      int elenk = elen[k];             // elenk = |Ek|
      int nvk = nv[k];                     // # of nodes k represents
      nel += nvk;                      // nv[k] nodes of A eliminated
      // Garbage collection
      if (elenk > 0 && nnz + mindeg >= nzmax) {
        for (int j = 0; j < n; j++) {
          int p;
          if ((p = colind[j]) >= 0) {  // j is a live node or element
            colind[j] = row[p];        // save first entry of object
            row[p] = FLIP(j);          // first entry is now FLIP(j)
          }
        }
        int q, p;
        for (q = 0, p = 0; p < nnz; ) { // scan all of memory
          int j;
          if ((j = FLIP(row[p++])) >= 0) { // found object j
            row[q] = colind[j];         // restore first entry of object
            colind[j] = q++;            // new pointer to object j
            for (int k3 = 0; k3 < len[j]-1; k3++) row[q++] = row[p++];
          }
        }
        nnz = q;                        // row[nnz...nzmax-1] now free
      }
      // Construct new element
      int dk = 0;
      nv[k] = -nvk;                     // flag k as in Lk
      int p = colind[k];
      int pk1 = (elenk == 0) ? p : nnz;      // do in place if elen[k] == 0
      int pk2 = pk1;
      int e, pj, ln;
      for (int k1 = 1; k1 <= elenk + 1; k1++) {
        if (k1 > elenk) {
          e = k;                   // search the nodes in k
          pj = p;                  // list of nodes starts at row[pj]
          ln = len[k] - elenk;     // length of list of nodes in k
        } else {
          e = row[p++];            // search the nodes in e
          pj = colind[e];
          ln = len[e];             // length of list of nodes in e
        }
        for (int k2 = 1; k2 <= ln; k2++) {
          int i = row[pj++];
          int nvi;
          if ((nvi = nv[i]) <= 0) continue; // node i dead, or seen
          dk += nvi;                 // degree[Lk] += size of node i
          nv[i] = -nvi;              // negate nv[i] to denote i in Lk
          row[pk2++] = i;            // place i in Lk
          if (next[i] != -1) P[next[i]] = P[i];
          if (P[i] != -1) {          // remove i from degree list
            next[P[i]] = next[i];
          } else {
            head[degree[i]] = next[i];
          }
        }
        if (e != k) {
          colind[e] = FLIP(k);      // absorb e into k
          w[e] = 0;                  // e is now a dead element
        }
      }
      if (elenk != 0) nnz = pk2;        // row[nnz...nzmax] is free
      degree[k] = dk;                   // external degree of k - |Lk\i|
      colind[k] = pk1;                  // element k is in row[pk1..pk2-1]
      len[k] = pk2 - pk1;
      elen[k] = -2;                     // k is now an element
      // Find set differences
      mark = wclear(mark, lemax, get_ptr(w), n);  // clear w if necessary
      for (int pk = pk1; pk < pk2; pk++) {   // scan 1: find |Le\Lk|
        int i = row[pk];
        int eln;
        if ((eln = elen[i]) <= 0) continue; // skip if elen[i] empty
        int nvi = -nv[i];                   // nv[i] was negated
        int wnvi = mark - nvi;
        for (p = colind[i]; p <= colind[i] + eln - 1; p++) { // scan Ei
          e = row[p];
          if (w[e] >= mark) {
            w[e] -= nvi;          // decrement |Le\Lk|
          } else if (w[e] != 0) {   // ensure e is a live element
            w[e] = degree[e] + wnvi; /* 1st time e seen in scan 1 */
          }
        }
      }
      // Degree update
      for (int pk = pk1; pk < pk2; pk++) { // scan2: degree update
        int i = row[pk];                   // consider node i in Lk
        int p1 = colind[i];
        int p2 = p1 + elen[i] - 1;
        int pn = p1;
        for (h = 0, d = 0, p = p1; p <= p2; p++) { // scan Ei
          e = row[p];
          if (w[e] != 0) {         // e is an unabsorbed element
            int dext = w[e] - mark;    // dext = |Le\Lk|
            if (dext > 0) {
              d += dext;          // sum up the set differences
              row[pn++] = e;      // keep e in Ei
              h += e;             // compute the hash of node i
            } else {
              colind[e] = FLIP(k);  // aggressive absorb. e->k
              w[e] = 0;             // e is a dead element
            }
          }
        }
        elen[i] = pn - p1 + 1; // elen[i] = |Ei|
        int p3 = pn;
        int p4 = p1 + len[i];
        for (p = p2 + 1; p < p4; p++) { // prune edges in Ai
          int j = row[p];
          int nvj;
          if ((nvj = nv[j]) <= 0) continue; // node j dead or in Lk
          d += nvj;                  // degree(i) += |j|
          row[pn++] = j;             // place j in node list of i
          h += j;                    // compute hash for node i
        }
        if (d == 0) {                    // check for mass elimination
          colind[i] = FLIP(k);      // absorb i into k
          int nvi = -nv[i];
          dk -= nvi;                 // |Lk| -= |i|
          nvk += nvi;                // |k| += nv[i]
          nel += nvi;
          nv[i] = 0;
          elen[i] = -1;             // node i is dead
        } else {
          degree[i] = std::min(degree[i], d);   // update degree(i)
          row[pn] = row[p3];         // move first node to end
          row[p3] = row[p1];         // move 1st el. to end of Ei
          row[p1] = k;               // add k as 1st element in of Ei
          len[i] = pn - p1 + 1;     // new len of adj. list of node i
          h %= n;                    // finalize hash of i
          next[i] = hhead[h];      // place i in hash bucket
          hhead[h] = i;
          P[i] = h;              // save hash of i in P[i]
        }
      }                          // scan2 is done
      degree[k] = dk;          // finalize |Lk|
      lemax = std::max(lemax, dk);
      mark = wclear(mark+lemax, lemax, get_ptr(w), n);  // clear w
      // Supernode detection
      for (int pk = pk1; pk < pk2; pk++) {
        int i = row[pk];
        if (nv[i] >= 0) continue;      // skip if i is dead
        h = P[i];                      // scan hash bucket of node i
        i = hhead[h];
        hhead[h] = -1;                 // hash bucket will be empty
        for (; i != -1 && next[i] != -1; i = next[i], mark++) {
          ln = len[i];
          int eln = elen[i];
          for (p = colind[i]+1; p <= colind[i] + ln-1; p++) w[row[p]] = mark;
          int jlast = i;
          for (int j = next[i]; j != -1; ) { // compare i with all j
            int ok = (len[j] == ln) && (elen[j] == eln);
            for (p = colind[j] + 1; ok && p <= colind[j] + ln - 1; p++) {
              if (w[row[p]] != mark) ok = 0; // compare i and j
            }
            if (ok) {                    // i and j are identical
              colind[j] = FLIP(i);  // absorb j into i
              nv[i] += nv[j];
              nv[j] = 0;
              elen[j] = -1;         // node j is dead
              j = next[j];          // delete j from hash bucket
              next[jlast] = j;
            } else {
              jlast = j;             // j and i are different
              j = next[j];
            }
          }
        }
      }
      // Finalize new element
      int pk;
      for (p = pk1, pk = pk1; pk < pk2; pk++) {  // finalize Lk
        int i = row[pk];
        int nvi;
        if ((nvi = -nv[i]) <= 0) continue; // skip if i is dead
        nv[i] = nvi;                      // restore nv[i]
        d = degree[i] + dk - nvi;         // compute external degree(i)
        d = std::min(d, n - nel - nvi);
        if (head[d] != -1) P[head[d]] = i;
        next[i] = head[d];               // put i back in degree list
        P[i] = -1;
        head[d] = i;
        mindeg = std::min(mindeg, d);    // find new minimum degree
        degree[i] = d;
        row[p++] = i;                    // place i in Lk
      }
      nv[k] = nvk;                      // # nodes absorbed into k
      if ((len[k] = p-pk1) == 0) {      // length of adj list of element k
        colind[k] = -1;                 // k is a root of the tree
        w[k] = 0;                       // k is now a dead element
      }
      if (elenk != 0) nnz = p;          // free unused space in Lk
    }
    // Postordering
    for (int i = 0; i < n; i++) colind[i] = FLIP(colind[i]); // fix assembly tree
    for (int j = 0; j <= n; j++) head[j] = -1;
    for (int j = n; j >= 0; j--) {            // place unordered nodes in lists
      if (nv[j] > 0) continue;            // skip if j is an element
      next[j] = head[colind[j]];          // place j in list of its parent
      head[colind[j]] = j;
    }
    for (int e = n; e >= 0; e--) {            // place elements in lists
      if (nv[e] <= 0) continue;           // skip unless e is an element
      if (colind[e] != -1) {
        next[e] = head[colind[e]];        // place e in list of its parent
        head[colind[e]] = e;
      }
    }
    for (int k = 0, i = 0; i <= n; i++) {     // postorder the assembly tree
      if (colind[i] == -1) k = postorder_dfs(i, k, get_ptr(head), get_ptr(next),
                                             get_ptr(P), get_ptr(w));
    }
    P.resize(n);
    return P;
    #undef FLIP
  }

  void SparsityInternal::breadthFirstSearch(int n, std::vector<int>& wi, std::vector<int>& wj,
                                            std::vector<int>& queue, const std::vector<int>& imatch,
                                            const std::vector<int>& jmatch, int mark) const {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    int head = 0, tail = 0, j, i, p, j2 ;

    // place all unmatched nodes in queue
    for (j=0; j<n; ++j) {
      // skip j if matched
      if (imatch[j] >= 0) continue;

      // j in set C0 (R0 if transpose)
      wj[j] = 0;

      // place unmatched row j in queue
      queue[tail++] = j;
    }

    // quick return if no unmatched nodes
    if (tail == 0) return;

    Sparsity trans;
    const int *C_row, *C_colind;
    if (mark == 1) {
      C_row = row();
      C_colind = colind();
    } else {
      trans = T();
      C_row = trans.row();
      C_colind = trans.colind();
    }

    // while queue is not empty
    while (head < tail) {

      // get the head of the queue
      j = queue[head++];
      for (p = C_colind[j] ; p < C_colind[j+1] ; p++) {
        i = C_row[p] ;

        // skip if i is marked
        if (wi[i] >= 0) continue;

        // i in set R1 (C3 if transpose)
        wi[i] = mark;

        // traverse alternating path to j2
        j2 = jmatch[i];

        // skip j2 if it is marked
        if (wj[j2] >= 0) continue;

        // j2 in set C1 (R3 if transpose)
        wj[j2] = mark;

        // add j2 to queue
        queue[tail++] = j2;
      }
    }
  }

  void SparsityInternal::matched(int n, const std::vector<int>& wj,
                                 const std::vector<int>& imatch, std::vector<int>& p,
                                 std::vector<int>& q, std::vector<int>& cc, std::vector<int>& rr,
                                 int set, int mark) {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    int kc = cc[set];
    int kr = rr[set-1] ;
    for (int j=0; j<n; ++j) {
      // skip if j is not in C set
      if (wj[j] != mark) continue;

      p[kr++] = imatch[j] ;
      q[kc++] = j ;
    }

    cc[set+1] = kc ;
    rr[set] = kr ;
  }

  void SparsityInternal::unmatched(int m, const std::vector<int>& wi, std::vector<int>& p,
                                   std::vector<int>& rr, int set) {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    int i, kr = rr[set] ;
    for (i=0; i<m; i++)
      if (wi[i] == 0)
        p[kr++] = i;

    rr[set+1] = kr;
  }

  int SparsityInternal::rprune(int i, int j, double aij, void *other) {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    vector<int> &rr = *static_cast<vector<int> *>(other);
    return (i >= rr[1] && i < rr[2]) ;
  }

  void SparsityInternal::augmentingPath(int k, std::vector<int>& jmatch, int *cheap,
                                        std::vector<int>& w, int *js, int *is, int *ps) const {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    const int* colind = this->colind();
    const int* row = this->row();

    int found = 0, p, i = -1, head = 0, j ;

    // start with just node k in jstack
    js[0] = k ;

    while (head >= 0) {
      // --- Start (or continue) depth-first-search at node j -------------

      // get j from top of jstack
      j = js[head];

      // 1st time j visited for kth path
      if (w[j] != k) {

        // mark j as visited for kth path
        w[j] = k;
        for (p = cheap[j] ; p < colind[j+1] && !found; ++p) {
          i = row[p] ;            /* try a cheap assignment (i, j) */
          found = (jmatch[i] == -1) ;
        }

        // start here next time j is traversed
        cheap[j] = p;
        if (found) {
          // row j matched with col i
          is[head] = i;

          // end of augmenting path
          break;
        }

        // no cheap match: start dfs for j
        ps[head] = colind[j];
      }

      // --- Depth-first-search of neighbors of j -------------------------
      for (p = ps[head]; p<colind[j+1]; ++p) {

        // consider col i
        i = row[p];

        // skip jmatch[i] if marked
        if (w[jmatch[i]] == k) continue;

        // pause dfs of node j
        ps[head] = p + 1;

        // i will be matched with j if found
        is[head] = i;

        // start dfs at row jmatch[i]
        js[++head] = jmatch[i];
        break ;
      }

      // node j is done; pop from stack
      if (p == colind[j+1]) head--;
    } // augment the match if path found:

    if (found)
      for (p = head; p>=0; --p)
        jmatch[is[p]] = js[p];
  }

  void SparsityInternal::maxTransversal(std::vector<int>& imatch, std::vector<int>& jmatch,
                                        Sparsity& trans, int seed) const {
    // NOTE: This implementation has been copied from CSparse and then modified,
    // it needs cleaning up to be proper C++
    const int* colind = this->colind();
    const int* row = this->row();

    int n2 = 0, m2 = 0;

    // allocate result
    jmatch.resize(size1());
    imatch.resize(size2());
    vector<int> w(size1()+size2());

    // count nonempty columns and rows
    int k=0;
    for (int j=0; j<size2(); ++j) {
      n2 += (colind[j] < colind[j+1]);
      for (int p=colind[j]; p < colind[j+1]; ++p) {
        w[row[p]] = 1;

        // count entries already on diagonal
        k += (j == row[p]);
      }
    }

    // quick return if diagonal zero-free
    if (k == std::min(size1(), size2())) {
      int i;
      for (i=0; i<k; ++i) jmatch[i] = i;
      for (;    i<size1(); ++i) jmatch[i] = -1;

      int j;
      for (j=0; j<k; ++j) imatch[j] = j;
      for (;    j<size2(); ++j) imatch[j] = -1;
    }

    for (int i=0; i<size1(); ++i) m2 += w[i];

    // transpose if needed
    if (m2 < n2 && trans.is_null())
      trans = T();

    // Get pointer to sparsity
    const SparsityInternal* C = m2 < n2 ? static_cast<const SparsityInternal*>(trans.get()) : this;
    const int* C_colind = C->colind();

    std::vector<int>& Cjmatch = m2 < n2 ? imatch : jmatch;
    std::vector<int>& Cimatch = m2 < n2 ? jmatch : imatch;

    // get workspace
    w.resize(5 * C->size2());

    int *cheap = &w.front() + C->size2();
    int *js = &w.front() + 2*C->size2();
    int *is = &w.front() + 3*C->size2();
    int *ps = &w.front() + 4*C->size2();

    // for cheap assignment
    for (int j=0; j<C->size2(); ++j)
      cheap[j] = C_colind[j];

    // all rows unflagged
    for (int j=0; j<C->size2(); ++j)
      w[j] = -1;

    // nothing matched yet
    for (int i=0; i<C->size1(); ++i)
      Cjmatch[i] = -1;

    // q = random permutation
    std::vector<int> q = randomPermutation(C->size2(), seed);

    // augment, starting at row q[k]
    for (k=0; k<C->size2(); ++k) {
      C->augmentingPath(!q.empty() ? q[k]: k, Cjmatch, cheap, w, js, is, ps);
    }

    // find col match
    for (int j=0; j<C->size2(); ++j)
      Cimatch[j] = -1;

    for (int i = 0; i<C->size1(); ++i)
      if (Cjmatch[i] >= 0)
        Cimatch[Cjmatch[i]] = i;
  }

  void SparsityInternal::dmperm(std::vector<int>& rowperm,
                                std::vector<int>& colperm,
                                std::vector<int>& rowblock,
                                std::vector<int>& colblock,
                                std::vector<int>& coarse_rowblock,
                                std::vector<int>& coarse_colblock) const {
    int seed = 0;

    // The transpose of the expression
    Sparsity trans;

    // Part 1: Maximum matching

    // col permutation
    rowperm.resize(size1());

    // row permutation
    colperm.resize(size2());

    // size nb+1, block k is columns r[k] to r[k+1]-1 in A(p, q)
    rowblock.resize(size1()+6);

    // size nb+1, block k is rows s[k] to s[k+1]-1 in A(p, q)
    colblock.resize(size2()+6);

    // coarse col decomposition
    coarse_rowblock.resize(5);
    fill(coarse_rowblock.begin(), coarse_rowblock.end(), 0);

    // coarse row decomposition
    coarse_colblock.resize(5);
    fill(coarse_colblock.begin(), coarse_colblock.end(), 0);

    // max transversal
    vector<int> imatch, jmatch;
    maxTransversal(imatch, jmatch, trans, seed);

    // Coarse decomposition

    // use rowblock and colblock as workspace
    vector<int>& wi = rowblock;
    vector<int>& wj = colblock;

    // unmark all rows for bfs
    for (int j=0; j<size2(); ++j)
      wj[j] = -1;

    // unmark all columns for bfs
    for (int i=0; i<size1(); ++i)
      wi[i] = -1 ;

    // find C1, R1 from C0
    breadthFirstSearch(size2(), wi, wj, colperm, imatch, jmatch, 1);

    // find R3, C3 from R0
    breadthFirstSearch(size1(), wj, wi, rowperm, jmatch, imatch, 3);

    // unmatched set C0
    unmatched(size2(), wj, colperm, coarse_colblock, 0);

    // set R1 and C1
    matched(size2(), wj, imatch, rowperm, colperm, coarse_colblock, coarse_rowblock, 1, 1);

    // set R2 and C2
    matched(size2(), wj, imatch, rowperm, colperm, coarse_colblock, coarse_rowblock, 2, -1);

    // set R3 and C3
    matched(size2(), wj, imatch, rowperm, colperm, coarse_colblock, coarse_rowblock, 3, 3);

    // unmatched set R0
    unmatched(size1(), wi, rowperm, coarse_rowblock, 3);

    // --- Fine decomposition -----------------------------------------------
    // pinv=p'
    vector<int> pinv = invertPermutation(rowperm);

    // C=A(p, q) (it will hold A(R2, C2))
    std::vector<int> colind_C, row_C;
    permute(pinv, colperm, 0, colind_C, row_C);

    // delete rows C0, C1, and C3 from C
    int nc = coarse_colblock[3] - coarse_colblock[2];
    if (coarse_colblock[2] > 0) {
      for (int j = coarse_colblock[2]; j <= coarse_colblock[3]; ++j)
        colind_C[j-coarse_colblock[2]] = colind_C[j];
    }
    int ncol_C = nc;

    colind_C.resize(nc+1);
    // delete columns R0, R1, and R3 from C
    if (coarse_rowblock[2] - coarse_rowblock[1] < size1()) {
      drop(rprune, &coarse_rowblock, size1(), ncol_C, colind_C, row_C);
      int cnz = colind_C[nc];
      if (coarse_rowblock[1] > 0)
        for (int k=0; k<cnz; ++k)
          row_C[k] -= coarse_rowblock[1];
    }
    row_C.resize(colind_C.back());
    int nrow_C = nc ;
    Sparsity C(nrow_C, ncol_C, colind_C, row_C);

    // find strongly connected components of C
    vector<int> scc_p, scc_r;
    int scc_nb = C.scc(scc_p, scc_r);

    // --- Combine coarse and fine decompositions ---------------------------

    // C(ps, ps) is the permuted matrix
    vector<int> ps = scc_p;

    // kth block is rs[k]..rs[k+1]-1
    vector<int> rs = scc_r;

    // # of blocks of A(R2, C2)
    int nb1 = scc_nb;

    for (int k=0; k<nc; ++k)
      wj[k] = colperm[ps[k] + coarse_colblock[2]];

    for (int k=0; k<nc; ++k)
      colperm[k + coarse_colblock[2]] = wj[k];

    for (int k=0; k<nc; ++k)
      wi[k] = rowperm[ps[k] + coarse_rowblock[1]];

    for (int k=0; k<nc; ++k)
      rowperm[k + coarse_rowblock[1]] = wi[k];

    // create the fine block partitions
    int nb2 = 0;
    rowblock[0] = colblock[0] = 0;

    // leading coarse block A (R1, [C0 C1])
    if (coarse_colblock[2] > 0)
      nb2++ ;

    // coarse block A (R2, C2)
    for (int k=0; k<nb1; ++k) {
      // A (R2, C2) splits into nb1 fine blocks
      rowblock[nb2] = rs[k] + coarse_rowblock[1];
      colblock[nb2] = rs[k] + coarse_colblock[2] ;
      nb2++ ;
    }

    if (coarse_rowblock[2] < size1()) {
      // trailing coarse block A ([R3 R0], C3)
      rowblock[nb2] = coarse_rowblock[2];
      colblock[nb2] = coarse_colblock[3];
      nb2++ ;
    }

    rowblock[nb2] = size1();
    colblock[nb2] = size2() ;

    // Shrink rowblock and colblock
    rowblock.resize(nb2+1);
    colblock.resize(nb2+1);
  }

  std::vector<int> SparsityInternal::randomPermutation(int n, int seed) {
    // Return object
    std::vector<int> p;

    // return p = empty (identity)
    if (seed==0) return p;

    // allocate result
    p.resize(n);

    for (int k=0; k<n; ++k)
      p[k] = n-k-1;

    // return reverse permutation
    if (seed==-1) return p;
    #if defined(_WIN32)
    srand(seed);
    #else
    u_t seedu = seed;
    #endif

    for (int k=0; k<n; ++k) {
      // j = rand int in range k to n-1
      #if defined(_WIN32)
      int j = k + (rand() % (n-k)); // NOLINT(runtime/threadsafe_fn)
      #else
      int j = k + (rand_r(&seedu) % (n-k));
      #endif
      // swap p[k] and p[j]
      int t = p[j];
      p[j] = p[k];
      p[k] = t;
    }

    return p;
  }

  std::vector<int> SparsityInternal::invertPermutation(const std::vector<int>& p) {
    // pinv = p', or p = pinv'

    // allocate result
    vector<int> pinv(p.size());

    // invert the permutation
    for (int k=0; k<p.size(); ++k)
      pinv[p[k]] = k;

    // return result
    return pinv;
  }

  Sparsity SparsityInternal::permute(const std::vector<int>& pinv,
                                     const std::vector<int>& q, int values) const {
    std::vector<int> colind_C, row_C;
    permute(pinv, q, values, colind_C, row_C);
    return Sparsity(size1(), size2(), colind_C, row_C);
  }

  void SparsityInternal::permute(const std::vector<int>& pinv,
                                 const std::vector<int>& q, int values,
                                 std::vector<int>& colind_C,
                                 std::vector<int>& row_C) const {
    const int* colind = this->colind();
    const int* row = this->row();

    // alloc column offsets
    colind_C.resize(size2()+1);

    // Row for each nonzero
    row_C.resize(nnz());
    int nz = 0;
    for (int k = 0; k<size2(); ++k) {
      // row k of C is row q[k] of A
      colind_C[k] = nz;

      int j = !q.empty() ? (q[k]) : k;

      for (int t = colind[j]; t<colind[j+1]; ++t) {
        row_C[nz++] = !pinv.empty() ? (pinv[row[t]]) : row[t] ;
      }
    }

    // finalize the last row of C
    colind_C[size2()] = nz;
  }

  int SparsityInternal::drop(int (*fkeep)(int, int, double, void *),
                             void *other, int nrow, int ncol,
                             std::vector<int>& colind, std::vector<int>& row) {
    int nz = 0;

    for (int j = 0; j<ncol; ++j) {
      // get current location of row j
      int p = colind[j];

      // record new location of row j
      colind[j] = nz;
      for ( ; p < colind[j+1] ; ++p) {
        if (fkeep(row[p], j, 1, other)) {
          // keep A(i, j)
          row[nz++] = row[p] ;
        }
      }
    }

    // finalize A
    colind[ncol] = nz;
    return nz ;
  }

  int SparsityInternal::wclear(int mark, int lemax, int *w, int n) {
    if (mark < 2 || (mark + lemax < 0)) {
      for (int k = 0; k<n; ++k) if (w[k] != 0) w[k] = 1;
      mark = 2 ;
    }
    // at this point, w [0..n-1] < mark holds
    return mark;
  }

  int SparsityInternal::diag(int i, int j, double aij, void *other) {
    return (i != j) ;
  }

  int SparsityInternal::scatter(int j, std::vector<int>& w, int mark, int* Ci, int nz) const {
    int i, p;
    const int *Ap = colind();
    const int *Ai = row();

    for (p = Ap[j]; p<Ap[j+1]; ++p) {
      // A(i, j) is nonzero
      i = Ai[p];

      if (w[i] < mark) {
        // i is new entry in row j
        w[i] = mark;

        // add i to pattern of C(:, j)
        Ci[nz++] = i;
      }
    }
    return nz;
  }

  Sparsity SparsityInternal::multiply(const Sparsity& B) const {
    int nz = 0;
    casadi_assert(size2() == B.size1(), "Dimension mismatch.");
    int m = size1();
    int anz = nnz();
    int n = B.size2();
    const int* Bp = B.colind();
    const int* Bi = B.row();
    int bnz = Bp[n];

    // get workspace
    vector<int> w(m);

    // allocate result
    vector<int> C_colind(n+1, 0), C_row;

    C_colind.resize(anz + bnz);

    int* Cp = &C_colind.front();
    for (int j=0; j<n; ++j) {
      if (nz+m > C_row.size()) {
        C_row.resize(2*(C_row.size())+m);
      }

      // row j of C starts here
      Cp[j] = nz;
      for (int p = Bp[j] ; p<Bp[j+1] ; ++p) {
        nz = scatter(Bi[p], w, j+1, &C_row.front(), nz);
      }
    }

    // finalize the last row of C
    Cp[n] = nz;
    C_row.resize(nz);

    // Success
    return Sparsity(m, n, C_colind, C_row);
  }

  Sparsity SparsityInternal::get_diag(std::vector<int>& mapping) const {
    int nrow = this->size1();
    int ncol = this->size2();
    const int* colind = this->colind();
    const int* row = this->row();

    // Mapping
    mapping.clear();

    if (is_vector()) {
      // Sparsity pattern
      int n = nrow * ncol;
      vector<int> ret_colind(n+1, 0), ret_row;

      // Loop over all entries
      int ret_i=0;
      for (int cc=0; cc<ncol; ++cc) {
        for (int k = colind[cc]; k<colind[cc+1]; ++k) {
          int rr=row[k];
          int el=rr+nrow*cc; // Corresponding row in the return matrix
          while (ret_i<=el) ret_colind[ret_i++]=ret_row.size();
          ret_row.push_back(el);
          mapping.push_back(k);
        }
      }
      while (ret_i<=n) ret_colind[ret_i++]=ret_row.size();

      // Construct sparsity pattern
      return Sparsity(n, n, ret_colind, ret_row);

    } else {
      // Sparsity pattern
      int n = std::min(nrow, ncol);
      vector<int> ret_row, ret_colind(2, 0);

      // Loop over diagonal nonzeros
      for (int cc=0; cc<n; ++cc) {
        for (int el = colind[cc]; el<colind[cc+1]; ++el) {
          if (row[el]==cc) {
            ret_row.push_back(row[el]);
            ret_colind[1]++;
            mapping.push_back(el);
          }
        }
      }

      // Construct sparsity pattern
      return Sparsity(n, 1, ret_colind, ret_row);
    }
  }

  bool SparsityInternal::has_diag() const {
    int nrow = this->size1();
    int ncol = this->size2();
    const int* colind = this->colind();
    const int* row = this->row();
    for (int c=0; c<ncol && c<nrow; ++c) {
      for (int k=colind[c]; k<colind[c+1]; ++k) {
        if (row[k]==c) return true;
      }
    }
    return false;
  }

  Sparsity SparsityInternal::drop_diag() const {
    int nrow = this->size1();
    int ncol = this->size2();
    const int* colind = this->colind();
    const int* row = this->row();
    // Return sparsity
    vector<int> ret_colind(ncol+1), ret_row;
    ret_colind[0] = 0;
    ret_row.reserve(nnz());
    for (int c=0; c<ncol; ++c) {
      for (int k=colind[c]; k<colind[c+1]; ++k) {
        if (row[k]!=c) {
          ret_row.push_back(row[k]);
        }
      }
      ret_colind[c+1] = ret_row.size();
    }
    return Sparsity(nrow, ncol, ret_colind, ret_row);
  }

  std::string SparsityInternal::dim(bool with_nz) const {
    std::string ret = str(size1()) + "x" + str(size2());
    if (with_nz) ret += "," + str(nnz()) + "nz";
    return ret;
  }

  std::string SparsityInternal::repr_el(int k) const {
    std::stringstream ss;
    if (numel()!=nnz()) {
      ss << "nonzero index " << k << " ";
    }
    int r = row()[k];
    int c = get_col()[k];
    ss << "(row " << r << ", col " << c << ")";

    return ss.str();
  }

  Sparsity SparsityInternal::_mtimes(const Sparsity& y) const {
    // Dimensions of the result
    int d1 = size1();
    int d2 = y.size2();

    // Elementwise multiplication if one factor is scalar
    if (is_scalar(false)) {
      return is_dense() ? y : Sparsity(y.size());
    } else if (y.is_scalar(false)) {
      return y.is_dense() ? shared_from_this<Sparsity>() : Sparsity(size());
    }

    // Quick return if both are dense
    if (is_dense() && y.is_dense()) {
      return !is_empty() && !y.is_empty() ? Sparsity::dense(d1, d2) :
        Sparsity(d1, d2);
    }

    // Quick return if first factor is diagonal
    if (is_diag()) return y;

    // Quick return if second factor is diagonal
    if (y.is_diag()) return shared_from_this<Sparsity>();

    // Direct access to the vectors
    const int* x_row = row();
    const int* x_colind = colind();
    const int* y_row = y.row();
    const int* y_colind = y.colind();

    // Sparsity pattern of the result
    vector<int> row, col;

    // Temporary vector for avoiding duplicate nonzeros
    vector<int> tmp(d1, -1);

    // Loop over the nonzeros of y
    for (int cc=0; cc<d2; ++cc) {
      for (int kk=y_colind[cc]; kk<y_colind[cc+1]; ++kk) {
        int rr = y_row[kk];

        // Loop over corresponding columns of x
        for (int kk1=x_colind[rr]; kk1<x_colind[rr+1]; ++kk1) {
          int rr1 = x_row[kk1];

          // Add to pattern if not already encountered
          if (tmp[rr1]!=cc) {
            tmp[rr1] = cc;
            row.push_back(rr1);
            col.push_back(cc);
          }
        }
      }
    }

    // Assemble sparsity pattern and return
    return Sparsity::triplet(d1, d2, row, col);
  }

  bool SparsityInternal::is_scalar(bool scalar_and_dense) const {
    return size2()==1 && size1()==1 && (!scalar_and_dense || nnz()==1);
  }

  bool SparsityInternal::is_dense() const {
    return nnz() == numel();
  }

  bool SparsityInternal::is_row() const {
    return size1()==1;
  }

  bool SparsityInternal::is_column() const {
    return size2()==1;
  }

  bool SparsityInternal::is_vector() const {
    return is_row() || is_column();
  }

  bool SparsityInternal::is_empty(bool both) const {
    return both ? size2()==0 && size1()==0 : size2()==0 || size1()==0;
  }

  bool SparsityInternal::is_diag() const {
    const int* colind = this->colind();
    const int* row = this->row();

    // Check if matrix is square
    if (size2() != size1()) return false;

    // Check if correct number of non-zeros (one per column)
    if (nnz() != size2()) return false;

    // Check that the row indices are correct
    for (int i=0; i<nnz(); ++i) {
      if (row[i]!=i)
        return false;
    }

    // Make sure that the col indices are correct
    for (int i=0; i<size2(); ++i) {
      if (colind[i]!=i)
        return false;
    }

    // Diagonal if reached this point
    return true;
  }

  bool SparsityInternal::is_square() const {
    return size2() == size1();
  }

  bool SparsityInternal::is_symmetric() const {
    return is_transpose(*this);
  }

  int SparsityInternal::nnz_lower(bool strictly) const {
    const int* colind = this->colind();
    const int* row = this->row();
    int nnz = 0;
    for (int cc=0; cc<size2(); ++cc) {
      for (int el = colind[cc]; el<colind[cc+1]; ++el) {
        if (cc<row[el] || (!strictly && cc==row[el])) nnz++;
      }
    }
    return nnz;
  }

  int SparsityInternal::nnz_diag() const {
    const int* colind = this->colind();
    const int* row = this->row();
    int nnz = 0;
    for (int cc=0; cc<size2(); ++cc) {
      for (int el = colind[cc]; el < colind[cc+1]; ++el) {
        nnz += row[el]==cc;
      }
    }
    return nnz;
  }

  int SparsityInternal::nnz_upper(bool strictly) const {
    const int* colind = this->colind();
    const int* row = this->row();
    int nnz = 0;
    for (int cc=0; cc<size2(); ++cc) {
      for (int el = colind[cc]; el<colind[cc+1]; ++el) {
        if (cc>row[el] || (!strictly && cc==row[el])) nnz++;
      }
    }
    return nnz;
  }

  std::pair<int, int> SparsityInternal::size() const {
    return std::pair<int, int>(size1(), size2());
  }

  Sparsity SparsityInternal::_erase(const vector<int>& rr, bool ind1,
                                      std::vector<int>& mapping) const {
    // Quick return if nothing to erase
    if (rr.empty()) {
      mapping = range(nnz());
      return shared_from_this<Sparsity>();
    }
    casadi_assert_in_range(rr, -numel()+ind1, numel()+ind1);

    // Handle index-1, negative indices
    if (ind1 || has_negative(rr)) {
      std::vector<int> rr_mod = rr;
      for (vector<int>::iterator i=rr_mod.begin(); i!=rr_mod.end(); ++i) {
        if (ind1) (*i)--;
        if (*i<0) *i += numel();
      }
      return _erase(rr_mod, false, mapping); // Call recursively
    }

    // Sort rr in non-deceasing order, if needed
    if (!is_nondecreasing(rr)) {
      std::vector<int> rr_sorted = rr;
      std::sort(rr_sorted.begin(), rr_sorted.end());
      return _erase(rr_sorted, false, mapping);
    }

    // Mapping
    mapping.resize(0);

    // Quick return if no elements
    if (numel()==0) return shared_from_this<Sparsity>();

    // Reserve memory
    mapping.reserve(nnz());

    // Number of non-zeros
    int nz=0;

    // Elements to be erased
    vector<int>::const_iterator next_rr = rr.begin();

    // Return value
    vector<int> ret_colind = get_colind(), ret_row = get_row();

    // First and last index for the column (note colind_ is being overwritten)
    int k_first, k_last=0;

    // Loop over columns
    for (int j=0; j<size2(); ++j) {
      // Update k range
      k_first = k_last;
      k_last = ret_colind[j+1];

      // Loop over nonzeros
      for (int k=k_first; k<k_last; ++k) {
        // Get row
        int i=ret_row[k];

        // Corresponding element
        int el = i+j*size1();

        // Continue to the next element to skip
        while (next_rr!=rr.end() && *next_rr<el) next_rr++;

        // Skip element if necessary
        if (next_rr!=rr.end() && *next_rr==el) {
          next_rr++;
          continue;
        }

        // Keep element
        mapping.push_back(k);

        // Update row
        ret_row[nz++] = i;
      }

      // Update colind
      ret_colind[j+1] = nz;
    }

    // Truncate row vector
    ret_row.resize(nz);

    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  Sparsity SparsityInternal::_erase(const vector<int>& rr, const vector<int>& cc,
                                      bool ind1, std::vector<int>& mapping) const {
    casadi_assert_in_range(rr, -size1()+ind1, size1()+ind1);
    casadi_assert_in_range(cc, -size2()+ind1, size2()+ind1);

    // Handle index-1, negative indices, non-monotone rr and cc
    if (ind1 || has_negative(rr) || has_negative(cc)
        || !is_nondecreasing(rr) || !is_nondecreasing(cc)) {
      // Create substitute rr
      std::vector<int> rr_mod = rr;
      for (vector<int>::iterator i=rr_mod.begin(); i!=rr_mod.end(); ++i) {
        if (ind1) (*i)--;
        if (*i<0) *i += size1();
      }
      std::sort(rr_mod.begin(), rr_mod.end());

      // Create substitute cc
      std::vector<int> cc_mod = cc;
      for (vector<int>::iterator i=cc_mod.begin(); i!=cc_mod.end(); ++i) {
        if (ind1) (*i)--;
        if (*i<0) *i += size2();
      }
      std::sort(cc_mod.begin(), cc_mod.end());

      // Call recursively
      return _erase(rr_mod, cc_mod, false, mapping);
    }

    // Mapping
    mapping.resize(0);

    // Quick return if no elements
    if (numel()==0) return shared_from_this<Sparsity>();

    // Reserve memory
    mapping.reserve(nnz());

    // Return value
    vector<int> ret_colind = get_colind(), ret_row = get_row();

    // Number of non-zeros
    int nz=0;

    // Columns to be erased
    vector<int>::const_iterator ie = cc.begin();

    // First and last index for the col
    int el_first=0, el_last=0;

    // Loop over columns
    for (int i=0; i<size2(); ++i) {
      // Update beginning and end of non-zero indices
      el_first = el_last;
      el_last = ret_colind[i+1];

      // Is it a col that can be deleted
      bool deletable_col = ie!=cc.end() && *ie==i;
      if (deletable_col) {
        ie++;

        // Rows to be erased
        vector<int>::const_iterator je = rr.begin();

        // Loop over nonzero elements of the col
        for (int el=el_first; el<el_last; ++el) {
          // Row
          int j=ret_row[el];

          // Continue to the next row to skip
          for (; je!=rr.end() && *je<j; ++je) {}

          // Remove row if necessary
          if (je!=rr.end() && *je==j) {
            je++;
            continue;
          }

          // Save old nonzero for each new nonzero
          mapping.push_back(el);

          // Update row and increase nonzero counter
          ret_row[nz++] = j;
        }
      } else {
        // Loop over nonzero elements of the col
        for (int el=el_first; el<el_last; ++el) {
          // Row
          int j=ret_row[el];

          // Save old nonzero for each new nonzero
          mapping.push_back(el);

          // Update row and increase nonzero counter
          ret_row[nz++] = j;
        }
      }

      // Register last nonzero of the col
      ret_colind[i+1]=nz;
    }

    // Truncate row matrix
    ret_row.resize(nz);

    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  vector<int> SparsityInternal::get_nz(const vector<int>& rr, const vector<int>& cc) const {
    casadi_assert_bounded(rr, size1());
    casadi_assert_bounded(cc, size2());

    std::vector<int> rr_sorted;
    std::vector<int> rr_sorted_index;

    sort(rr, rr_sorted, rr_sorted_index);

    vector<int> ret(cc.size()*rr.size());

    int stride = rr.size();
    const int* colind = this->colind();
    const int* row = this->row();

    for (int i=0;i<cc.size();++i) {
      int it = cc[i];
      int el=colind[it];
      for (int j=0;j<rr_sorted.size();++j) {
        int jt=rr_sorted[j];
        // Continue to the non-zero element
        for (; el<colind[it+1] && row[el]<jt; ++el) {}
        // Add the non-zero element, if there was an element in the location exists
        if (el<colind[it+1] && row[el]== jt) {
          ret[i*stride+rr_sorted_index[j]] = el;
        } else {
          ret[i*stride+rr_sorted_index[j]] = -1;
        }
      }
    }
    return ret;
  }

  Sparsity SparsityInternal::sub(const vector<int>& rr, const SparsityInternal& sp,
                                 vector<int>& mapping, bool ind1) const {
    casadi_assert_dev(rr.size()==sp.nnz());

    // Check bounds
    casadi_assert_in_range(rr, -numel()+ind1, numel()+ind1);

    // Handle index-1, negative indices
    if (ind1 || has_negative(rr)) {
      std::vector<int> rr_mod = rr;
      for (vector<int>::iterator i=rr_mod.begin(); i!=rr_mod.end(); ++i) {
        casadi_assert(!(ind1 && (*i)<=0),
          "Matlab is 1-based, but requested index " + str(*i) +  ". "
          "Note that negative slices are disabled in the Matlab interface. "
          "Possibly you may want to use 'end'.");
        if (ind1) (*i)--;
        if (*i<0) *i += numel();
      }
      return sub(rr_mod, sp, mapping, false); // Call recursively
    }

    // Find the nonzeros corresponding to rr
    mapping.resize(rr.size());
    std::copy(rr.begin(), rr.end(), mapping.begin());
    get_nz(mapping);

    // Construct new pattern of the corresponding elements
    vector<int> ret_colind(sp.size2()+1), ret_row;
    ret_colind[0] = 0;
    const int* sp_colind = sp.colind();
    const int* sp_row = sp.row();
    for (int c=0; c<sp.size2(); ++c) {
      for (int el=sp_colind[c]; el<sp_colind[c+1]; ++el) {
        if (mapping[el]>=0) {
          mapping[ret_row.size()] = mapping[el];
          ret_row.push_back(sp_row[el]);
        }
      }
      ret_colind[c+1] = ret_row.size();
    }
    mapping.resize(ret_row.size());
    return Sparsity(sp.size1(), sp.size2(), ret_colind, ret_row);
  }

  Sparsity SparsityInternal::sub(const vector<int>& rr, const vector<int>& cc,
                                 vector<int>& mapping, bool ind1) const {
    casadi_assert_in_range(rr, -size1()+ind1, size1()+ind1);
    casadi_assert_in_range(cc, -size2()+ind1, size2()+ind1);

    // Handle index-1, negative indices in rr
    std::vector<int> tmp = rr;
    for (vector<int>::iterator i=tmp.begin(); i!=tmp.end(); ++i) {
      if (ind1) (*i)--;
      if (*i<0) *i += size1();
    }
    std::vector<int> rr_sorted, rr_sorted_index;
    sort(tmp, rr_sorted, rr_sorted_index, false);

    // Handle index-1, negative indices in cc
    tmp = cc;
    for (vector<int>::iterator i=tmp.begin(); i!=tmp.end(); ++i) {
      if (ind1) (*i)--;
      if (*i<0) *i += size2();
    }
    std::vector<int> cc_sorted, cc_sorted_index;
    sort(tmp, cc_sorted, cc_sorted_index, false);
    vector<int> columns, rows;

    // With lookup vector
    bool with_lookup = static_cast<double>(cc.size())*static_cast<double>(rr.size()) > nnz();
    std::vector<int> rrlookup;
    if (with_lookup) {
      // Time complexity: O(ii.size()*(nnz per column))
      // Typical use case:
      // a = SX::sym("a", sp_diag(50000))
      // a[:, :]
      rrlookup = lookupvector(rr_sorted, size1());
      // Else: Time complexity: O(ii.size()*jj.size())
      // Typical use case:
      // a = DM.ones(1000, 1000)
      // a[[0, 1],[0, 1]]
    }

    // count the number of non-zeros
    int nnz = 0;

    // loop over the columns of the slice
    const int* colind = this->colind();
    const int* row = this->row();
    for (int i=0; i<cc.size(); ++i) {
      int it = cc_sorted[i];
      if (with_lookup) {
        // loop over the non-zeros of the matrix
        for (int el=colind[it]; el<colind[it+1]; ++el) {
          int j = row[el];
          int ji = rrlookup[j];
          if (ji!=-1) {
            int jv = rr_sorted[ji];
            while (ji>=0 && jv == rr_sorted[ji--]) nnz++;
          }
        }
      } else {
        // Loop over rr
        int el = colind[it];
        for (int j=0; j<rr_sorted.size(); ++j) {
          int jt=rr_sorted[j];
          // Continue to the non-zero element
          while (el<colind[it+1] && row[el]<jt) el++;
          // Add the non-zero element, if there was an element in the location exists
          if (el<colind[it+1] && row[el]== jt) nnz++;
        }
      }
    }

    mapping.resize(nnz);
    columns.resize(nnz);
    rows.resize(nnz);

    int k = 0;
    // loop over the columns of the slice
    for (int i=0; i<cc.size(); ++i) {
      int it = cc_sorted[i];
      if (with_lookup) {
        // loop over the non-zeros of the matrix
        for (int el=colind[it]; el<colind[it+1]; ++el) {
          int jt = row[el];
          int ji = rrlookup[jt];
          if (ji!=-1) {
            int jv = rr_sorted[ji];
            while (ji>=0 && jv == rr_sorted[ji]) {
              rows[k] = rr_sorted_index[ji];
              columns[k] = cc_sorted_index[i];
              mapping[k] = el;
              k++;
              ji--;
            }
          }
        }
      } else {
        // Loop over rr
        int el = colind[it];
        for (int j=0; j<rr_sorted.size(); ++j) {
          int jt=rr_sorted[j];
          // Continue to the non-zero element
          while (el<colind[it+1] && row[el]<jt) el++;
          // Add the non-zero element, if there was an element in the location exists
          if (el<colind[it+1] && row[el]== jt) {
            rows[k] = rr_sorted_index[j];
            columns[k] = cc_sorted_index[i];
            mapping[k] = el;
            k++;
          }
        }
      }
    }

    std::vector<int> sp_mapping;
    std::vector<int> mapping_ = mapping;
    Sparsity ret = Sparsity::triplet(rr.size(), cc.size(), rows, columns, sp_mapping, false);

    for (int i=0; i<mapping.size(); ++i)
      mapping[i] = mapping_[sp_mapping[i]];

    // Create sparsity pattern
    return ret;
  }

  Sparsity SparsityInternal::combine(const Sparsity& y, bool f0x_is_zero,
                                            bool function0_is_zero) const {
    static vector<unsigned char> mapping;
    return combineGen1<false>(y, f0x_is_zero, function0_is_zero, mapping);
  }

  Sparsity SparsityInternal::combine(const Sparsity& y, bool f0x_is_zero,
                                            bool function0_is_zero,
                                            vector<unsigned char>& mapping) const {
    return combineGen1<true>(y, f0x_is_zero, function0_is_zero, mapping);
  }

  template<bool with_mapping>
  Sparsity SparsityInternal::combineGen1(const Sparsity& y, bool f0x_is_zero,
                                                bool function0_is_zero,
                                                std::vector<unsigned char>& mapping) const {

    // Quick return if identical
    if (is_equal(y)) {
      if (with_mapping) {
        mapping.resize(y.nnz());
        fill(mapping.begin(), mapping.end(), 1 | 2);
      }
      return y;
    }

    if (f0x_is_zero) {
      if (function0_is_zero) {
        return combineGen<with_mapping, true, true>(y, mapping);
      } else {
        return combineGen<with_mapping, true, false>(y, mapping);
      }
    } else if (function0_is_zero) {
      return combineGen<with_mapping, false, true>(y, mapping);
    } else {
      return combineGen<with_mapping, false, false>(y, mapping);
    }
  }

  template<bool with_mapping, bool f0x_is_zero, bool function0_is_zero>
  Sparsity SparsityInternal::combineGen(const Sparsity& y,
                                               vector<unsigned char>& mapping) const {

    // Assert dimensions
    casadi_assert(size2()==y.size2() && size1()==y.size1(), "Dimension mismatch");

    // Sparsity pattern of the argument
    const int* y_colind = y.colind();
    const int* y_row = y.row();
    const int* colind = this->colind();
    const int* row = this->row();

    // Sparsity pattern of the result
    vector<int> ret_colind(size2()+1, 0);
    vector<int> ret_row;

    // Clear the mapping
    if (with_mapping) mapping.clear();

    // Loop over columns of both patterns
    for (int i=0; i<size2(); ++i) {
      // Non-zero element of the two matrices
      int el1 = colind[i];
      int el2 = y_colind[i];

      // End of the non-zero elements of the col for the two matrices
      int el1_last = colind[i+1];
      int el2_last = y_colind[i+1];

      // Loop over the non-zeros of both matrices
      while (el1<el1_last || el2<el2_last) {
        // Get the rows
        int row1 = el1<el1_last ? row[el1] : size1();
        int row2 = el2<el2_last ? y_row[el2] : size1();

        // Add to the return matrix
        if (row1==row2) { //  both nonzero
          ret_row.push_back(row1);
          if (with_mapping) mapping.push_back( 1 | 2);
          el1++; el2++;
        } else if (row1<row2) { //  only first argument is nonzero
          if (!function0_is_zero) {
            ret_row.push_back(row1);
            if (with_mapping) mapping.push_back(1);
          } else {
            if (with_mapping) mapping.push_back(1 | 4);
          }
          el1++;
        } else { //  only second argument is nonzero
          if (!f0x_is_zero) {
            ret_row.push_back(row2);
            if (with_mapping) mapping.push_back(2);
          } else {
            if (with_mapping) mapping.push_back(2 | 4);
          }
          el2++;
        }
      }

      // Save the index of the last nonzero on the col
      ret_colind[i+1] = ret_row.size();
    }

    // Return cached object
    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  bool SparsityInternal::is_equal(const Sparsity& y) const {
    // Quick true if the objects are the same
    if (this == y.get()) return true;

    // Otherwise, compare the patterns
    return is_equal(y.size1(), y.size2(), y.colind(), y.row());
  }

  Sparsity SparsityInternal::pattern_inverse() const {
    // Quick return clauses
    if (is_empty()) return Sparsity::dense(size1(), size2());
    if (is_dense()) return Sparsity(size1(), size2());

    // Sparsity of the result
    std::vector<int> row_ret;
    std::vector<int> colind_ret=get_colind();
    const int* colind = this->colind();
    const int* row = this->row();

    // Loop over columns
    for (int i=0;i<size2();++i) {
      // Update colind vector of the result
      colind_ret[i+1]=colind_ret[i]+size1()-(colind[i+1]-colind[i]);

      // Counter of new row indices
      int j=0;

      // Loop over all nonzeros
      for (int k=colind[i];k<colind[i+1];++k) {

        // Try to reach current nonzero
        while (j<row[k])  {
          // And meanwhile, add nonzeros to the result
          row_ret.push_back(j);
          j++;
        }
        j++;
      }
      // Process the remainder up to the row size
      while (j < size1())  {
        row_ret.push_back(j);
        j++;
      }
    }

    // Return result
    return Sparsity(size1(), size2(), colind_ret, row_ret);
  }


  bool SparsityInternal::is_equal(int y_nrow, int y_ncol, const std::vector<int>& y_colind,
                                 const std::vector<int>& y_row) const {
    casadi_assert_dev(y_colind.size()==y_ncol+1);
    casadi_assert_dev(y_row.size()==y_colind.back());
    return is_equal(y_nrow, y_ncol, get_ptr(y_colind), get_ptr(y_row));
  }

  bool SparsityInternal::is_equal(int y_nrow, int y_ncol,
                                 const int* y_colind, const int* y_row) const {
    const int* colind = this->colind();
    const int* row = this->row();

    // Get number of nonzeros
    int nz = y_colind[y_ncol];

    // First check dimensions and number of non-zeros
    if (nnz()!=nz || size2()!=y_ncol || size1()!=y_nrow) return false;

    // Check if dense
    if (nnz()==numel()) return true;

    // Check the number of non-zeros per col
    if (!equal(colind, colind+size2()+1, y_colind)) return false;

    // Finally check the row indices
    if (!equal(row, row+nz, y_row)) return false;

    // Equal if reached this point
    return true;
  }

  Sparsity SparsityInternal::_appendVector(const SparsityInternal& sp) const {
    casadi_assert(size2() == 1 && sp.size2() == 1,
      "_appendVector(sp): Both arguments must be vectors but got "
       + str(size2()) + " columns for lhs, and " + str(sp.size2()) + " columns for rhs.");

    // Get current number of non-zeros
    int sz = nnz();

    // Add row indices
    vector<int> new_row = get_row();
    const int* sp_row = sp.row();
    new_row.resize(sz + sp.nnz());
    for (int i=sz; i<new_row.size(); ++i)
      new_row[i] = sp_row[i-sz] + size1();

    // New column indices
    vector<int> new_colind(2, 0);
    new_colind[1] = new_row.size();
    return Sparsity(size1()+sp.size1(), 1, new_colind, new_row);
  }

  Sparsity SparsityInternal::_appendColumns(const SparsityInternal& sp) const {
    casadi_assert(size1()== sp.size1(),
      "_appendColumns(sp): row sizes must match but got " + str(size1())
                          + " for lhs, and " + str(sp.size1()) + " for rhs.");

    // Append rows
    vector<int> new_row = get_row();
    const int* sp_row = sp.row();
    new_row.insert(new_row.end(), sp_row, sp_row+sp.nnz());

    // Get column indices
    vector<int> new_colind = get_colind();
    const int* sp_colind = sp.colind();
    new_colind.resize(size2() + sp.size2() + 1);
    for (int i = size2()+1; i<new_colind.size(); ++i)
      new_colind[i] = sp_colind[i-size2()] + nnz();

    return Sparsity(size1(), size2()+sp.size2(), new_colind, new_row);
  }

  Sparsity SparsityInternal::_enlargeColumns(int ncol, const std::vector<int>& cc,
                                               bool ind1) const {
    casadi_assert_in_range(cc, -ncol+ind1, ncol+ind1);

    // Handle index-1, negative indices
    if (ind1 || has_negative(cc)) {
      std::vector<int> cc_mod = cc;
      for (vector<int>::iterator i=cc_mod.begin(); i!=cc_mod.end(); ++i) {
        if (ind1) (*i)--;
        if (*i<0) *i += ncol;
      }
      return _enlargeColumns(ncol, cc_mod, false); // Call recursively
    }

    // Sparsify the columns
    vector<int> new_colind = get_colind();
    new_colind.resize(ncol+1, nnz());

    int ik=cc.back(); // need only to update from the last new index
    int nz=nnz(); // number of nonzeros up till this column
    for (int i=cc.size()-1; i>=0; --i) {
      // Update colindex for new columns
      for (; ik>cc[i]; --ik) {
        new_colind[ik] = nz;
      }

      // Update non-zero counter
      nz = new_colind[i];

      // Update colindex for old colums
      new_colind[cc[i]] = nz;
    }

    // Append zeros to the beginning
    for (; ik>=0; --ik) {
      new_colind[ik] = 0;
    }
    return Sparsity(size1(), ncol, new_colind, get_row());
  }

  Sparsity SparsityInternal::_enlargeRows(int nrow, const std::vector<int>& rr, bool ind1) const {
    casadi_assert_in_range(rr, -nrow+ind1, nrow+ind1);

    // Handle index-1, negative indices
    if (ind1 || has_negative(rr)) {
      std::vector<int> rr_mod = rr;
      for (vector<int>::iterator i=rr_mod.begin(); i!=rr_mod.end(); ++i) {
        if (ind1) (*i)--;
        if (*i<0) *i += nrow;
      }
      return _enlargeRows(nrow, rr_mod, false); // Call recursively
    }

    // Assert dimensions
    casadi_assert_dev(rr.size() == size1());

    // Begin by sparsify the rows
    vector<int> new_row = get_row();
    for (int k=0; k<nnz(); ++k) {
      new_row[k] = rr[new_row[k]];
    }
    return Sparsity(nrow, size2(), get_colind(), new_row);
  }

  Sparsity SparsityInternal::makeDense(std::vector<int>& mapping) const {
    const int* colind = this->colind();
    const int* row = this->row();
    mapping.resize(nnz());
    for (int i=0; i<size2(); ++i) {
      for (int el=colind[i]; el<colind[i+1]; ++el) {
        int j = row[el];
        mapping[el] = j + i*size1();
      }
    }

    return Sparsity::dense(size1(), size2());
  }

  int SparsityInternal::get_nz(int rr, int cc) const {
    // If negative index, count from the back
    if (rr<0) rr += size1();
    if (cc<0) cc += size2();
    const int* colind = this->colind();
    const int* row = this->row();

    // Check consistency
    casadi_assert(rr>=0 && rr<size1(), "Row index " + str(rr)
                          + " out of bounds [0, " + str(size1()) + ")");
    casadi_assert(cc>=0 && cc<size2(), "Column index " + str(cc)
                          + " out of bounds [0, " + str(size2()) + ")");

    // Quick return if matrix is dense
    if (is_dense()) return rr+cc*size1();

    // Quick return if past the end
    if (colind[cc]==nnz() || (colind[cc+1]==nnz() && row[nnz()-1]<rr)) return -1;

    // Find sparse element
    for (int ind=colind[cc]; ind<colind[cc+1]; ++ind) {
      if (row[ind] == rr) {
        return ind;     // element exists
      } else if (row[ind] > rr) {
        break;          // break at the place where the element should be added
      }
    }
    return -1;
  }

  Sparsity SparsityInternal::_reshape(int nrow, int ncol) const {
    // If a dimension is negative, call recursively
    if (nrow<0 && ncol>0) {
      return _reshape(numel()/ncol, ncol);
    } else if (nrow>0 && ncol<0) {
      return _reshape(nrow, numel()/nrow);
    }

    casadi_assert(numel() == nrow*ncol,
                          "reshape: number of elements must remain the same. Old shape is "
                          + dim() + ". New shape is " + str(nrow) + "x" + str(ncol)
                          + "=" + str(nrow*ncol) + ".");
    std::vector<int> ret_col(nnz());
    std::vector<int> ret_row(nnz());
    const int* colind = this->colind();
    const int* row = this->row();
    for (int i=0; i<size2(); ++i) {
      for (int el=colind[i]; el<colind[i+1]; ++el) {
        int j = row[el];

        // Element number
        int k_ret = j+i*size1();

        // Col and row in the new matrix
        int i_ret = k_ret/nrow;
        int j_ret = k_ret%nrow;
        ret_col[el] = i_ret;
        ret_row[el] = j_ret;
      }
    }
    return Sparsity::triplet(nrow, ncol, ret_row, ret_col);
  }

  Sparsity SparsityInternal::_resize(int nrow, int ncol) const {
    // Col and row index of the new
    vector<int> row_new, colind_new(ncol+1, 0);
    const int* colind = this->colind();
    const int* row = this->row();

    // Loop over the columns which may contain nonzeros
    int i;
    for (i=0; i<size2() && i<ncol; ++i) {
      // First nonzero element of the col
      colind_new[i] = row_new.size();

      // Record rows of the nonzeros
      for (int el=colind[i]; el<colind[i+1] && row[el]<nrow; ++el) {
        row_new.push_back(row[el]);
      }
    }

    // Save col-indices for the rest of the columns
    for (; i<ncol+1; ++i) {
      colind_new[i] = row_new.size();
    }

    return Sparsity(nrow, ncol, colind_new, row_new);
  }

  bool SparsityInternal::rowsSequential(bool strictly) const {
    const int* colind = this->colind();
    const int* row = this->row();
    for (int i=0; i<size2(); ++i) {
      int lastrow = -1;
      for (int k=colind[i]; k<colind[i+1]; ++k) {

        // check if not in sequence
        if (row[k] < lastrow)
          return false;

        // Check if duplicate
        if (strictly && row[k] == lastrow)
          return false;

        // update last row of the col
        lastrow = row[k];
      }
    }

    // sequential if reached this point
    return true;
  }

  Sparsity SparsityInternal::_removeDuplicates(std::vector<int>& mapping) const {
    casadi_assert_dev(mapping.size()==nnz());

    // Return value (to be hashed)
    vector<int> ret_colind = get_colind(), ret_row = get_row();

    // Nonzero counter without duplicates
    int k_strict=0;

    // Loop over columns
    for (int i=0; i<size2(); ++i) {

      // Last row encountered on the col so far
      int lastrow = -1;

      // Save new col offset (cannot set it yet, since we will need the old value below)
      int new_colind = k_strict;

      // Loop over nonzeros (including duplicates)
      for (int k=ret_colind[i]; k<ret_colind[i+1]; ++k) {

        // Make sure that the rows appear sequentially
        casadi_assert(ret_row[k] >= lastrow, "rows are not sequential");

        // Skip if duplicate
        if (ret_row[k] == lastrow) continue;

        // update last row encounterd on the col
        lastrow = ret_row[k];

        // Update mapping
        mapping[k_strict] = mapping[k];

        // Update row index
        ret_row[k_strict] = ret_row[k];

        // Increase the strict nonzero counter
        k_strict++;
      }

      // Update col offset
      ret_colind[i] = new_colind;
    }

    // Finalize the sparsity pattern
    ret_colind[size2()] = k_strict;
    ret_row.resize(k_strict);
    mapping.resize(k_strict);
    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  void SparsityInternal::find(std::vector<int>& loc, bool ind1) const {
    casadi_assert(!mul_overflows(size1(), size2()), "Integer overflow detected");
    const int* colind = this->colind();
    const int* row = this->row();

    // Element for each nonzero
    loc.resize(nnz());

    // Loop over columns
    for (int cc=0; cc<size2(); ++cc) {

      // Loop over the nonzeros
      for (int el=colind[cc]; el<colind[cc+1]; ++el) {

        // Get row
        int rr = row[el];

        // Get the element
        loc[el] = rr+cc*size1()+ind1;
      }
    }
  }

  void SparsityInternal::get_nz(std::vector<int>& indices) const {
    // Quick return if no elements
    if (indices.empty()) return;
    const int* colind = this->colind();
    const int* row = this->row();

    // Make a sanity check
    int last=-1;
    for (vector<int>::iterator it=indices.begin(); it!=indices.end(); ++it) {
      if (*it>=0) {
        int el = *it;
        if (el<last) {
          // Sort rr in nondecreasing order, if needed
          std::vector<int> indices_sorted, mapping;
          sort(indices, indices_sorted, mapping, false);
          get_nz(indices_sorted);
          for (size_t i=0; i<indices.size(); ++i) {
            indices[mapping[i]] = indices_sorted[i];
          }
          return;
        }
        last = el;
      }
    }

    // Quick return if no elements
    if (last<0) return;

    // Iterator to input/output
    vector<int>::iterator it=indices.begin();
    while (*it<0) it++; // first non-ignored

    // Current element sought
    int el_row = *it % size1();
    int el_col = *it / size1();

    // Loop over columns
    for (int i=0; i<size2(); ++i) {

      // Loop over the nonzeros
      for (int el=colind[i]; el<colind[i+1] && el_col<=i; ++el) {

        // Get row
        int j = row[el];

        // Add leading elements not in pattern
        while (i>el_col || (i==el_col && j>el_row)) {
          // Mark as not found
          *it = -1;

          // Increase index and terminate if end of vector reached
          if (++it==indices.end()) return;

          // Next element sought
          el_row = *it % size1();
          el_col = *it / size1();
        }

        // Add elements in pattern
        while (i==el_col && j==el_row) {
          // Save element index
          *it = el;

          // Increase index and terminate if end of vector reached
          do {
            if (++it==indices.end()) return;
          } while (*it<0);

          // Next element sought
          el_row = *it % size1();
          el_col = *it / size1();
        }
      }
    }

    // Add trailing elements not in pattern
    fill(it, indices.end(), -1);
  }

  Sparsity SparsityInternal::uni_coloring(const Sparsity& AT, int cutoff) const {

    // Allocate temporary vectors
    vector<int> forbiddenColors;
    forbiddenColors.reserve(size2());
    vector<int> color(size2(), 0);

    // Access the sparsity of the transpose
    const int* AT_colind = AT.colind();
    const int* AT_row = AT.row();
    const int* colind = this->colind();
    const int* row = this->row();

    // Loop over columns
    for (int i=0; i<size2(); ++i) {

      // Loop over nonzero elements
      for (int el=colind[i]; el<colind[i+1]; ++el) {

        // Get row
        int c = row[el];

        // Loop over previous columns that have an element in row c
        for (int el_prev=AT_colind[c]; el_prev<AT_colind[c+1]; ++el_prev) {

          // Get the col
          int i_prev = AT_row[el_prev];

          // Escape loop if we have arrived at the current col
          if (i_prev>=i)
            break;

          // Get the color of the col
          int color_prev = color[i_prev];

          // Mark the color as forbidden for the current col
          forbiddenColors[color_prev] = i;
        }
      }

      // Get the first nonforbidden color
      int color_i;
      for (color_i=0; color_i<forbiddenColors.size(); ++color_i) {
        // Break if color is ok
        if (forbiddenColors[color_i]!=i) break;
      }
      color[i] = color_i;

      // Add color if reached end
      if (color_i==forbiddenColors.size()) {
        forbiddenColors.push_back(0);

        // Cutoff if too many colors
        if (forbiddenColors.size()>cutoff) {
          return Sparsity();
        }
      }
    }

    // Create return sparsity containing the coloring
    vector<int> ret_colind(forbiddenColors.size()+1, 0), ret_row;

    // Get the number of rows for each col
    for (int i=0; i<color.size(); ++i) {
      ret_colind[color[i]+1]++;
    }

    // Cumsum
    for (int j=0; j<forbiddenColors.size(); ++j) {
      ret_colind[j+1] += ret_colind[j];
    }

    // Get row for each col
    ret_row.resize(color.size());
    for (int j=0; j<ret_row.size(); ++j) {
      ret_row[ret_colind[color[j]]++] = j;
    }

    // Swap index back one step
    for (int j=ret_colind.size()-2; j>=0; --j) {
      ret_colind[j+1] = ret_colind[j];
    }
    ret_colind[0] = 0;

    // Return the coloring
    return Sparsity(size2(), forbiddenColors.size(), ret_colind, ret_row);
;
  }

  Sparsity SparsityInternal::star_coloring2(int ordering, int cutoff) const {
    if (!is_square()) {
      // NOTE(@jaeandersson) Why warning and not error?
      casadi_message("StarColoring requires a square matrix, got " + dim() + ".");
    }

    // TODO(Joel): What we need here, is a distance-2 smallest last ordering
    // Reorder, if necessary
    const int* colind = this->colind();
    const int* row = this->row();
    if (ordering!=0) {
      casadi_assert_dev(ordering==1);

      // Ordering
      vector<int> ord = largest_first();

      // Create a new sparsity pattern
      Sparsity sp_permuted = pmult(ord, true, true, true);

      // Star coloring for the permuted matrix
      Sparsity ret_permuted = sp_permuted.star_coloring2(0);

      // Permute result back
      return ret_permuted.pmult(ord, true, false, false);
    }

    // Allocate temporary vectors
    vector<int> forbiddenColors;
    forbiddenColors.reserve(size2());
    vector<int> color(size2(), -1);

    vector<int> firstNeighborP(size2(), -1);
    vector<int> firstNeighborQ(size2(), -1);
    vector<int> firstNeighborQ_el(size2(), -1);

    vector<int> treated(size2(), -1);
    vector<int> hub(nnz_upper(), -1);

    vector<int> Tmapping;
    transpose(Tmapping);

    vector<int> star(nnz());
    int k = 0;
    for (int i=0; i<size2(); ++i) {
      for (int j_el=colind[i]; j_el<colind[i+1]; ++j_el) {
        int j = row[j_el];
        if (i<j) {
          star[j_el] = k;
          star[Tmapping[j]] = k;
          k++;
        }
      }
    }



    int starID = 0;

    // 3: for each v \in V do
    for (int v=0; v<size2(); ++v) {

      // 4: for each colored w \in N1(v) do
      for (int w_el=colind[v]; w_el<colind[v+1]; ++w_el) {
          int w = row[w_el];
          int colorW = color[w];
          if (colorW==-1) continue;

          // 5: forbiddenColors[color[w]] <- v
          forbiddenColors[colorW] = v;

          // 6: (p, q) <- firstNeighbor[color[w]]
          int p = firstNeighborP[colorW];
          int q = firstNeighborQ[colorW];

          // 7: if p = v then    <   Case 1
          if (v==p) {

            // 8: if treated[q] != v then
            if (treated[q]!=v) {

              // 9: treat(v, q)  < forbid colors of neighbors of q

                // treat@2: for each colored x \in N1 (q) do
                for (int x_el=colind[q]; x_el<colind[q+1]; ++x_el) {
                  int x = row[x_el];
                  if (color[x]==-1) continue;

                  // treat@3: forbiddenColors[color[x]] <- v
                  forbiddenColors[color[x]] = v;
                }

                // treat@4: treated[q] <- v
                treated[q] = v;

            }
            // 10: treat(v, w) < forbid colors of neighbors of w

              // treat@2: for each colored x \in N1 (w) do
              for (int x_el=colind[w]; x_el<colind[w+1]; ++x_el) {
                int x = row[x_el];
                if (color[x]==-1) continue;

                // treat@3: forbiddenColors[color[x]] <- v
                forbiddenColors[color[x]] = v;
              }

              // treat@4: treated[w] <- v
              treated[w] = v;

          // 11: else
          } else {

            // 12: firstNeighbor[color[w]] <- (v, w)
            firstNeighborP[colorW] = v;
            firstNeighborQ[colorW] = w;
            firstNeighborQ_el[colorW] = w_el;

            // 13: for each colored vertex x \in N1 (w) do
            int x_el_end = colind[w+1];
            int x, colorx;
            for (int x_el=colind[w]; x_el < x_el_end; ++x_el) {
              x = row[x_el];
              colorx = color[x];
              if (colorx==-1 || x==v) continue;

              // 14: if x = hub[star[wx]] then potential Case 2
              if (hub[star[x_el]]==x) {

                // 15: forbiddenColors[color[x]] <- v
                forbiddenColors[colorx] = v;

              }
            }
          }

      }

      // 16: color[v] <- min {c > 0 : forbiddenColors[c] != v}
      bool new_color = true;
      for (int color_i=0; color_i<forbiddenColors.size(); ++color_i) {
        // Break if color is ok
        if (forbiddenColors[color_i]!=v) {
          color[v] = color_i;
          new_color = false;
          break;
        }
      }

      // New color if reached end
      if (new_color) {
        color[v] = forbiddenColors.size();
        forbiddenColors.push_back(-1);

        // Cutoff if too many colors
        if (forbiddenColors.size()>cutoff) {
          return Sparsity();
        }
      }

      // 17: updateStars(v)

        // updateStars@2: for each colored w \in N1 (v) do
        for (int w_el=colind[v]; w_el<colind[v+1]; ++w_el) {
            int w = row[w_el];
            int colorW = color[w];
            if (colorW==-1) continue;

            // updateStars@3: if exits x \in N1 (w) where x = v and color[x] = color[v] then
            bool check = false;
            int x;
            int x_el;
            for (x_el=colind[w]; x_el<colind[w+1]; ++x_el) {
              x = row[x_el];
              if (x==v || color[x]!=color[v]) continue;
              check = true;
              break;
            }
            if (check) {

              // updateStars@4: hub[star[wx]] <- w
              int starwx = star[x_el];
              hub[starwx] = w;

              // updateStars@5: star[vw] <- star[wx]
              star[w_el]  = starwx;
              star[Tmapping[w_el]] = starwx;

            // updateStars@6: else
            } else {

              // updateStars@7: (p, q) <- firstNeighbor[color[w]]
              int p = firstNeighborP[colorW];
              int q = firstNeighborQ[colorW];
              int q_el = firstNeighborQ_el[colorW];

              // updateStars@8: if (p = v) and (q = w) then
              if (p==v && q!=w) {

                // updateStars@9: hub[star[vq]] <- v
                int starvq = star[q_el];
                hub[starvq] = v;

                // updateStars@10: star[vw] <- star[vq]
                star[w_el]  = starvq;
                star[Tmapping[w_el]] = starvq;

              // updateStars@11: else
              } else {

                // updateStars@12: starID <- starID + 1
                starID+= 1;

                // updateStars@13: star[vw] <- starID
                star[w_el] = starID;
                star[Tmapping[w_el]]= starID;

              }

            }

         }

    }

    // Create return sparsity containing the coloring
    vector<int> ret_colind(forbiddenColors.size()+1, 0), ret_row;

    // Get the number of rows for each col
    for (int i=0; i<color.size(); ++i) {
      ret_colind[color[i]+1]++;
    }

    // Cumsum
    for (int j=0; j<forbiddenColors.size(); ++j) {
      ret_colind[j+1] += ret_colind[j];
    }

    // Get row for each col
    ret_row.resize(color.size());
    for (int j=0; j<ret_row.size(); ++j) {
      ret_row[ret_colind[color[j]]++] = j;
    }

    // Swap index back one step
    for (int j=ret_colind.size()-2; j>=0; --j) {
      ret_colind[j+1] = ret_colind[j];
    }
    ret_colind[0] = 0;

    // Return the coloring
    return Sparsity(size2(), forbiddenColors.size(), ret_colind, ret_row);
  }

  Sparsity SparsityInternal::star_coloring(int ordering, int cutoff) const {
    if (!is_square()) {
      // NOTE(@jaeandersson) Why warning and not error?
      casadi_message("StarColoring requires a square matrix, got " + dim() + ".");
    }

    // Reorder, if necessary
    if (ordering!=0) {
      casadi_assert_dev(ordering==1);

      // Ordering
      vector<int> ord = largest_first();

      // Create a new sparsity pattern
      Sparsity sp_permuted = pmult(ord, true, true, true);

      // Star coloring for the permuted matrix
      Sparsity ret_permuted = sp_permuted.star_coloring(0);

      // Permute result back
      return ret_permuted.pmult(ord, true, false, false);
    }

    // Allocate temporary vectors
    const int* colind = this->colind();
    const int* row = this->row();
    vector<int> forbiddenColors;
    forbiddenColors.reserve(size2());
    vector<int> color(size2(), -1);

    // 4: for i <- 1 to |V | do
    for (int i=0; i<size2(); ++i) {

      // 5: for each w \in N1 (vi) do
      for (int w_el=colind[i]; w_el<colind[i+1]; ++w_el) {
        int w = row[w_el];

        // 6: if w is colored then
        if (color[w]!=-1) {

          // 7: forbiddenColors[color[w]] <- v
          forbiddenColors[color[w]] = i;

        } // 8: end if

        // 9: for each colored vertex x \in N1 (w) do
        for (int x_el=colind[w]; x_el<colind[w+1]; ++x_el) {
          int x = row[x_el];
          if (color[x]==-1) continue;

          // 10: if w is not colored then
          if (color[w]==-1) {

            //11: forbiddenColors[color[x]] <- vi
            forbiddenColors[color[x]] = i;

          } else { // 12: else

            // 13: for each colored vertex y \in N1 (x), y != w do
            for (int y_el=colind[x]; y_el<colind[x+1]; ++y_el) {
              int y = row[y_el];
              if (color[y]==-1 || y==w) continue;

              // 14: if color[y] = color[w] then
              if (color[y]==color[w]) {

                // 15: forbiddenColors[color[x]] <- vi
                forbiddenColors[color[x]] = i;

                // 16: break
                break;

              } // 17: end if

            } // 18: end for

          } // 19: end if

        } // 20 end for

      } // 21 end for

      // 22: color[v] <- min {c > 0 : forbiddenColors[c] = v}
      bool new_color = true;
      for (int color_i=0; color_i<forbiddenColors.size(); ++color_i) {
        // Break if color is ok
        if (forbiddenColors[color_i]!=i) {
          color[i] = color_i;
          new_color = false;
          break;
        }
      }

      // New color if reached end
      if (new_color) {
        color[i] = forbiddenColors.size();
        forbiddenColors.push_back(-1);

        // Cutoff if too many colors
        if (forbiddenColors.size()>cutoff) {
          return Sparsity();
        }
      }

    } // 23 end for

    // Number of colors used
    int num_colors = forbiddenColors.size();

    // Return sparsity in sparse triplet format
    return Sparsity::triplet(size2(), num_colors, range(color.size()), color);
  }

  std::vector<int> SparsityInternal::largest_first() const {
    vector<int> degree = get_colind();
    int max_degree = 0;
    for (int k=0; k<size2(); ++k) {
      degree[k] = degree[k+1]-degree[k];
      max_degree = max(max_degree, 1+degree[k]);
    }
    degree.resize(size2());

    // Vector for binary sort
    vector<int> degree_count(max_degree+1, 0);
    for (vector<int>::const_iterator it=degree.begin(); it!=degree.end(); ++it) {
      degree_count.at(*it+1)++;
    }

    // Cumsum to get the offset for each degree
    for (int d=0; d<max_degree; ++d) {
      degree_count[d+1] += degree_count[d];
    }

    // Now a bucket sort
    vector<int> ordering(size2());
    for (int k=size2()-1; k>=0; --k) {
      ordering[degree_count[degree[k]]++] = k;
    }

    // Invert the ordering
    vector<int>& reverse_ordering = degree_count; // reuse memory
    reverse_ordering.resize(ordering.size());
    copy(ordering.begin(), ordering.end(), reverse_ordering.rbegin());

    // Return the ordering
    return reverse_ordering;
  }

  Sparsity SparsityInternal::pmult(const std::vector<int>& p, bool permute_rows,
                                   bool permute_columns, bool invert_permutation) const {
    // Invert p, possibly
    vector<int> p_inv;
    if (invert_permutation) {
      p_inv.resize(p.size());
      for (int k=0; k<p.size(); ++k) {
        p_inv[p[k]] = k;
      }
    }
    const vector<int>& pp = invert_permutation ? p_inv : p;

    // Get columns
    vector<int> col = get_col();

    // Get rows
    const int* row = this->row();

    // Sparsity of the return matrix
    vector<int> new_row(col.size()), new_col(col.size());

    // Possibly permute columns
    if (permute_columns) {
      // Assert dimensions
      casadi_assert_dev(p.size()==size2());

      // Permute
      for (int k=0; k<col.size(); ++k) {
        new_col[k] = pp[col[k]];
      }

    } else {
      // No permutation of columns
      copy(col.begin(), col.end(), new_col.begin());
    }

    // Possibly permute rows
    if (permute_rows) {
      // Assert dimensions
      casadi_assert_dev(p.size()==size1());

      // Permute
      for (int k=0; k<nnz(); ++k) {
        new_row[k] = pp[row[k]];
      }

    } else {
      // No permutation of rows
      copy(row, row+nnz(), new_row.begin());
    }

    // Return permuted matrix
    return Sparsity::triplet(size1(), size2(), new_row, new_col);
  }

  bool SparsityInternal::is_transpose(const SparsityInternal& y) const {
    // Assert dimensions and number of nonzeros
    if (size2()!=y.size1() || size1()!=y.size2() || nnz()!=y.nnz())
      return false;

    // Quick return if empty interior or dense
    if (nnz()==0 || is_dense())
      return true;

    // Run algorithm on the pattern with the least number of rows
    if (size1()>size2()) return y.is_transpose(*this);

    // Index counter for columns of the possible transpose
    vector<int> y_col_count(y.size2(), 0);
    const int* colind = this->colind();
    const int* row = this->row();
    const int* y_colind = y.colind();
    const int* y_row = y.row();

    // Loop over the columns
    for (int i=0; i<size2(); ++i) {

      // Loop over the nonzeros
      for (int el=colind[i]; el<colind[i+1]; ++el) {

        // Get the row
        int j=row[el];

        // Get the element of the possible transpose
        int el_y = y_colind[j] + y_col_count[j]++;

        // Quick return if element doesn't exist
        if (el_y>=y_colind[j+1]) return false;

        // Get the row of the possible transpose
        int j_y = y_row[el_y];

        // Quick return if mismatch
        if (j_y != i) return false;
      }
    }

    // Transpose if reached this point
    return true;
  }

  bool SparsityInternal::is_reshape(const SparsityInternal& y) const {
    // Quick return if the objects are the same
    if (this==&y) return true;

    // Check if same number of entries and nonzeros
    if (numel()!=y.numel() || nnz()!=y.nnz()) return false;

    // Quick return if empty interior or dense
    if (nnz()==0 || is_dense()) return true;

    // Get Pattern
    const int* colind = this->colind();
    const int* row = this->row();
    const int* y_colind = y.colind();
    const int* y_row = y.row();

    // If same number of rows, check if patterns are identical
    if (size1()==y.size1()) return is_equal(y.size1(), y.size2(), y_colind, y_row);

    // Loop over the elements
    for (int cc=0; cc<size2(); ++cc) {
      for (int el=colind[cc]; el<colind[cc+1]; ++el) {
        int rr=row[el];

        // Get row and column of y
        int loc = rr+size1()*cc;
        int rr_y = loc % y.size1();
        int cc_y = loc / y.size1();

        // Make sure matching
        if (rr_y != y_row[el] || el<y_colind[cc_y] || el>=y_colind[cc_y+1])
          return false;
      }
    }

    // Reshape if reached this point
    return true;
  }

  void SparsityInternal::spy(std::ostream &stream) const {

    // Index counter for each column
    std::vector<int> cind = get_colind();
    const int* colind = this->colind();
    const int* row = this->row();

    // Loop over rows
    for (int rr=0; rr<size1(); ++rr) {

      // Loop over columns
      for (int cc=0; cc<size2(); ++cc) {
        // Check if nonzero
        if (cind[cc]<colind[cc+1] && row[cind[cc]]==rr) {
          stream << "*";
          cind[cc]++;
        } else {
          stream << ".";
        }
      }

      // End of row
      stream << endl;
    }
  }

  void SparsityInternal::export_code(const std::string& lang,
      std::ostream &stream, const Dict& options) const {
    casadi_assert(lang=="matlab", "Only matlab language supported for now.");

    // Default values for options
    bool opt_inline = false;
    std::string name = "sp";
    bool as_matrix = true;
    int indent_level = 0;

    // Read options
    for (auto&& op : options) {
      if (op.first=="inline") {
        opt_inline = op.second;
      } else if (op.first=="name") {
        name = op.second.to_string();
      } else if (op.first=="as_matrix") {
        as_matrix = op.second;
      } else if (op.first=="indent_level") {
        indent_level = op.second;
      } else {
        casadi_error("Unknown option '" + op.first + "'.");
      }
    }

    // Construct indent string
    std::string indent = "";
    for (int i=0;i<indent_level;++i) {
      indent += "  ";
    }

    casadi_assert(!opt_inline, "Inline not supported for now.");

    // Export dimensions
    stream << indent << name << "_m = " << size1() << ";" << endl;
    stream << indent << name << "_n = " << size2() << ";" << endl;

    // Matlab indices are one-based
    const int index_offset = 1;

    // Print columns
    const int* colind = this->colind();
    const int* row = this->row();
    stream << indent << name<< "_j = [";
    bool first = true;
    for (int i=0; i<size2(); ++i) {
      for (int el=colind[i]; el<colind[i+1]; ++el) {
        if (!first) stream << ", ";
        stream << (i+index_offset);
        first = false;
      }
    }
    stream << "];" << endl;

    // Print rows
    stream << indent << name << "_i = [";
    first = true;
    int nz = nnz();
    for (int i=0; i<nz; ++i) {
      if (!first) stream << ", ";
      stream << (row[i]+index_offset);
      first = false;
    }
    stream << "];" << endl;

    if (as_matrix) {
      // Generate matrix
      stream << indent << name << " = sparse(" << name << "_i, " << name << "_j, ";
      stream << "ones(size(" << name << "_i)), ";
      stream << name << "_m, " << name << "_n);" << endl;
    }

  }

  void SparsityInternal::spy_matlab(const std::string& mfile_name) const {
    // Create the .m file
    ofstream mfile;
    mfile.open(mfile_name.c_str());

    // Header
    mfile << "% This function was automatically generated by CasADi" << endl;

    Dict opts;
    opts["name"] = "A";
    export_code("matlab", mfile, opts);

    // Issue spy command
    mfile << "spy(A);" << endl;

    mfile.close();
  }

  std::size_t SparsityInternal::hash() const {
    return hash_sparsity(size1(), size2(), colind(), row());
  }

  bool SparsityInternal::is_tril() const {
    const int* colind = this->colind();
    const int* row = this->row();
    // loop over columns
    for (int i=0; i<size2(); ++i) {
      if (colind[i] != colind[i+1]) { // if there are any elements of the column
        // check row of the top-most element of the column
        int rr = row[colind[i]];

        // not lower triangular if row>i
        if (rr<i) return false;
      }
    }
    // all columns ok
    return true;
  }

  bool SparsityInternal::is_triu() const {
    const int* colind = this->colind();
    const int* row = this->row();
    // loop over columns
    for (int i=0; i<size2(); ++i) {
      if (colind[i] != colind[i+1]) { // if there are any elements of the column
        // check row of the bottom-most element of the column
        int rr = row[colind[i+1]-1];

        // not upper triangular if row>i
        if (rr>i) return false;
      }
    }
    // all columns ok
    return true;
  }

  Sparsity SparsityInternal::_tril(bool includeDiagonal) const {
    const int* colind = this->colind();
    const int* row = this->row();
    vector<int> ret_colind, ret_row;
    ret_colind.reserve(size2()+1);
    ret_colind.push_back(0);
    for (int cc=0; cc<size2(); ++cc) {
      for (int el=colind[cc]; el<colind[cc+1]; ++el) {
        int rr=row[el];
        if (rr>cc || (includeDiagonal && rr==cc)) {
          ret_row.push_back(rr);
        }
      }
      ret_colind.push_back(ret_row.size());
    }
    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  Sparsity SparsityInternal::_triu(bool includeDiagonal) const {
    const int* colind = this->colind();
    const int* row = this->row();
    vector<int> ret_colind, ret_row;
    ret_colind.reserve(size2()+1);
    ret_colind.push_back(0);
    for (int cc=0; cc<size2(); ++cc) {
      for (int el=colind[cc]; el<colind[cc+1]; ++el) {
        int rr=row[el];
        if (rr<cc || (includeDiagonal && rr==cc)) {
          ret_row.push_back(rr);
        }
      }
      ret_colind.push_back(ret_row.size());
    }
    return Sparsity(size1(), size2(), ret_colind, ret_row);
  }

  std::vector<int> SparsityInternal::get_lower() const {
    const int* colind = this->colind();
    const int* row = this->row();
    vector<int> ret;
    for (int cc=0; cc<size2(); ++cc) {
      for (int el = colind[cc]; el<colind[cc+1]; ++el) {
        if (row[el]>=cc) {
          ret.push_back(el);
        }
      }
    }
    return ret;
  }

  std::vector<int> SparsityInternal::get_upper() const {
    const int* colind = this->colind();
    const int* row = this->row();
    vector<int> ret;
    for (int cc=0; cc<size2(); ++cc) {
      for (int el = colind[cc]; el<colind[cc+1] && row[el]<=cc; ++el) {
        ret.push_back(el);
      }
    }
    return ret;
  }

  int SparsityInternal::bw_upper() const {
    int bw = 0;
    const int* colind = this->colind();
    const int* row = this->row();
    for (int cc=0; cc<size2(); ++cc) {
      if (colind[cc] != colind[cc+1]) { // if there are any elements of the column
        int rr = row[colind[cc]];
        bw = std::max(bw, cc-rr);
      }
    }
    return bw;
  }

  int SparsityInternal::bw_lower() const {
    int bw = 0;
    const int* colind = this->colind();
    const int* row = this->row();
    for (int cc=0; cc<size2(); ++cc) {
      if (colind[cc] != colind[cc+1]) { // if there are any elements of the column
        int rr = row[colind[cc+1]-1];
        bw = std::max(bw, rr-cc);
      }
    }
    return bw;
  }

  vector<int> SparsityInternal::get_colind() const {
    const int* colind = this->colind();
    return vector<int>(colind, colind+size2()+1);
  }

  vector<int> SparsityInternal::get_row() const {
    const int* row = this->row();
    return vector<int>(row, row+nnz());
  }

  void SparsityInternal::
  spsolve(bvec_t* X, const bvec_t* B, bool tr) const {
    const Btf& btf = this->btf();
    const int* colind = this->colind();
    const int* row = this->row();

    if (!tr) {
      for (int b=0; b<btf.nb; ++b) { // loop over the blocks forward

        // Get dependencies from all right-hand-sides in the block ...
        bvec_t block_dep = 0;
        for (int el=btf.rowblock[b]; el<btf.rowblock[b+1]; ++el) {
          int rr = btf.rowperm[el];
          block_dep |= B[rr];
        }

        // ... as well as all other variables in the block
        for (int el=btf.colblock[b]; el<btf.colblock[b+1]; ++el) {
          int cc = btf.colperm[el];
          block_dep |= X[cc];
        }

        // Propagate ...
        for (int el=btf.colblock[b]; el<btf.colblock[b+1]; ++el) {
          int cc = btf.colperm[el];

          // ... to all variables in the block ...
          X[cc] |= block_dep;

          // ... as well as to other variables which depends on variables in the block
          for (int k=colind[cc]; k<colind[cc+1]; ++k) {
            int rr=row[k];
            X[rr] |= block_dep;
          }
        }
      }

    } else { // transpose
      for (int b=btf.nb-1; b>=0; --b) { // loop over the blocks backward

        // Get dependencies ...
        bvec_t block_dep = 0;
        for (int el=btf.colblock[b]; el<btf.colblock[b+1]; ++el) {
          int cc = btf.colperm[el];

          // .. from all right-hand-sides in the block ...
          block_dep |= B[cc];

          // ... as well as from all depending variables ...
          for (int k=colind[cc]; k<colind[cc+1]; ++k) {
            int rr=row[k];
            block_dep |= X[rr];
          }
        }

        // Propagate to all variables in the block
        for (int el=btf.rowblock[b]; el<btf.rowblock[b+1]; ++el) {
          int rr = btf.rowperm[el];
          X[rr] |= block_dep;
        }
      }
    }
  }

} // namespace casadi
