// NOLINT(legal/copyright)
// SYMBOL "norm_inf_mul"
template<typename T1>
T1 casadi_norm_inf_mul(const T1* x, const s_t* sp_x, const T1* y, const s_t* sp_y, T1* dwork, s_t* iwork) { // NOLINT(whitespace/line_length)
  T1 res = 0;
  // Get sparsities
  s_t nrow_x = sp_x[0], ncol_x = sp_x[1];
  const s_t *colind_x = sp_x+2, *row_x = sp_x + 2 + ncol_x+1;
  s_t ncol_y = sp_y[1];
  const s_t *colind_y = sp_y+2, *row_y = sp_y + 2 + ncol_y+1;

  // Implementation borrowed from Scipy's sparsetools/csr.h
  // method that uses O(n) temp storage
  s_t *mask = iwork + ncol_y+1;

  s_t i, jj, kk;
  // Pass 1
  for (i=0; i<nrow_x; ++i) mask[i] = -1;
  iwork[0] = 0;
  s_t nnz = 0;
  for (i=0; i<ncol_y; ++i) {
    s_t row_nnz = 0;
    for (jj=colind_y[i]; jj < colind_y[i+1]; jj++) {
      s_t j = row_y[jj];
      for (kk=colind_x[j]; kk < colind_x[j+1]; kk++) {
        s_t k = row_x[kk];
        if (mask[k] != i) {
          mask[k] = i;
          row_nnz++;
        }
      }
    }
    s_t next_nnz = nnz + row_nnz;
    nnz = next_nnz;
    iwork[i+1] = nnz;
  }

  // Pass 2
  s_t *next = iwork + ncol_y+1;
  for (i=0; i<nrow_x; ++i) next[i] = -1;
  T1* sums = dwork;
  for (i=0; i<nrow_x; ++i) sums[i] = 0;
  nnz = 0;
  iwork[0] = 0;
  for (i=0; i<ncol_y; ++i) {
    s_t head   = -2;
    s_t length =  0;
    s_t jj_start = colind_y[i];
    s_t jj_end   = colind_y[i+1];
    for (jj=jj_start; jj<jj_end; ++jj) {
      s_t j = row_y[jj];
      T1 v = y[jj];
      s_t kk_start = colind_x[j];
      s_t kk_end   = colind_x[j+1];
      for (kk = kk_start; kk<kk_end; ++kk) {
        s_t k = row_x[kk];
        sums[k] += v*x[kk];
        if (next[k] == -1) {
          next[k] = head;
          head  = k;
          length++;
        }
      }
    }
    for (jj=0; jj<length; ++jj) {
      if (!is_zero(sums[head])) {
        T1 a = fabs(sums[head]);
        res = fmax(res, a);
        nnz++;
      }
      s_t temp = head;
      head = next[head];
      next[temp] = -1; //clear arrays
      sums[temp] =  0;
    }
    iwork[i+1] = nnz;
  }
  return res;
}
