// NOLINT(legal/copyright)
// SYMBOL "trans"
template<typename T1>
void casadi_trans(const T1* x, const s_t* sp_x, T1* y, const s_t* sp_y, s_t* tmp) {
  s_t ncol_x = sp_x[1];
  s_t nnz_x = sp_x[2 + ncol_x];
  const s_t* row_x = sp_x + 2 + ncol_x+1;
  s_t ncol_y = sp_y[1];
  const s_t* colind_y = sp_y+2;
  s_t k;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}
