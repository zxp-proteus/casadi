// NOLINT(legal/copyright)
// SYMBOL "sparsify"
template<typename T1, typename T2>
void casadi_sparsify(const T1* x, T2* y, const s_t* sp_y, s_t tr) {
  s_t nrow_y = sp_y[0], ncol_y = sp_y[1];
  const s_t *colind_y = sp_y+2, *row_y = sp_y+ncol_y+3;
  s_t i, el;
  if (tr) {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(T2, x[i + row_y[el]*ncol_y]);
      }
    }
  } else {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(T2, x[row_y[el] + i*nrow_y]);
      }
    }
  }
}
