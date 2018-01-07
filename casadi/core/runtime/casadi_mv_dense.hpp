// NOLINT(legal/copyright)
// SYMBOL "mv_dense"
template<typename T1>
void casadi_mv_dense(const T1* x, s_t nrow_x, s_t ncol_x, const T1* y, T1* z, s_t tr) {
  if (!x || !y || !z) return;
  s_t i, j;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (j=0; j<nrow_x; ++j) {
        z[i] += *x++ * y[j];
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (j=0; j<nrow_x; ++j) {
        z[j] += *x++ * y[i];
      }
    }
  }
}
