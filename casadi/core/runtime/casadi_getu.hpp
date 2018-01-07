// NOLINT(legal/copyright)
// SYMBOL "getu"
template<typename T1>
void casadi_getu(const T1* x, const s_t* sp_x, T1* v) {
  // Get sparsities
  s_t ncol_x = sp_x[1];
  const s_t *colind_x = sp_x+2, *row_x = sp_x + 2 + ncol_x+1;
  // Loop over the columns of x
  s_t cc, el;
  for (cc=0; cc<ncol_x; ++cc) {
    // Loop over the nonzeros of x
    for (el=colind_x[cc]; el<colind_x[cc+1] && row_x[el]<=cc; ++el) {
      *v++ = x[el];
    }
  }
}
