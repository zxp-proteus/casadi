// NOLINT(legal/copyright)
// SYMBOL "mtimes"
template<typename T1>
void casadi_mtimes(const T1* x, const s_t* sp_x, const T1* y, const s_t* sp_y, T1* z, const s_t* sp_z, T1* w, s_t tr) { // NOLINT(whitespace/line_length)
  // Get sparsities
  s_t ncol_x = sp_x[1];
  const s_t *colind_x = sp_x+2, *row_x = sp_x + 2 + ncol_x+1;
  s_t ncol_y = sp_y[1];
  const s_t *colind_y = sp_y+2, *row_y = sp_y + 2 + ncol_y+1;
  s_t ncol_z = sp_z[1];
  const s_t *colind_z = sp_z+2, *row_z = sp_z + 2 + ncol_z+1;

  s_t cc, kk, kk1;
  if (tr) {
    // Loop over the columns of y and z
    for (cc=0; cc<ncol_z; ++cc) {
      // Get the dense column of y
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      // Loop over the nonzeros of z
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        s_t rr = row_z[kk];
        // Loop over corresponding columns of x
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    // Loop over the columns of y and z
    for (cc=0; cc<ncol_y; ++cc) {
      // Get the dense column of z
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      // Loop over the nonzeros of y
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        s_t rr = row_y[kk];
        // Loop over corresponding columns of x
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      // Get the sparse column of z
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}
