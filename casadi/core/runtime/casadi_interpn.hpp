// NOLINT(legal/copyright)
// SYMBOL "interpn"
template<typename T1>
void casadi_interpn(T1* res, s_t ndim, const T1* grid, const s_t* offset, const T1* values, const T1* x, const s_t* lookup_mode, s_t m, s_t* iw, T1* w) { // NOLINT(whitespace/line_length)
  // Work vectors
  T1* alpha = w; w += ndim;
  s_t* index = iw; iw += ndim;
  s_t* corner = iw; iw += ndim;
  // Left index and fraction of interval
  casadi_interpn_weights(ndim, grid, offset, x, alpha, index, lookup_mode);
  // Loop over all corners, add contribution to output
  casadi_fill_int(corner, ndim, 0);
  casadi_fill(res, m, 0.0);
  do {
    T1* coeff = 0;
    casadi_interpn_interpolate(res, ndim, offset, values,
      alpha, index, corner, coeff, m);
  } while (casadi_flip(corner, ndim));
}
