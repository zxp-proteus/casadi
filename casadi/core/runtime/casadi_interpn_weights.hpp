// NOLINT(legal/copyright)
// SYMBOL "interpn_weights"
template<typename T1>
void casadi_interpn_weights(s_t ndim, const T1* grid, const s_t* offset, const T1* x, T1* alpha, s_t* index, const s_t* lookup_mode) { // NOLINT(whitespace/line_length)
  // Left index and fraction of interval
  s_t i;
  for (i=0; i<ndim; ++i) {
    // Grid point
    T1 xi = x ? x[i] : 0;
    // Grid
    const T1* g = grid + offset[i];
    s_t ng = offset[i+1]-offset[i];
    // Find left index
    s_t j = index[i] = casadi_low(xi, g, ng, lookup_mode[i]);
    // Get interpolation/extrapolation alpha
    alpha[i] = (xi-g[j])/(g[j+1]-g[j]);
  }
}
