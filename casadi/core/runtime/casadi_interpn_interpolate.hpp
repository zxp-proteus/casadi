// NOLINT(legal/copyright)
// SYMBOL "interpn_interpolate"
template<typename T1>
void casadi_interpn_interpolate(T1* res, s_t ndim, const s_t* offset, const T1* values, const T1* alpha, const s_t* index, const s_t* corner, T1* coeff, s_t m) { // NOLINT(whitespace/line_length)
  // Get weight and value for corner
  T1 c=1;
  s_t ld=1; // leading dimension
  s_t i;
  for (i=0; i<ndim; ++i) {
    if (coeff) *coeff++ = c;
    if (corner[i]) {
      c *= alpha[i];
    } else {
      c *= 1-alpha[i];
    }
    values += (index[i]+corner[i])*ld*m;
    ld *= offset[i+1]-offset[i];
  }
  if (coeff) {
    for (i=0;i<m;++i) res[i] += values[i];
  } else {
    for (i=0;i<m;++i) res[i] += c*values[i];
  }
}
