// NOLINT(legal/copyright)
// SYMBOL "dot"
template<typename T1>
T1 casadi_dot(s_t n, const T1* x, const T1* y) {
  T1 r = 0;
  s_t i;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}
