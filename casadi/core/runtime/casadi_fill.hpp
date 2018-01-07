// NOLINT(legal/copyright)
// SYMBOL "fill"
template<typename T1>
void casadi_fill(T1* x, s_t n, T1 alpha) {
  s_t i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}
