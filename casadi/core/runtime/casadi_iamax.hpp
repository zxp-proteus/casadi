// NOLINT(legal/copyright)
// SYMBOL "iamax"
template<typename T1>
s_t casadi_iamax(s_t n, const T1* x, s_t inc_x) {
  T1 t;
  T1 largest_value = -1.0;
  s_t largest_index = -1;
  s_t i;
  for (i=0; i<n; ++i) {
    t = fabs(*x);
    x += inc_x;
    if (t>largest_value) {
      largest_value = t;
      largest_index = i;
    }
  }
  return largest_index;
}
