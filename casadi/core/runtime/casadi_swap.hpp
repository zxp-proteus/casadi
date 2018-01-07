// NOLINT(legal/copyright)
// SYMBOL "swap"
template<typename T1>
void casadi_swap(s_t n, T1* x, s_t inc_x, T1* y, s_t inc_y) {
  T1 t;
  s_t i;
  for (i=0; i<n; ++i) {
    t = *x;
    *x = *y;
    *y = t;
    x += inc_x;
    y += inc_y;
  }
}
