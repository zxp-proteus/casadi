// NOLINT(legal/copyright)
// SYMBOL "copy"
template<typename T1>
void casadi_copy(const T1* x, s_t n, T1* y) {
  s_t i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}
