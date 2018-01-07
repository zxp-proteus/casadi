// NOLINT(legal/copyright)
// SYMBOL "low"
template<typename T1>
s_t casadi_low(T1 x, const double* grid, s_t ng, s_t lookup_mode) {
  if (lookup_mode) {
    double g0 = grid[0];
    s_t ret = (s_t) ((x-g0)*(ng-1)/(grid[ng-1]-g0)); // NOLINT(readability/casting)
    if (ret<0) ret=0;
    if (ret>ng-2) ret=ng-2;
    return ret;
  } else {
    s_t i;
    for (i=0; i<ng-2; ++i) {
      if (x < grid[i+1]) break;
    }
    return i;
  }
}
