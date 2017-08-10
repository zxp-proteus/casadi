// NOLINT(legal/copyright)
template<typename T1>
struct CASADI_PREFIX(central_diff_mem) {
  // Dimensions
  int n_x, n_r;
  // Perturbation size
  T1 h;
  // (Current) differentiable inputs
  T1 *x, *x0;
  // (Current) differentiable outputs
  T1 *r, *r0;
  // Jacobian
  T1* J;
  // Control
  int status, i;
};