// NOLINT(legal/copyright)
// SYMBOL "flip"
inline
s_t casadi_flip(s_t* corner, s_t ndim) {
  s_t i;
  for (i=0; i<ndim; ++i) {
    if (corner[i]) {
      corner[i]=0;
    } else {
      corner[i]=1;
      return 1;
    }
  }
  return 0;
}
