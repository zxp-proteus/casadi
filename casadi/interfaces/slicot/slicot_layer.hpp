/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#ifndef CASADI_SLICOT_LAYER_HPP
#define CASADI_SLICOT_LAYER_HPP

namespace casadi {
  s_t slicot_mb03vd(s_t n, s_t p, s_t ilo, s_t ihi, double * a, s_t lda1, s_t lda2, double * tau,
                     s_t ldtau, double * dwork=0);

  s_t slicot_mb03vy(s_t n, s_t p, s_t ilo, s_t ihi, double * a, s_t lda1, s_t lda2,
                     const double * tau, s_t ldtau, double * dwork=0, s_t ldwork=0);

  s_t slicot_mb03wd(char job, char compz, s_t n, s_t p, s_t ilo, s_t ihi, s_t iloz, s_t ihiz,
                     double *h, s_t ldh1, s_t ldh2, double* z, s_t ldz1, s_t ldz2, double* wr,
                     double *wi, double * dwork=0, s_t ldwork=0);

  s_t slicot_mb05nd(s_t n, double delta, const double* a, s_t lda,
                     double* ex, s_t ldex, double * exint, s_t ldexin,
                     double tol, s_t* iwork, double * dwork, s_t ldwork);

} // namespace casadi

/// \endcond
#endif // CASADI_SLICOT_LAYER_HPP
