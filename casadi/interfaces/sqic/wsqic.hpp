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

/// \cond INTERNAL

extern "C" {

  extern void sqic(
   const s_t *m, // Number of constraints + 1 (for the objective)
   const s_t* n, // Number of decision variables
   const s_t* nnzA, // Number of nonzeros in objective-augmented linear constraint matrix  A
   const s_t *indA, // colind of Compressed Column Storage A , length: nnzA
   const s_t *locA, // row of  Compressed Column Storage A, length n + 1
   const double *valA, // Values of A
   const double* bl, // Lower bounds to decision variables + objective
   const double* bu, // Upper bounds to decision variables + objective
   const s_t *hEtype, // ?
   const s_t *hs, // ?
   double *x,  // Decision variables + evaluated linear constraints ((initial + optimal), length n+m
   double *pi, // ?
   double *rc, // Multipliers (initial + optimal), length n+m
   const s_t* nnzH, // Number of nonzeros in full hessian H
   const s_t* indH, // colind of Compressed Column Storage H , length: nnzH
   const s_t* locH, // row of  Compressed Column Storage H, length n + 1
   double* valH
   );

  extern void sqicSolve(
   double* Obj // Output: hessian part of the resulting objective
  );

  extern void sqicSolveStabilized(
   double* Obj, // Output: hessian part of the resulting objective
   double *mu,
   s_t *lenpi,
   double* piE
  );

  extern void sqicDestroy();
}
/// \endcond
