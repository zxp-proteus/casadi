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


#ifndef CASADI_SLICE_HPP
#define CASADI_SLICE_HPP

#include <vector>
#include "exception.hpp"
#include "printable.hpp"
#include <limits>
#include <iostream>
#include "generic_type.hpp"

namespace casadi {

  /** \brief Class representing a Slice
   *
   * Note that Python or Octave do not need to use this class.
   * They can just use slicing utility from the host language ( M[0:6]  in Python, M(1:7) )
   */
  class CASADI_EXPORT Slice
    : public SWIG_IF_ELSE(PrintableCommon, Printable<Slice>) {
  public:
    /// start value: negative values will get added to length
    s_t start;
    /// stop value: use std::numeric_limits<s_t>::max() to indicate unboundedness
    s_t stop;
    s_t step;

    /// Default constructor - all elements
    Slice();

    /// A single element (explicit to avoid ambiguity with IM overload
    explicit Slice(s_t i, bool ind1=false);

    /// A slice
    Slice(s_t start, s_t stop, s_t step=1);
    Slice(int start, int stop, int step=1);
    Slice(int start, s_t stop, int step=1);
    Slice(s_t start, int stop, int step=1);

    /// Get a vector of indices
    std::vector<s_t> all(s_t len, bool ind1=false) const;

    /// Get a vector of indices (nested slice)
    std::vector<s_t> all(const Slice& outer, s_t len) const;

    /// Is the slice a scalar
    bool is_scalar(s_t len) const;

    /// Get scalar (if is_scalar)
    s_t scalar(s_t len) const;

    /// Check equality
    bool operator==(const Slice& other) const {
      return start==other.start && stop==other.stop && step==other.step;
    }

    /// Check inequality
    bool operator!=(const Slice& other) const { return !(*this == other);}

    /// Get name of the class
    std::string type_name() const {return "Slice";}

    /// Print a description of the object
    void disp(std::ostream& stream, bool more=false) const;

    /// Get string representation
    std::string get_str(bool more=false) const {
      std::stringstream ss;
      disp(ss, more);
      return ss.str();
    }

    /** Obtain information */
    Dict info() const {
      return {{"start", start}, {"stop", stop}, {"step", step}};
    }
  };

  /// Construct from an index vector (requires is_slice(v) to be true)
  Slice CASADI_EXPORT to_slice(const std::vector<s_t>& v, bool ind1=false);

  /// Construct nested slices from an index vector (requires is_slice2(v) to be true)
  std::pair<Slice, Slice> CASADI_EXPORT to_slice2(const std::vector<s_t>& v);

  /// Check if an index vector can be represented more efficiently as a slice
  bool CASADI_EXPORT is_slice(const std::vector<s_t>& v, bool ind1=false);

  /// Check if an index vector can be represented more efficiently as two nested slices
  bool CASADI_EXPORT is_slice2(const std::vector<s_t>& v);

} // namespace casadi

#endif // CASADI_SLICE_HPP
