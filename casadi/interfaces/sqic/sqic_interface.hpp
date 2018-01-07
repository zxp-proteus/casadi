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


#ifndef CASADI_SQIC_INTERFACE_HPP
#define CASADI_SQIC_INTERFACE_HPP

#include "casadi/core/conic.hpp"
#include <casadi/interfaces/sqic/casadi_conic_sqic_export.h>

/** \defgroup plugin_Conic_sqic
       Interface to the SQIC solver for quadratic programming
*/

/** \pluginsection{Conic,sqic} */

/// \cond INTERNAL
namespace casadi {

  /**  \brief \pluginbrief{Conic,sqic}

       @copydoc Conic_doc
       @copydoc plugin_Conic_sqic
       \author Joris Gillis
       \date 2013

  */
  class CASADI_CONIC_SQIC_EXPORT SqicInterface : public Conic {
  public:
    /** \brief  Constructor */
    explicit SqicInterface();

    /** \brief  Create a new QP Solver */
    static Conic* creator(const std::map<std::string, Sparsity>& st) {
      return new SqicInterface(st);
    }

    /** \brief  Create a new Solver */
    explicit SqicInterface(const std::map<std::string, Sparsity>& st);

    /** \brief  Destructor */
    ~SqicInterface() override;

    // Get name of the plugin
    const char* plugin_name() const override { return "sqic";}

    // Get name of the class
    std::string class_name() const override { return "SqicInterface";}

    /** \brief  Initialize */
    virtual void init();

    /** \brief Generate native code for debugging */
    virtual void generateNativeCode(std::ostream& file) const;

    virtual void evaluate();

    /// Throw error
    static void sqic_error(const std::string& module, s_t flag);

    /// Calculate the error message map
    static std::map<s_t, std::string> calc_flagmap();

    /// Error message map
    static std::map<s_t, std::string> flagmap;

    /// A documentation string
    static const std::string meta_doc;

  protected:

    /// Flag: is already initialized
    bool is_init_;

    /// Storage space for sqic \p bl variable
    std::vector<double> bl_;
    /// Storage space for sqic \p bu variable
    std::vector<double> bu_;
    /// Storage space for sqic \p x variable
    std::vector<double> x_;
    /// Storage space for sqic \p locA variable
    std::vector<s_t> locA_;
    /// Storage space for sqic \p indA variable
    std::vector<s_t> indA_;
    /// Storage space for sqic \p hs variable
    std::vector<s_t> hs_;
    /// Storage space for sqic \p hEtype variable
    std::vector<s_t> hEtype_;
    /// Storage space for sqic \p indH variable
    std::vector<s_t> indH_;
    /// Storage space for sqic \p locH variable
    std::vector<s_t> locH_;
    /// Storage space for sqic \p rc variable
    std::vector<double> rc_;
    /// Storage space for sqic \p rc variable
    std::vector<double> pi_;

    /// Helper function to bring A into correct format
    Function formatA_;

    /// SQIC inf
    double inf_;
  };


} // namespace casadi

/// \endcond
#endif // CASADI_SQIC_INTERFACE_HPP
