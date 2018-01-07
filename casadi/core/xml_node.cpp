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


#include "xml_node.hpp"
#include "casadi_misc.hpp"

using namespace std;
namespace casadi {

  XmlNode::XmlNode() {
  }

  XmlNode::~XmlNode() {
  }

  bool XmlNode::hasAttribute(const string& attribute_name) const {
    map<string, string>::const_iterator it = attributes_.find(attribute_name);
    return it!=attributes_.end();
  }

  XmlNode& XmlNode::operator[](s_t i) {
    casadi_assert(i>=0 && i < size(),
      "index out of bounds for element " + str(i) + " of node " + name());
    return children_.at(i);
  }

  const XmlNode& XmlNode::operator[](s_t i) const {
    return const_cast<XmlNode*>(this)->operator[](i);
  }

  bool XmlNode::hasChild(const string& childname) const {
    map<string, s_t>::const_iterator it = child_indices_.find(childname);
    return it!=child_indices_.end();
  }

  XmlNode& XmlNode::operator[](const string& childname) {
    // Find the child
    map<string, s_t>::const_iterator it = child_indices_.find(childname);

    // check that the child was indeed found
    if (it == child_indices_.end()) {
      casadi_error("could not find " + childname);
    }

    // Return an index to the child
    return children_[it->second];
  }

  const XmlNode& XmlNode::operator[](const string& childname) const {
    return const_cast<XmlNode*>(this)->operator[](childname);
  }

  void XmlNode::set_attribute(const string& attribute_name, const string& attribute) {
    attributes_[attribute_name] = attribute;
  }

  ostream& operator<<(ostream &stream, const XmlNode& node) {
    node.dump(stream);
    return stream;
  }

  s_t XmlNode::size() const {
    return children_.size();
  }

  const string& XmlNode::name() const {
    return name_;
  }

  void XmlNode::setName(const string& name) {
    name_ = name;
  }

  void XmlNode::dump(ostream &stream, s_t indent) const {
    // Print name
    stream << string(indent, ' ') << "Node: " << name_ << endl;

    // Print comment
    if (!comment_.empty()) {
      stream << string(indent, ' ') << "----- comment starts ----- "  << endl;
      stream << comment_ << endl;
      stream << string(indent, ' ') << "----- comment ends ----- "  << endl;
    }

    // Print text
    if (!text_.empty())
      stream << string(indent+2, ' ') << "Text: " << text_ << endl;

    // Print attributes
    for (map<string, string>::const_iterator it=attributes_.begin(); it != attributes_.end(); ++it)
      stream << string(indent+2, ' ') << "attribute " << it->first << " = " << it->second << endl;

    // Print Children
    for (s_t i=0; i<size(); ++i) {
      stream << string(indent, ' ') << "Child " << i << ":" << endl;
      (*this)[i].dump(stream, indent+2);
    }
  }

  bool XmlNode::checkName(const string& str) const {
    return name_.compare(str) == 0;
  }

  void XmlNode::readString(const std::string& str, std::string& val) {
    val = str;
  }

  void XmlNode::readString(const std::string& str, bool& val) {
    if (str.compare("true")==0)
      val = true;
    else if (str.compare("false")==0)
      val = false;
    else
      throw CasadiException("XML argument not true or false");
  }

  void XmlNode::readString(const std::string& str, s_t& val) {
    std::istringstream buffer(str);
    buffer >> val;
  }

  void XmlNode::readString(const std::string& str, double& val) {
    std::istringstream buffer(str);
    buffer >> val;
  }

} // namespace casadi
