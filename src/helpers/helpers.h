#ifndef HELPERS_H
#define HELPERS_H

#include <Eigen/Dense>
#include <string>
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "FEM/mapping/fe_mapping.h"
#include "math/function.h"
#include "mesh/mesh.h"
#include "mesh/parser.h"

using Eigen::VectorXd;

std::string strip_quotes(std::string& str);

// n - resolution of mesh
Mesh make_unit_square_mesh(int n);
Mesh make_two_triangle_mesh();

// tolerance (used for tests that use EXPECT_NEAR)
const double test_dtol = 1e-12;

// interpolate a function at finite element's degrees of freedom
VectorXd interpolate_at_dofs(const Mesh& mesh,
                             const FE& fe,
                             const FE& geometry_fe,
                             const DOFHandler& dofh,
                             const STFunction& func,
                             double t);

#endif
