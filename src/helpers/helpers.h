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
using nsfem::fem::FE, nsfem::fem::DOFHandler;
using nsfem::mesh::Mesh, nsfem::mesh::Node, nsfem::mesh::Element,
    nsfem::mesh::ElementType, nsfem::mesh::PhysicalGroup,
    nsfem::mesh::DIRICHLET;
;
using nsfem::mapping::FEMap2D;

std::string strip_quotes(std::string& str);

// n - resolution of mesh
Mesh make_unit_square_mesh(int n);
Mesh make_two_triangle_mesh();

// tolerance (used for tests that use EXPECT_NEAR)
const double test_dtol = 1e-12;

// interpolate a function at finite element's degrees of freedom
VectorXd interpolate_at_dofs(const Mesh& mesh,
                             const FE& fe,
                             const DOFHandler& dofh,
                             const STFunction& func,
                             double t);

const std::vector<Node> phys_nodes_tr3{{1, 1.0, 2.0},
                                       {2, 4.0, 2.0},
                                       {3, 1.0, 6.0}};

const std::vector<Node> phys_nodes_tr6_linear{{1, 1.0, 2.0}, {2, 4.0, 2.0},
                                              {3, 1.0, 6.0}, {4, 2.5, 2.0},
                                              {5, 2.5, 4.0}, {6, 1.0, 4.0}};

const std::vector<Node> phys_nodes_tr6_nonlinear{{1, 1.0, 2.0}, {2, 4.0, 2.0},
                                                 {3, 1.0, 6.0}, {4, 2.5, 2.0},
                                                 {5, 3.0, 4.5}, {6, 1.0, 4.0}};

const Element phys_element_tr3{.dim = 2,
                               .element_tag = 1,
                               .type = ElementType::Triangle3,
                               .physical_tags = {},
                               .node_ids = {0, 1, 2},
                               .edge_ids = {0, 1, 2}};

const Element phys_element_tr6{.dim = 2,
                               .element_tag = 1,
                               .type = ElementType::Triangle6,
                               .physical_tags = {},
                               .node_ids = {0, 1, 2, 3, 4, 5},
                               .edge_ids = {0, 1, 2}};

#endif
