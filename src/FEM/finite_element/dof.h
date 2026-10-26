#ifndef DOF_H
#define DOF_H

namespace nsfem::fem {

enum class DOFLoc { Vertex, Edge, Interior };

/**
 * Represents a degree of freedom
 * local_index: the local index of the DOF for the specific finite element
 * location: whether the DOF lives on a vertex, edge or in the interior
 * entity_index: the local index of the edge / vertex (0,1,2)
 * entity_dof_index: allows support for multiple dofs on the same edge
 * (higher-order elements) ; defaults to 0
 *
 * Example for Lagrange P2 triangle with nodes (a,b,c) and edges (ab, bc, ca)
 * To describe the midpoint on edge ab:
 * DOF {4, DOFLoc::Edge, 0, 0};
 */
struct DOF {
  std::size_t local_index;
  DOFLoc location;
  std::size_t entity_index;
  std::size_t entity_dof_index = 0;
};

}  // namespace nsfem::fem

#endif
