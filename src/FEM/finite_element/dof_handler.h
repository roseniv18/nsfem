#ifndef DOF_HANDLER_H
#define DOF_HANDLER_H

#include <memory>
#include <vector>
#include "finite_element.h"

struct Mesh;

enum class DOFLoc { Vertex, Edge, Interior };

/**
 * Represents a degree of freedom
 * local_index: the local index of the DOF for the specific finite element
 * location: whether the DOF lives on a vertex, edge or in the interior
 * entity_index: the local index of the edge / vertex (0,1,2)
 *
 * Example for Lagrange P2 triangle with nodes (a,b,c) and edges (ab, bc, ca)
 * To describe the midpoint on edge ab:
 * DOF {4, DOFLoc::Edge, 0};
 */
struct DOF {
  int local_index;
  DOFLoc location;
  int entity_index;
};

class DOFHandler {
 public:
  DOFHandler(const Mesh& mesh, const FE& element);

  std::size_t get_global_ndofs() const;
  //   Returns the global DOF positions for the given element
  std::vector<std::size_t> get_element_dof_indices(
      std::size_t element_id) const;
  //   Determine the number of global degrees of freedom based on the finite
  //   element type
  void generate_global_ndofs(const Mesh& mesh, const FEType& fe_type);

 private:
  std::size_t global_ndofs{};
  std::vector<int> local_to_global{};
  std::vector<std::vector<std::size_t>> element_dof_indices;
};

#endif
