#ifndef DOF_HANDLER_H
#define DOF_HANDLER_H

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "FEM/finite_element/finite_element.h"

struct Mesh;

class DOFHandler {
 public:
  DOFHandler(const Mesh& mesh, const FE& element);

  std::size_t get_global_ndofs() const;
  //   Returns the global DOF positions for the given element
  std::vector<std::size_t> get_element_dof_indices(
      std::size_t element_id) const;

  //   Map mesh node index to global DOF index
  std::size_t get_vertex_dof_index(const std::size_t mesh_node_id) const;

  //   Map mesh edge midpoints to global DOF indices
  std::vector<std::size_t> get_edge_dof_indices(
      const std::size_t endpoint_a,
      const std::size_t endpoint_b) const;

 private:
  const Mesh& mesh;
  std::size_t global_ndofs{};
  std::vector<int> local_to_global{};
  std::vector<std::vector<std::size_t>> element_dof_indices;
  std::vector<std::size_t> node_to_vertex_dof;
  //   Global solution DOFs, mapped by unique mesh edge
  std::vector<std::vector<std::size_t>> edge_global_dofs;
};

#endif
