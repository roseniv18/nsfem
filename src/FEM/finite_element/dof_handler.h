#ifndef DOF_HANDLER_H
#define DOF_HANDLER_H

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "finite_element.h"

/** ------------------
 *
 * DOFHandler
 *
 * Handles the mapping from local degrees of freedom to global degrees of
 * freedom, which are then used in the assembly phase and the Dirichlet boundary
 * condition modification phase.
 *
 * ------------------
 */

// forward declaration
namespace nsfem::mesh {
struct Mesh;
struct Node;
struct Element;
struct Edge;
enum class ElementType;
constexpr bool is_triangle_type(const ElementType type);
}  // namespace nsfem::mesh

namespace nsfem::fem {

using nsfem::fem::FE;

class DOFHandler {
 public:
  DOFHandler(const nsfem::mesh::Mesh& mesh, const FE& element);

  // Returns the amount of global DOFs
  std::size_t get_global_ndofs() const;

  //   Returns the global DOF positions for the given element
  const std::vector<std::size_t>& get_element_dof_indices(
      std::size_t element_id) const;

  //   Map mesh node index to global DOF index
  std::size_t get_vertex_dof_index(const std::size_t mesh_node_id) const;

  //   Map mesh edge midpoints to global DOF indices
  const std::vector<std::size_t>& get_edge_dof_indices(
      const std::size_t endpoint_a,
      const std::size_t endpoint_b) const;

 private:
  const nsfem::mesh::Mesh& mesh;
  //   Amount of global DOFs
  std::size_t global_ndofs{};
  //   Matrix which associates each triangular element with a set of global DOFs
  std::vector<std::vector<std::size_t>> element_dof_indices;
  //   Map global mesh node to unique global vertex DOF
  std::vector<std::size_t> node_to_vertex_dof;
  //   Map global mesh edges to a set of unique global edge DOFs
  std::vector<std::vector<std::size_t>> edge_global_dofs;
};

/** Calculate the local-to-global mapping
 * for elements whose DOFs have 2 components [x y]
 * The global mapping follows a component-major ordering:
 * [x1 x2 ... xn y1 y2 ... yn]
 *
 */
std::size_t component_major_global_dof(
    const std::size_t& local_dof,
    const std::size_t& n_local_scalar_dofs,
    const std::size_t& n_global_scalar_dofs,
    const std::vector<std::size_t>& element_scalar_dofs);

}  // namespace nsfem::fem

#endif
