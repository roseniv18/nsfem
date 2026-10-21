#include "dof_handler.h"
#include <limits>
#include "mesh/mesh.h"

constexpr std::size_t invalid_dof = std::numeric_limits<std::size_t>::max();

DOFHandler::DOFHandler(const Mesh& mesh_, const FE& element) : mesh(mesh_) {
  const std::size_t nelements = mesh.elements.size();
  const int local_ndofs = element.get_ndofs();
  const std::vector<DOF> fe_dofs = element.get_dofs();

  //   Determine the number of global DOFs
  // Create a vector that maps geometric nodes to unique vertex DOFs
  node_to_vertex_dof.assign(mesh.nodes.size(), invalid_dof);

  std::size_t n_vertex_dofs = 0;

  for (const Element& el : mesh.elements) {
    if (!is_triangle_type(el.type))
      continue;

    //   The first 3 nodes are always vertices
    //   (https://gmsh.info/doc/texinfo/#Node-ordering)
    for (std::size_t i = 0; i < 3; i++) {
      const std::size_t node_id = el.node_ids.at(i);

      //   Only assign unique vertices
      //   shared vertices account for a single DOF
      if (node_to_vertex_dof.at(node_id) == invalid_dof) {
        node_to_vertex_dof.at(node_id) = n_vertex_dofs;
        n_vertex_dofs++;
      }
    }
  }

  global_ndofs = n_vertex_dofs;

  if (element.has_edge_dofs())
    global_ndofs += element.n_edge_dofs() * mesh.unique_edges.size();

  edge_global_dofs.resize(mesh.unique_edges.size());
  const std::size_t dofs_per_edge = element.n_edge_dofs();
  std::size_t next_dof = n_vertex_dofs;

  for (std::size_t e = 0; e < mesh.unique_edges.size(); e++) {
    auto& edge_dofs = edge_global_dofs.at(e);

    edge_dofs.resize(dofs_per_edge);

    for (std::size_t j = 0; j < dofs_per_edge; j++) {
      edge_dofs.at(j) = next_dof;
      next_dof++;
    }
  }

  /**
   * ! TODO
   * For elements P3, P4, ...
   * It is necessary to account for interior DOFs
   */

  //   Create the local to global mapping of DOF indices
  element_dof_indices.resize(nelements);

  //   Create mapping for each (triangle) element in the mesh
  for (std::size_t e = 0; e < nelements; e++) {
    const Element& mesh_element = mesh.elements.at(e);

    if (!is_triangle_type(mesh_element.type))
      continue;

    // Each element is described by a set of global DOFs (mapping vector)
    std::vector<std::size_t>& mapping = element_dof_indices.at(e);
    mapping.resize(local_ndofs);

    /**
     * Conceptually, the global DOF mapping indices are structured like this:
     * element_dof_indices = [ -node_ids- -edge_ids- -interior_ids- ]
     * Meaning, the first positions are taken by the node_ids
     * the middle positions by the edge_ids
     * the final positions by the interior_ids
     */
    for (const DOF& dof : fe_dofs) {
      switch (dof.location) {
        case DOFLoc::Vertex: {
          const std::size_t node_id =
              mesh_element.node_ids.at(dof.entity_index);

          mapping.at(dof.local_index) = node_to_vertex_dof.at(node_id);
          break;
        }
        //   Global Edge DOF index are offset with
        case DOFLoc::Edge: {
          const std::size_t edge_id =
              mesh_element.edge_ids.at(dof.entity_index);

          const std::size_t edge_dof_index = dof.entity_dof_index;

          mapping.at(dof.local_index) =
              edge_global_dofs.at(edge_id).at(edge_dof_index);
          break;
        }

        case DOFLoc::Interior:
          throw std::runtime_error("Interior DOFs not yet implemented.");

        default:
          throw std::runtime_error("Invalid DOF specified!");
      }
    }
  }
}

std::size_t DOFHandler::get_global_ndofs() const {
  return global_ndofs;
}

std::vector<std::size_t> DOFHandler::get_element_dof_indices(
    std::size_t element_id) const {
  return element_dof_indices.at(element_id);
}

std::size_t DOFHandler::get_vertex_dof_index(
    const std::size_t mesh_node_id) const {
  const std::size_t dof = node_to_vertex_dof.at(mesh_node_id);

  if (dof == invalid_dof) {
    throw std::invalid_argument(
        "Mesh node is not a topological triangle vertex");
  }

  return dof;
}

std::vector<std::size_t> DOFHandler::get_edge_dof_indices(
    const std::size_t endpoint_a,
    const std::size_t endpoint_b) const {
  for (std::size_t e = 0; e < mesh.unique_edges.size(); e++) {
    const Edge& edge = mesh.unique_edges.at(e);

    const bool match =
        (edge.node_id_1 == endpoint_a && edge.node_id_2 == endpoint_b) ||
        (edge.node_id_1 == endpoint_b && edge.node_id_2 == endpoint_a);

    if (match)
      return edge_global_dofs.at(e);
  }

  throw std::invalid_argument("Edge not found in mesh.unique_edges");
}
