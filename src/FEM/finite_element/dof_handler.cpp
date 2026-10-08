#include "dof_handler.h"
#include "mesh/mesh.h"

DOFHandler::DOFHandler(const Mesh& mesh, const FE& element) {
  const int nelements = mesh.elements.size();
  const int nnodes = mesh.nodes.size();
  const int local_ndofs = element.get_ndofs();
  const std::vector<DOF> fe_dofs = element.get_dofs();

  //   Determine the number of global DOFs
  global_ndofs = mesh.nodes.size();

  for (const DOF& dof : fe_dofs) {
    if (dof.location == DOFLoc::Edge) {
      global_ndofs += mesh.unique_edges.size();
      break;
    }
  }

  //   Create the local to global mapping of DOF indices
  element_dof_indices.resize(nelements);

  //   Create mapping for each element in the mesh
  for (std::size_t e = 0; e < nelements; e++) {
    const Element& mesh_element = mesh.elements.at(e);

    if (mesh_element.type != ElementType::Triangle3)
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
        case DOFLoc::Vertex:
          mapping.at(dof.local_index) =
              mesh_element.node_ids.at(dof.entity_index);
          break;

          //   Global Edge DOF index are offset with
        case DOFLoc::Edge:
          mapping.at(dof.local_index) =
              nnodes + mesh_element.edge_ids.at(dof.entity_index);
          break;

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
