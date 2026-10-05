#include "dof_handler.h"
#include "mesh/mesh.h"

DOFHandler::DOFHandler(const Mesh& mesh, const FE& element) {
  const int nelements = mesh.elements.size();
  const int nnodes = mesh.nodes.size();
  const int local_ndofs = element.get_ndofs();

  //   global number of dofs
  generate_global_ndofs(mesh, element.type);

  element_dof_indices.resize(nelements);

  for (std::size_t e = 0; e < nelements; e++) {
    const Element& mesh_element = mesh.elements.at(e);

    if (mesh_element.type == ElementType::Triangle3) {
      auto& mapping = element_dof_indices.at(e);
      mapping.resize(local_ndofs);

      if (element.type == FEType::P1) {
        for (int i = 0; i < local_ndofs; i++) {
          mapping.at(i) = mesh_element.node_ids.at(i);
        }
      } else if (element.type == FEType::P2) {
        // 1. first assign the 3 DOFs to Vertices
        for (int i = 0; i < 3; i++) {
          mapping.at(i) = mesh_element.node_ids.at(i);
        }

        // 2. next assign 3 DOFs to the Edge Midpoints
        for (int i = 0; i < 3; i++) {
          mapping.at(i + 3) = mesh.nodes.size() + mesh_element.edge_ids.at(i);
        }
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

void DOFHandler::generate_global_ndofs(const Mesh& mesh,
                                       const FEType& fe_type) {
  switch (fe_type) {
    case FEType::P1:
      global_ndofs = mesh.nodes.size();
      break;
    case FEType::P2:
      global_ndofs = mesh.nodes.size() + mesh.unique_edges.size();
      break;
    default:
      throw std::runtime_error("Cannot determine count of global DOFs!");
  }
}
