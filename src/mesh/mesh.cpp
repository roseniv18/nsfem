
#include "mesh/mesh.h"

#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "FEM/mapping/fe_mapping.h"

std::vector<Node> get_element_nodes(const Element& element, const Mesh& mesh) {
  std::vector<Node> nodes{};
  nodes.reserve(element.node_ids.size());

  for (std::size_t id : element.node_ids) {
    nodes.push_back(Node{mesh.nodes.at(id)});
  }

  return nodes;
}

void build_unique_edges_list(Mesh& mesh) {
  std::map<Edge, int> edge_to_id_map;
  int edge_counter = 0;

  mesh.unique_edges.reserve(mesh.nodes.size());

  //   1 - HANDLE TRIANGLES
  for (auto& element : mesh.elements) {
    if (element.type == ElementType::Triangle3 ||
        element.type == ElementType::Triangle6) {
      element.edge_ids.resize(3);

      std::array<Edge, 3> local_edges = {
          Edge{element.node_ids.at(0), element.node_ids.at(1)},
          Edge{element.node_ids.at(1), element.node_ids.at(2)},
          Edge{element.node_ids.at(2), element.node_ids.at(0)},
      };

      for (int i = 0; i < 3; i++) {
        const Edge& edge = local_edges.at(i);

        // If new edge (unique)
        if (edge_to_id_map.find(edge) == edge_to_id_map.end()) {
          edge_to_id_map[edge] = edge_counter;
          mesh.unique_edges.push_back(edge);
          edge_counter++;
        }

        // Assign the global edge id to the element
        element.edge_ids.at(i) = edge_to_id_map[edge];
      }
    }

    // 2 - HANDLE BOUNDARY LINES
    else if (element.type == ElementType::Line2 ||
             element.type == ElementType::Line3) {
      element.edge_ids.resize(1);

      //   corner nodes are always at index 0 and index 1 for line elements in
      //   gmsh
      Edge boundary_edge{element.node_ids.at(0), element.node_ids.at(1)};

      if (edge_to_id_map.find(boundary_edge) == edge_to_id_map.end()) {
        edge_to_id_map[boundary_edge] = edge_counter;
        mesh.unique_edges.push_back(boundary_edge);
        edge_counter++;
      }

      element.edge_ids.at(0) = edge_to_id_map[boundary_edge];
    }
  }
}

std::vector<bool> get_dirichlet_dofs(const Mesh& mesh, const DOFHandler& dofh) {
  std::vector<bool> is_dirichlet(dofh.get_global_ndofs(), false);

  for (const Element& element : mesh.elements) {
    if (!is_line(element.type))
      continue;

    bool is_dirichlet_boundary = false;

    for (const int tag : element.physical_tags) {
      if (mesh.physical_groups.at(tag).name == DIRICHLET) {
        is_dirichlet_boundary = true;
        break;
      }
    }

    if (!is_dirichlet_boundary)
      continue;

    //   Mark the 2 endpoint vertex DOFS
    for (std::size_t i = 0; i < 2; i++) {
      const std::size_t node_id = element.node_ids.at(i);
      const std::size_t dof = dofh.get_vertex_dof_index(node_id);

      is_dirichlet.at(dof) = true;
    }

    // Mark edge DOFs
    const auto edge_dofs = dofh.get_edge_dof_indices(element.node_ids.at(0),
                                                     element.node_ids.at(1));

    for (const std::size_t dof : edge_dofs)
      is_dirichlet.at(dof) = true;
  }

  return is_dirichlet;
}

std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const FE& fe,
                                         const DOFHandler& dofh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         double t) {
  const std::size_t global_ndofs = dofh.get_global_ndofs();

  if (is_dirichlet.size() != global_ndofs) {
    throw std::invalid_argument(
        "Dirichlet DOF count does not match global DOF count");
  }

  std::vector<double> values(global_ndofs, 0.0);

  const auto ref_nodes = fe.get_ref_nodes();

  if (ref_nodes.size() != static_cast<std::size_t>(fe.get_ndofs())) {
    throw std::runtime_error(
        "Reference node count does not match Finite Element DOF count");
  }

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle(element.type))
      continue;

    const auto geo_fe = FE::build_geo_type(element.type);

    const auto element_nodes = get_element_nodes(element, mesh);

    const FEMap2D mapping(element, element_nodes, *geo_fe);

    const auto element_dofs = dofh.get_element_dof_indices(e);

    for (std::size_t i = 0; i < element_dofs.size(); i++) {
      const std::size_t global_dof = element_dofs.at(i);

      if (!is_dirichlet.at(global_dof))
        continue;

      const auto phys_point = mapping.physical_point(ref_nodes.at(i));

      values.at(global_dof) = fn(phys_point.x, phys_point.y, t);
    }
  }

  return values;
}
