
#include "mesh/mesh.h"

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

std::vector<bool> get_dirichlet_nodes(const Mesh& mesh,
                                      const int global_ndofs,
                                      const FEType& fe_type) {
  const std::size_t nvertices = mesh.nodes.size();

  // this vector tells us which nodes are Dirichlet
  std::vector<bool> is_dirichlet(global_ndofs, false);

  /**
   * loop through all elements in the mesh
   * for each element, analyze which physical groups it belongs to, as well if
   * it is a line element (Dirichlet boundary line)
   */
  for (const auto& element : mesh.elements) {
    for (const int pt : element.physical_tags) {
      const auto& physical_group = mesh.physical_groups.at(pt);

      //   std::cout << physical_group.name << '\n';

      if ((element.type == ElementType::Line2 ||
           element.type == ElementType::Line3) &&
          physical_group.name == "Dirichlet") {
        // 1. Fix vertex DOFs (true for P1, P2)
        for (int i = 0; i < 2; i++) {
          const int global_vertex_id = element.node_ids.at(i);
          is_dirichlet[global_vertex_id] = true;
        }

        // 2. Fix edge DOFs (for P2)
        if (fe_type == FEType::P2) {
          const int global_edge_id = element.edge_ids.at(0);
          const int global_edge_dof = nvertices + global_edge_id;
          is_dirichlet[global_edge_dof] = true;
        }
      }
    }
  }

  return is_dirichlet;
}

std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         const int global_ndofs,
                                         const FEType& fe_type,
                                         double t) {
  const std::size_t nvertices = mesh.nodes.size();

  std::vector<double> values(global_ndofs, 0.0);

  //   Assign values to vertex DOFs
  for (std::size_t i = 0; i < nvertices; i++) {
    if (is_dirichlet.at(i)) {
      values.at(i) = fn(mesh.nodes.at(i).x, mesh.nodes.at(i).y, t);
    }
  }

  //   Assign values to edge DOFs (for P2 elements)
  if (fe_type == FEType::P2) {
    for (const auto& element : mesh.elements) {
      for (const int pt : element.physical_tags) {
        const auto& physical_group = mesh.physical_groups.at(pt);

        if ((element.type == ElementType::Line2 ||
             element.type == ElementType::Line3) &&
            physical_group.name == "Dirichlet") {
          const int global_edge_id = element.edge_ids.at(0);
          const int global_edge_dof = nvertices + global_edge_id;

          if (is_dirichlet.at(global_edge_dof)) {
            const auto& node1 = mesh.nodes.at(element.node_ids.at(0));
            const auto& node2 = mesh.nodes.at(element.node_ids.at(1));
            const double mid_x = 0.5 * (node1.x + node2.x);
            const double mid_y = 0.5 * (node1.y + node2.y);

            values.at(global_edge_dof) = fn(mid_x, mid_y, t);
          }
        }
      }
    }
  }
  return values;
}
