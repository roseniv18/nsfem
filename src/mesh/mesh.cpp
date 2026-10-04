
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

  for (auto& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
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
  }
}

std::vector<bool> get_dirichlet_nodes(const Mesh& mesh) {
  // this vector tells us which nodes are Dirichlet
  std::vector<bool> is_dirichlet(mesh.nodes.size(), false);

  /**
   * loop through all elements in the mesh
   * for each element, analyze which physical groups it belongs to, as well if
   * it is a line element (Dirichlet boundary line)
   */
  for (const auto& element : mesh.elements) {
    for (int pt : element.physical_tags) {
      const auto& physical_group = mesh.physical_groups.at(pt);

      //   std::cout << physical_group.name << '\n';

      if (element.type == ElementType::Line2 &&
          physical_group.name == "Dirichlet") {
        for (int node_id : element.node_ids) {
          is_dirichlet[node_id] = true;
        }
      }
    }
  }

  return is_dirichlet;
}

std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         double t) {
  std::vector<double> values(mesh.nodes.size());

  for (int node_id = 0; node_id < mesh.nodes.size(); node_id++) {
    if (is_dirichlet[node_id]) {
      const auto& node = mesh.nodes[node_id];

      values[node_id] = fn(node.x, node.y, t);
    }
  }

  return values;
}
