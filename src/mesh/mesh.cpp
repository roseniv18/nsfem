
#include "mesh/mesh.h"

std::vector<Node> get_element_nodes(const Element& element, const Mesh& mesh) {
  std::vector<Node> nodes{};
  nodes.reserve(element.node_indices.size());

  for (std::size_t id : element.node_indices) {
    nodes.push_back(Node{mesh.nodes.at(id)});
  }

  return nodes;
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
        for (int node_id : element.node_indices) {
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
