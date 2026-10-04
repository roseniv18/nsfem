#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "math/function.h"

enum class ElementType { Line2, Triangle3, Line3, Triangle6 };

struct Node {
  int tag{};
  double x{};
  double y{};
  double z{};  // msh files store z coordinate even for 2D meshes
};

struct Edge {
  int node_id_1;
  int node_id_2;

  bool operator<(const Edge& other) const {
    int min_this = std::min(node_id_1, node_id_2);
    int max_this = std::max(node_id_1, node_id_2);

    int min_other = std::min(other.node_id_1, other.node_id_2);
    int max_other = std::max(other.node_id_1, other.node_id_2);

    if (min_this != min_other)
      return min_this < min_other;

    return max_this < max_other;
  }
};

struct Element {
  int dim;
  int element_tag;
  ElementType type;
  std::vector<int> physical_tags;
  //   store the global ids of the nodes
  std::vector<std::size_t> node_ids;
  //   store the global ids of the edges
  std::vector<std::size_t> edge_ids;
};

/** Entity Physical Groups
 *
 * gmsh allows a single entity to belong to multiple physical groups.
 * Physical groups can be used to describe mathematical notions like boundary
 * conditions.
 *
 * ? Note 'tag' is omitted.
 * ? This is because in the unordered_map of physical groups,
 * ? the tag is the given key of the map
 *
 */
struct PhysicalGroup {
  int dim;
  std::string name;
};

struct Mesh {
  std::vector<Node> nodes;
  std::vector<Edge> unique_edges;
  std::vector<Element> elements;
  std::unordered_map<int, PhysicalGroup> physical_groups;
};

// get the nodes of an element
std::vector<Node> get_element_nodes(const Element& element, const Mesh& mesh);

// get the unique edges of the mesh elements
void build_unique_edges_list(Mesh& mesh);

// get tags of dirichlet nodes
std::vector<bool> get_dirichlet_nodes(const Mesh& mesh);

// get values of function at dirichlet nodes
std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         double t);

#endif
