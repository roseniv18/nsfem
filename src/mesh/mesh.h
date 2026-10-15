#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "math/function.h"

class DOFHandler;
class FE;
class FEMap2D;

enum class ElementType {
  Line2,
  Triangle3,
  Line3,
  Triangle6,
  Triangle9,
  Triangle10,
  Triangle12,
  Triangle15,
  Triangle21
};

constexpr bool is_triangle(const ElementType type) {
  switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
    case ElementType::Triangle9:
    case ElementType::Triangle10:
    case ElementType::Triangle12:
    case ElementType::Triangle15:
    case ElementType::Triangle21:
      return true;

    default:
      return false;
  }
}

constexpr bool is_line(const ElementType type) {
  switch (type) {
    case ElementType::Line2:
    case ElementType::Line3:
      return true;

    default:
      return false;
  }
}

inline constexpr std::string DIRICHLET = "Dirichlet";

struct Node {
  int tag{};
  double x{};
  double y{};
  double z{};  // msh files store z coordinate even for 2D meshes
};

struct Edge {
  std::size_t node_id_1;
  std::size_t node_id_2;

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

// get tags of dirichlet dofs
std::vector<bool> get_dirichlet_dofs(const Mesh& mesh, const DOFHandler& dofh);

// get values of function at dirichlet nodes
std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const FE& fe,
                                         const DOFHandler& dofh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         double t);

#endif
