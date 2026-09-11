#ifndef MESH_H
#define MESH_H

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

struct Element {
  int dim;
  int element_tag;
  ElementType type;
  std::vector<int> physical_tags;
  std::vector<std::size_t> node_indices;
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
  std::vector<Element> elements;
  std::unordered_map<int, PhysicalGroup> physical_groups;
};

// get the nodes of an element
std::vector<Node> get_element_nodes(const Element& element, const Mesh& mesh);

// get tags of dirichlet nodes
std::vector<bool> get_dirichlet_nodes(const Mesh& mesh);

// get values of function at dirichlet nodes
std::vector<double> get_dirichlet_values(const Mesh& mesh,
                                         const std::vector<bool>& is_dirichlet,
                                         STFunction fn,
                                         double t);

#endif
