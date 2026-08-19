#ifndef PARSER_H
#define PARSER_H

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "geometry/point2d.h"

struct Node {
  int tag;
  double x;
  double y;
  double z;  // msh files store z coordinate even for 2D meshes
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

struct Element {
  int dim;
  int element_tag;
  int type;
  std::vector<int> physical_tags;
  std::vector<int> node_indices;
  std::vector<Node> nodes;
};

/** Entity Physical Tags
 *
 * Store the physical groups associated with each entity.
 * For example: entities.curves[1] = {5} says:
 * Entity 1 (curve) belongs to physical group(s) 5.
 *
 * ? Entities contain more information as seen in .msh files,
 * ? but at the moment we only need the physical tags
 *
 */
struct EntityPhysicalTags {
  std::unordered_map<int, std::vector<int>> points;
  std::unordered_map<int, std::vector<int>> curves;
  std::unordered_map<int, std::vector<int>> surfaces;
  std::unordered_map<int, std::vector<int>> volumes;
};

struct Mesh {
  std::vector<Node> nodes;
  std::vector<Element> elements;
  std::unordered_map<int, PhysicalGroup> physical_groups;
};

struct SectionHeader {
  int num_blocks;
  int num_objects;
  int min_tag;
  int max_tag;
};

struct EntitySectionHeader {
  int num_points;
  int num_curves;
  int num_surfaces;
  int num_volumes;
};

EntityPhysicalTags read_entities(std::ifstream& file,
                                 EntitySectionHeader& entity_header);

// get the physical tags associated with the given entities
std::vector<int> get_physical_tags(const EntityPhysicalTags& entities,
                                   int dim,
                                   int tag);

// get tags of dirichlet nodes
std::unordered_set<int> get_dirichlet_nodes(const Mesh& mesh);

// get values of function at dirichlet nodes
std::unordered_map<int, double> get_dirichlet_values(
    const Mesh& mesh,
    const std::unordered_set<int>& dirichlet_nodes,
    std::function<double(const Point2D&)> fn);

// main function to read .msh
Mesh read_mesh(std::ifstream& file,
               const EntityPhysicalTags& entities,
               const std::unordered_map<int, PhysicalGroup>& physical_groups);

void read_nodes(std::ifstream& file,
                Mesh& mesh,
                std::unordered_map<int, std::size_t>& node_index,
                SectionHeader& node_header);

void read_elements(std::ifstream& file,
                   Mesh& mesh,
                   std::unordered_map<int, std::size_t>& node_index,
                   SectionHeader& element_header,
                   const EntityPhysicalTags& entities);

void read_block_header(std::ifstream& file, SectionHeader& h);

std::unordered_map<int, PhysicalGroup> read_physical_names(std::ifstream& file);

void read_entity_block_header(std::ifstream& file, EntitySectionHeader& h);

int nodes_per_element(int type);

void print_mesh(const Mesh& mesh);

#endif
