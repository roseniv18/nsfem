#ifndef PARSER_H
#define PARSER_H

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Node {
  int tag;
  double x;
  double y;
  double z;  // msh files store z coordinate even for 2D meshes
};

struct Element {
  int dim;
  int entity_tag;
  std::vector<int> physical_tags;  // gmsh allows multiple physical groups
  int type;
  std::vector<int> node_indices;
};

struct EntityPhysicalTags {
  std::unordered_map<int, std::vector<int>> points;
  std::unordered_map<int, std::vector<int>> curves;
  std::unordered_map<int, std::vector<int>> surfaces;
  std::unordered_map<int, std::vector<int>> volumes;
};

struct Mesh {
  std::vector<Node> nodes;
  std::vector<Element> elements;
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

std::vector<int> get_physical_tags(const EntityPhysicalTags& entities,
                                   int dim,
                                   int tag);

Mesh read_mesh(std::ifstream& file, const EntityPhysicalTags& entities);

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

void read_entity_block_header(std::ifstream& file, EntitySectionHeader& h);

int nodes_per_element(int type);

void print_mesh(const Mesh& mesh);

#endif
