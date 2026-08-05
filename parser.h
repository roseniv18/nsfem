#ifndef PARSER_H
#define PARSER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Node {
  int tag;
  double x;
  double y;
  double z;  // msh files store z coordinate even for 2D meshes
};

struct Element {
  int dim;
  int tag;
  int type;
  std::vector<int> node_tags;
};

struct SectionHeader {
  int num_blocks;
  int num_objects;
  int min_tag;
  int max_tag;
};

struct NodeBlockHeader {
  int entity_dim;
  int entity_tag;
  int parametric;
  int num_nodes;
};

struct ElementBlockHeader {
  int entity_dim;
  int entity_tag;
  int element_type;
  int num_elements;
};

void read_nodes(std::ifstream& file,
                std::vector<Node>& nodes,
                SectionHeader& node_header);

void read_elements(std::ifstream& file,
                   std::vector<Element>& elements,
                   SectionHeader& element_header);

void read_block_header(std::ifstream& file, SectionHeader& h);

int nodes_per_element(int type);

#endif
