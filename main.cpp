#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "parser.h"

int main() {
  std::ifstream file;

  file.open("square.msh");

  Mesh mesh;

  if (file.is_open()) {
    std::string line;
    SectionHeader nodes_header;
    SectionHeader elements_header;

    EntitySectionHeader entities_header;

    EntityPhysicalTags entities;

    int num_el_blocks, num_elements, min_el_tag, max_el_tag;

    while (std::getline(file, line)) {
      // ~~~~~~~~~
      // read entities
      // ~~~~~~~~~
      if (line == "$Entities") {
        read_entity_block_header(file, entities_header);
        entities = read_entities(file, entities_header);
      }

      // ~~~~~~~~~
      // read nodes
      // ~~~~~~~~~
      else if (line == "$Nodes") {
        read_block_header(file, nodes_header);
        read_nodes(file, mesh, nodes_header);
      }

      // ~~~~~~~~~
      // read elements
      // ~~~~~~~~~
      else if (line == "$Elements") {
        read_block_header(file, elements_header);
        read_elements(file, mesh, elements_header, entities);
      }
    }
  }

  file.close();

  print_mesh(mesh);

  return 0;
}
