#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "parser.h"

int main() {
  std::ifstream file;

  file.open("square.msh");

  std::vector<Node> nodes;
  std::vector<Element> elements;

  if (file.is_open()) {
    std::string line;
    SectionHeader nodes_header;
    SectionHeader elements_header;
    int num_el_blocks, num_elements, min_el_tag, max_el_tag;

    while (std::getline(file, line)) {
      // ~~~~~~~~~
      // read nodes
      // ~~~~~~~~~
      if (line == "$Nodes") {
        read_block_header(file, nodes_header);
        read_nodes(file, nodes, nodes_header);
      }

      // ~~~~~~~~~
      // read elements
      // ~~~~~~~~~
      else if (line == "$Elements") {
        read_block_header(file, elements_header);
        read_elements(file, elements, elements_header);
      }
    }
  }

  file.close();

  // ~~~~~~~
  // print stored nodes and elements
  // ~~~~~~~
  std::cout << "Stored " << nodes.size() << " nodes\n";
  for (const auto& n : nodes) {
    std::cout << n.tag << ": (" << n.x << ", " << n.y << ", " << n.z << ")\n";
  }

  std::cout << "Stored " << elements.size() << " elements\n";
  for (const auto& e : elements) {
    std::cout << e.tag << "\n";
  }

  return 0;
}
