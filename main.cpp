#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "parser.h"

int main() {
  std::ifstream file;

  file.open("square.msh");

  if (!file.is_open()) {
    std::cerr << "Could not open square.msh\n";
    return 1;
  }

  std::string line;
  EntitySectionHeader entities_header;
  EntityPhysicalTags entities;

  // entities must exists before elements are parsed
  while (std::getline(file, line)) {
    if (line == "$Entities") {
      read_entity_block_header(file, entities_header);
      entities = read_entities(file, entities_header);
      break;
    }
  }

  Mesh mesh = read_mesh(file, entities);

  file.close();

  print_mesh(mesh);

  return 0;
}
