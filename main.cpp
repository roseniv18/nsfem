#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "affine.h"
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

  AffineMap am = compute_affine(mesh.elements[4], mesh);

  std::cout << "Affine transformation for triangle 4: " << '\n';
  std::cout << "Jacobian: " << '\n';
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << am.Jacobian[i][j] << '\t';
    }
    std::cout << '\n';
  }

  std::cout << "----------" << '\n';

  std::cout << "Det J: " << '\n';
  std::cout << am.detJ << '\n';

  std::cout << "----------" << '\n';

  std::cout << "Physical gradients: " << '\n';
  for (int i = 0; i < 3; i++) {
    std::cout << "phys_grad " << i << ".x : " << am.phys_grads[i].x << '\n';
    std::cout << "phys_grad " << i << ".y : " << am.phys_grads[i].y << '\n';
  }

  return 0;
}
