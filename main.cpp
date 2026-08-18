#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "assemble/assemble.h"
#include "geometry/affine.h"
#include "mesh/parser.h"

using std::sin;
using std::numbers::pi;

double func(const Point2D& pt);

int main() {
  std::ifstream file;

  file.open("square.msh");

  if (!file.is_open()) {
    std::cerr << "Could not open square.msh\n";
    return 1;
  }

  std::string line;
  EntitySectionHeader entities_header{};
  EntityPhysicalTags entities{};
  std::unordered_map<int, PhysicalGroup> physical_groups{};

  // entities must exists before elements are parsed
  while (std::getline(file, line)) {
    if (line == "$PhysicalNames") {
      physical_groups = read_physical_names(file);
    } else if (line == "$Entities") {
      read_entity_block_header(file, entities_header);
      entities = read_entities(file, entities_header);
      break;
    }
  }

  Mesh mesh = read_mesh(file, entities, physical_groups);

  file.close();

  print_mesh(mesh);

  AffineMap am = compute_affine(mesh.elements[4]);

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

  std::cout << "----------" << '\n';

  std::cout << "Nodes (coordinates): " << '\n';
  for (int i = 0; i < mesh.elements[4].nodes.size(); i++) {
    Node n = mesh.elements[4].nodes[i];
    std::cout << "(Node  " << i
              << "): "
                 "x = "
              << n.x << ", y = " << n.y << ", z = " << n.z << '\n';
  }

  local_matr ls_matrix = generate_ls_matrix(mesh.elements.at(4));

  std::cout << "----------" << '\n';

  std::cout << "Local stiffness matrix (Poisson): " << '\n';
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << ls_matrix[i][j] << '\t';
    }
    std::cout << '\n';
  }

  std::vector<std::vector<double>> gs_matrix = assemble_gs_matrix(mesh);

  std::cout << "----------" << '\n';

  std::cout << "Global stiffness matrix " << '\n';
  for (int i = 0; i < mesh.nodes.size(); i++) {
    for (int j = 0; j < mesh.nodes.size(); j++) {
      std::cout << gs_matrix.at(i).at(j) << '\t';
    }
    std::cout << '\n';
  }

  std::vector<double> gl_vector = assemble_gl_vector(mesh, func);

  std::cout << "----------" << '\n';

  std::cout << "Global load vector " << '\n';
  for (int i = 0; i < mesh.nodes.size(); i++) {
    std::cout << gl_vector.at(i) << '\t';
    std::cout << '\n';
  }

  std::cout << "----------" << '\n';

  std::cout << "Entity tags for elements: " << '\n';
  for (int i = 0; i < mesh.elements.size(); i++) {
    std::cout << "Element " << i << ": " << mesh.elements.at(i).element_tag
              << '\n';
  }

  auto dirichlet_nodes = get_dirichlet_nodes(mesh);

  std::cout << "Dirichlet nodes: ";

  for (int node : dirichlet_nodes)
    std::cout << node << ' ';

  std::cout << '\n';

  return 0;
}

double func(const Point2D& pt) {
  return pt.y;
}
