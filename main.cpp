#include <fstream>
#include <iostream>
#include <sstream>
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

int nodes_per_element(int type);

int main() {
  std::ifstream file;

  file.open("square.msh");

  std::vector<Node> nodes;
  std::vector<Element> elements;

  if (file.is_open()) {
    std::string line;
    int num_n_blocks, num_nodes, min_n_tag, max_n_tag;
    int num_el_blocks, num_elements, min_el_tag, max_el_tag;

    while (std::getline(file, line)) {
      // ~~~~~~~~~
      // read nodes
      // ~~~~~~~~~
      if (line == "$Nodes") {
        // read nodes header
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> num_n_blocks >> num_nodes >> min_n_tag >> max_n_tag;

        // reserve memory for storing the nodes
        nodes.reserve(num_nodes);

        std::cout << "Node Blocks: " << num_n_blocks << '\n';
        std::cout << "Nodes : " << num_nodes << '\n';
        std::cout << "Min tag: " << min_n_tag << '\n';
        std::cout << "Max tag: " << max_n_tag << '\n';

        for (int i = 0; i < num_n_blocks; i++) {
          int entity_dim, entity_tag, parametric, num_bnodes;

          // read the header for the block
          std::getline(file, line);
          std::istringstream hb(line);
          hb >> entity_dim >> entity_tag >> parametric >> num_bnodes;

          // read node tags into temporary buffer
          std::vector<int> tags(num_bnodes);
          for (int tag = 0; tag < num_bnodes; tag++) {
            std::getline(file, line);
            std::istringstream ts(line);
            ts >> tags[tag];
          }

          // read node coordinates and pair them with the tags
          for (int coord = 0; coord < num_bnodes; coord++) {
            std::getline(file, line);
            std::istringstream cs(line);
            Node n;
            n.tag = tags[coord];
            cs >> n.x >> n.y >> n.z;
            nodes.push_back(n);
          }
        }
      }

      // ~~~~~~~~~
      // read elements
      // ~~~~~~~~~
      if (line == "$Elements") {
        // read elements header
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> num_el_blocks >> num_elements >> min_el_tag >> max_el_tag;

        // reserve memory for storing the elements
        elements.reserve(num_elements);

        std::cout << "Element Blocks: " << num_el_blocks << '\n';
        std::cout << "Elements: " << num_elements << '\n';
        std::cout << "Min tag: " << min_el_tag << '\n';
        std::cout << "Max tag: " << max_el_tag << '\n';

        for (int i = 0; i < num_el_blocks; i++) {
          int entity_dim, entity_tag, element_type, num_belements;
          // read the header for the block
          std::getline(file, line);
          std::istringstream hb(line);
          hb >> entity_dim >> entity_tag >> element_type >> num_belements;

          int n_nodes = nodes_per_element(element_type);

          // read element data
          for (int element = 0; element < num_belements; element++) {
            std::getline(file, line);
            std::istringstream es(line);

            Element el;

            es >> el.tag;
            el.dim = entity_dim;
            el.type = element_type;

            // loop through nodes
            for (int i = 0; i < n_nodes; i++) {
              int node_tag;
              es >> node_tag;
              el.node_tags.push_back(node_tag);
            }

            elements.push_back(el);
          }
        }

        break;
      }
    }
  }

  file.close();

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

int nodes_per_element(int type) {
  switch (type) {
    case 1:
      return 2;  // line
    case 2:
      return 3;  // triangle
    case 3:
      return 4;  // quad
    case 4:
      return 4;  // tetrahedron
    case 5:
      return 8;  // hexahedron
                 // ... add more if needed
    default:
      return 0;
  }
}
