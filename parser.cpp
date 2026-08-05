#include "parser.h"

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

void read_block_header(std::ifstream& file, SectionHeader& h) {
  file >> h.num_blocks >> h.num_objects >> h.min_tag >> h.max_tag;
}

void read_nodes(std::ifstream& file,
                std::vector<Node>& nodes,
                SectionHeader& nh) {
  // reserve memory for storing the nodes
  nodes.reserve(nh.num_objects);

  std::cout << "Node Blocks: " << nh.num_blocks << '\n';
  std::cout << "Nodes : " << nh.num_objects << '\n';
  std::cout << "Min tag: " << nh.min_tag << '\n';
  std::cout << "Max tag: " << nh.max_tag << '\n';

  for (int i = 0; i < nh.num_blocks; i++) {
    int entity_dim, entity_tag, parametric, num_bnodes;

    // read the header for the block
    file >> entity_dim >> entity_tag >> parametric >> num_bnodes;

    // read node tags into temporary buffer
    std::vector<int> tags(num_bnodes);
    for (int tag = 0; tag < num_bnodes; tag++) {
      file >> tags[tag];
    }

    // read node coordinates and pair them with the tags
    for (int coord = 0; coord < num_bnodes; coord++) {
      Node n;
      n.tag = tags[coord];
      file >> n.x >> n.y >> n.z;
      nodes.push_back(n);
    }
  }
}

void read_elements(std::ifstream& file,
                   std::vector<Element>& elements,
                   SectionHeader& eh) {
  // reserve memory for storing the elements
  elements.reserve(eh.num_objects);

  std::cout << "Element Blocks: " << eh.num_blocks << '\n';
  std::cout << "Elements: " << eh.num_objects << '\n';
  std::cout << "Min tag: " << eh.min_tag << '\n';
  std::cout << "Max tag: " << eh.max_tag << '\n';

  for (int i = 0; i < eh.num_blocks; i++) {
    int entity_dim, entity_tag, element_type, num_belements;
    // read the header for the block
    file >> entity_dim >> entity_tag >> element_type >> num_belements;

    int n_nodes = nodes_per_element(element_type);

    // read element data
    for (int element = 0; element < num_belements; element++) {
      Element el;

      file >> el.tag;
      el.dim = entity_dim;
      el.type = element_type;

      // loop through nodes
      for (int i = 0; i < n_nodes; i++) {
        int node_tag;
        file >> node_tag;
        el.node_tags.push_back(node_tag);
      }

      elements.push_back(el);
    }
  }
}
