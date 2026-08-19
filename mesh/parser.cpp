#include "parser.h"
#include "helpers/helpers.h"

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

void read_entity_block_header(std::ifstream& file, EntitySectionHeader& e) {
  file >> e.num_points >> e.num_curves >> e.num_surfaces >> e.num_volumes;
}

std::unordered_map<int, PhysicalGroup> read_physical_names(
    std::ifstream& file) {
  int num_groups{};
  file >> num_groups;

  std::unordered_map<int, PhysicalGroup> pgs;

  for (int i = 0; i < num_groups; i++) {
    int dim{}, tag{};
    std::string name{};

    file >> dim >> tag >> name;

    PhysicalGroup pg{};
    pg.dim = dim;
    // .msh stores the names surrounded by quotes
    pg.name = strip_quotes(name);

    pgs[tag] = pg;
  }

  return pgs;
}

Mesh read_mesh(std::ifstream& file,
               const EntityPhysicalTags& entities,
               const std::unordered_map<int, PhysicalGroup>& physical_groups) {
  Mesh mesh{};
  std::unordered_map<int, std::size_t> node_index{};
  std::string line{};

  mesh.physical_groups = physical_groups;

  while (std::getline(file, line)) {
    if (line == "$Nodes") {
      SectionHeader nodes_header;
      read_block_header(file, nodes_header);
      read_nodes(file, mesh, node_index, nodes_header);
    } else if (line == "$Elements") {
      SectionHeader elements_header;
      read_block_header(file, elements_header);
      read_elements(file, mesh, node_index, elements_header, entities);
      break;  // nothing after $Elements matters here
    }
  }

  return mesh;
}

EntityPhysicalTags read_entities(std::ifstream& file, EntitySectionHeader& eh) {
  EntityPhysicalTags entities{};

  // points
  for (int p = 0; p < eh.num_points; p++) {
    int tag{};
    double x{}, y{}, z{};
    int num_phys{};

    file >> tag >> x >> y >> z >> num_phys;

    std::vector<int> phys_tags(num_phys);

    for (int j = 0; j < num_phys; j++) {
      file >> phys_tags[j];
    }

    entities.points[tag] = phys_tags;
  }

  // curves, surfaces, volumes

  auto read_high_dimensional = [&](auto& map, int count) {
    for (int i = 0; i < count; i++) {
      int tag{};

      double xmin{}, ymin{}, zmin{};
      double xmax{}, ymax{}, zmax{};

      int num_phys{};

      file >> tag >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax >> num_phys;

      std::vector<int> phys_tags(num_phys);

      for (int j = 0; j < num_phys; j++) {
        file >> phys_tags[j];
      }

      map[tag] = phys_tags;

      // skip bounding entities
      int num_bounding{};

      file >> num_bounding;

      int dummy{};

      for (int j = 0; j < num_bounding; j++) {
        file >> dummy;
      }
    }
  };

  read_high_dimensional(entities.curves, eh.num_curves);
  read_high_dimensional(entities.surfaces, eh.num_surfaces);
  read_high_dimensional(entities.volumes, eh.num_volumes);

  return entities;
}

std::vector<int> get_physical_tags(const EntityPhysicalTags& entities,
                                   int dim,
                                   int tag) {
  switch (dim) {
    case 0:
      return entities.points.at(tag);

    case 1:
      return entities.curves.at(tag);

    case 2:
      return entities.surfaces.at(tag);

    case 3:
      return entities.volumes.at(tag);
  }

  throw std::runtime_error("Invalid dimension");
}

std::vector<Node> get_element_nodes(const Element& element, const Mesh& mesh) {
  std::vector<Node> nodes{};
  nodes.reserve(element.node_indices.size());

  for (int id : element.node_indices) {
    nodes.push_back(Node{mesh.nodes.at(id)});
  }

  return nodes;
}

std::unordered_set<int> get_dirichlet_nodes(const Mesh& mesh) {
  std::unordered_set<int> dirichlet_nodes{};

  for (const auto& element : mesh.elements) {
    for (int pt : element.physical_tags) {
      const PhysicalGroup physical_group = mesh.physical_groups.at(pt);

      //   std::cout << physical_group.name << '\n';

      if (element.type == 1 && physical_group.name == "Dirichlet") {
        // std::cout << "DIRICHLET EDGE FOUND\n";
        // std::cout << "nodes: ";

        // for (int node : element.node_indices)
        //   std::cout << node << ' ';

        // std::cout << '\n';

        dirichlet_nodes.insert(element.node_indices.begin(),
                               element.node_indices.end());

        // std::cout << "set now: ";

        // for (int node : dirichlet_nodes)
        //   std::cout << node << ' ';

        // std::cout << "\n\n";
      }
    }
  }

  return dirichlet_nodes;
}

std::unordered_map<int, double> get_dirichlet_values(
    const Mesh& mesh,
    const std::unordered_set<int>& dirichlet_nodes,
    std::function<double(const Point2D&)> fn) {
  std::unordered_map<int, double> values;

  for (int node_id : dirichlet_nodes) {
    const auto& node = mesh.nodes.at(node_id);
    const Point2D pt{node.x, node.y};

    values[node_id] = fn(pt);
  }

  return values;
}

void read_nodes(std::ifstream& file,
                Mesh& mesh,
                std::unordered_map<int, std::size_t>& node_index,
                SectionHeader& nh) {
  // reserve memory for storing the nodes
  mesh.nodes.reserve(nh.num_objects);

  for (int i = 0; i < nh.num_blocks; i++) {
    int entity_dim{}, entity_tag{}, parametric{}, num_bnodes{};

    // read the header for the block
    file >> entity_dim >> entity_tag >> parametric >> num_bnodes;

    // read node tags into temporary buffer
    std::vector<int> tags(num_bnodes);
    for (int tag = 0; tag < num_bnodes; tag++) {
      file >> tags[tag];
    }

    // read node coordinates and pair them with the tags
    for (int coord = 0; coord < num_bnodes; coord++) {
      Node n{};
      n.tag = tags[coord];
      file >> n.x >> n.y >> n.z;

      node_index[n.tag] = mesh.nodes.size();

      mesh.nodes.push_back(n);
    }
  }
}

void read_elements(std::ifstream& file,
                   Mesh& mesh,
                   std::unordered_map<int, std::size_t>& node_index,
                   SectionHeader& eh,
                   const EntityPhysicalTags& entities) {
  // reserve memory for storing the elements
  mesh.elements.reserve(eh.num_objects);

  for (int i = 0; i < eh.num_blocks; i++) {
    int entity_dim{}, entity_tag{}, element_type{}, num_belements{};
    // read the header for the block
    file >> entity_dim >> entity_tag >> element_type >> num_belements;

    int n_nodes = nodes_per_element(element_type);

    // read element data
    for (int element = 0; element < num_belements; element++) {
      Element el{};

      file >> el.element_tag;
      el.dim = entity_dim;
      el.type = element_type;

      el.physical_tags = get_physical_tags(entities, entity_dim, entity_tag);

      // loop through nodes
      for (int i = 0; i < n_nodes; i++) {
        int node_tag;
        file >> node_tag;
        // node position in mesh
        // note this is not the same as its tag!
        int node_pos = node_index.at(node_tag);
        el.node_indices.push_back(node_pos);  // tag -> position
      }

      mesh.elements.push_back(el);
    }
  }
}

void print_mesh(const Mesh& mesh) {
  std::cout << "\nNodes\n";
  std::cout << "-----\n";

  for (auto& n : mesh.nodes) {
    std::cout << n.tag << ": " << n.x << " " << n.y << " " << n.z << "\n";
  }

  std::cout << "\nPhysical groups:\n";

  for (const auto& [tag, pg] : mesh.physical_groups) {
    std::cout << "tag = " << tag << ", dim = " << pg.dim
              << ", name = " << pg.name << '\n';
  }

  std::cout << "\nElements\n";
  std::cout << "--------\n";

  for (const auto& e : mesh.elements) {
    std::cout << "Element " << e.element_tag << "\n";

    std::cout << "dim = " << e.dim << "\n";

    for (auto pt : e.physical_tags)
      std::cout << "physical = " << pt << "\n";

    std::cout << "nodes: ";

    for (auto id : e.node_indices)
      std::cout << id << " ";

    std::cout << "\n";
    std::cout << "physical tags: ";

    for (int tag : e.physical_tags) {
      std::cout << "    physical tag = " << tag
                << ", name = " << mesh.physical_groups.at(tag).name << '\n';
    }
  }

  std::cout << "\n\n";
}
