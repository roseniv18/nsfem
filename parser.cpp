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

void read_entity_block_header(std::ifstream& file, EntitySectionHeader& e) {
  file >> e.num_points >> e.num_curves >> e.num_surfaces >> e.num_volumes;
}

EntityPhysicalTags read_entities(std::ifstream& file, EntitySectionHeader& eh) {
  EntityPhysicalTags entities;

  // points
  for (int p = 0; p < eh.num_points; p++) {
    int tag;
    double x, y, z;
    int num_phys;

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
      int tag;

      double xmin, ymin, zmin;
      double xmax, ymax, zmax;

      int num_phys;

      file >> tag >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax >> num_phys;

      std::vector<int> phys_tags(num_phys);

      for (int j = 0; j < num_phys; j++) {
        file >> phys_tags[j];
      }

      map[tag] = phys_tags;

      // skip bounding entities
      int num_bounding;

      file >> num_bounding;

      int dummy;

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

void read_nodes(std::ifstream& file, Mesh& mesh, SectionHeader& nh) {
  // reserve memory for storing the nodes
  mesh.nodes.reserve(nh.num_objects);

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
      mesh.nodes.push_back(n);
    }
  }
}

void read_elements(std::ifstream& file,
                   Mesh& mesh,
                   SectionHeader& eh,
                   const EntityPhysicalTags& entities) {
  // reserve memory for storing the elements
  mesh.elements.reserve(eh.num_objects);

  for (int i = 0; i < eh.num_blocks; i++) {
    int entity_dim, entity_tag, element_type, num_belements;
    // read the header for the block
    file >> entity_dim >> entity_tag >> element_type >> num_belements;

    auto physical_tags = get_physical_tags(entities, entity_dim, entity_tag);
    int n_nodes = nodes_per_element(element_type);

    // read element data
    for (int element = 0; element < num_belements; element++) {
      Element el;

      file >> el.entity_tag;
      el.dim = entity_dim;
      el.type = element_type;
      el.physical_tags = physical_tags;

      // loop through nodes
      for (int i = 0; i < n_nodes; i++) {
        int node_tag;
        file >> node_tag;
        el.node_tags.push_back(node_tag);
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

  std::cout << "\nElements\n";
  std::cout << "--------\n";

  for (auto& e : mesh.elements) {
    std::cout << "Element " << e.entity_tag << "\n";

    std::cout << "dim = " << e.dim << "\n";

    std::cout << "entity = " << e.entity_tag << "\n";

    for (auto pt : e.physical_tags)
      std::cout << "physical = " << pt << "\n";

    std::cout << "nodes: ";

    for (auto id : e.node_tags)
      std::cout << id << " ";

    std::cout << "\n\n";
  }
}
