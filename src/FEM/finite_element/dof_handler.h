#ifndef DOF_HANDLER_H
#define DOF_HANDLER_H

#include <memory>
#include <vector>
#include "finite_element.h"

struct Mesh;

class DOFHandler {
 public:
  DOFHandler(const Mesh& mesh, const FE& element);

  std::size_t get_global_ndofs() const;
  //   Returns the global DOF positions for the given element
  std::vector<std::size_t> get_element_dof_indices(
      std::size_t element_id) const;
  //   Determine the number of global degrees of freedom based on the finite
  //   element type
  void generate_global_ndofs(const Mesh& mesh, const FEType& fe_type);

 private:
  std::size_t global_ndofs{};
  std::vector<int> local_to_global{};
  std::vector<std::vector<std::size_t>> element_dof_indices;
};

#endif
