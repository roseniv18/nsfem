#ifndef POISSON_H
#define POISSON_H

#include <vector>
#include "assemble/assemble.h"
#include "linalg/conjugate_gradient.h"
#include "math/function.h"
#include "mesh/mesh.h"

class Poisson {
 public:
  Poisson(const Mesh& mesh, STFunction f_func, STFunction dir_func);
  std::vector<double> solve(const Mesh& mesh);

 private:
  STFunction f_func;
  STFunction dir_func;

  std::vector<double> rhs_vec;
};

#endif
