#ifndef FINITE_ELEMENT_H
#define FINITE_ELEMENT_H

#include <vector>

enum class DOFLoc { Vertex, Edge, Interior };

/**
 * Represents a degree of freedom
 * local_index: the local index of the DOF for the specific finite element
 * location: whether the DOF lives on a vertex, edge or in the interior
 * entity_index: the local index of the edge / vertex (0,1,2)
 */
struct DOF {
  int local_index;
  DOFLoc location;
  int entity_index;
};

/**
 * An abstraction class (interface) for a finite element object
 * Note that the functions are pure virtual (=0)
 * Represents a reference element (2D simplex / triangle) ((0,0), (1,0), (0,1))
 */
class FE {
 public:
  // Get the number of degrees of freedom
  virtual int get_ndofs() const = 0;
  // Get the order of the finite element
  virtual int get_order() const = 0;
  // Get the degrees of freedom
  virtual std::vector<DOF> get_dofs() const = 0;
  // Evaluate basis (shape) functions at given reference point
  virtual std::vector<double> evaluate_bfs(double xi, double eta) const = 0;
  // Evaluate the gradients of the basis (shape) functions at given reference
  // point
  virtual std::vector<std::vector<double>> evaluate_grad_bfs(
      double xi,
      double eta) const = 0;
  // Evaluate basis (shape) functions at list of quadrature nodes
  virtual std::vector<std::vector<double>> bfs_at_quad(
      std::vector<std::vector<double>> quad_nodes) const = 0;
  // Evaluate the gradients of the basis (shape) functions at given list
  // of quadrature nodes
  virtual std::vector<std::vector<double>> grad_bfs_at_quad(
      std::vector<std::vector<double>> quad_nodes) const = 0;
};

#endif
