#ifndef FINITE_ELEMENT_H
#define FINITE_ELEMENT_H

#include <memory>
#include <vector>
#include "FEM/dof_handler/dof.h"
#include "FEM/mapping/ref_node.h"

enum class FEType { P1, P2 };

enum class ElementType;

/**
 * An abstraction class (interface) for a finite element object
 * Note that the functions are pure virtual (=0)
 * Represents a reference element (2D simplex / triangle) ((0,0), (1,0), (0,1))
 */
class FE {
 public:
  // Type of finite element
  FEType type;
  // Build finite element object of given type
  static std::unique_ptr<FE> build_fe_type(const FEType type);
  // Create a geometry element
  static std::unique_ptr<FE> build_geo_type(const ElementType& type);
  //  Get ref nodes
  virtual std::vector<RefNode> get_ref_nodes() const = 0;
  //   Get DOFs
  virtual std::vector<DOF> get_dofs() const = 0;
  // Get the number of degrees of freedom
  virtual int get_ndofs() const = 0;
  // Get the order of the finite element
  virtual int get_order() const = 0;
  // Evaluate basis (shape) functions at given reference point
  virtual std::vector<double> evaluate_bfs(const RefNode& pt) const = 0;
  // Evaluate the gradients of the basis (shape) functions at given reference
  // point
  virtual std::vector<std::vector<double>> evaluate_grad_bfs(
      const RefNode& pt) const = 0;
  // Evaluate basis (shape) functions at list of quadrature nodes
  virtual std::vector<std::vector<double>> bfs_at_quad(
      const std::vector<RefNode>& quad_nodes) const = 0;

  //   Check if element has edge DOFs
  virtual bool has_edge_dofs() const = 0;
  //   Number of DOFs per edge
  virtual std::size_t n_edge_dofs() const = 0;
};

#endif
