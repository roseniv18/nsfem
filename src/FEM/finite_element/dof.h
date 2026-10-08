#ifndef DOF_H
#define DOF_H

enum class DOFLoc { Vertex, Edge, Interior };

/**
 * Represents a degree of freedom
 * local_index: the local index of the DOF for the specific finite element
 * location: whether the DOF lives on a vertex, edge or in the interior
 * entity_index: the local index of the edge / vertex (0,1,2)
 *
 * Example for Lagrange P2 triangle with nodes (a,b,c) and edges (ab, bc, ca)
 * To describe the midpoint on edge ab:
 * DOF {4, DOFLoc::Edge, 0};
 */
struct DOF {
  int local_index;
  DOFLoc location;
  int entity_index;
};

#endif
