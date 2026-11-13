SetFactory("Built-in");

// ====================
// Geometry parameters
// ====================

L 		= 2.2;
H 		= 0.41;

xc 		= 0.2;
yc 		= 0.2;
R 		= 0.05;

// Initial mesh sizes
lc		= 0.025;
lc_cyl 	= 0.01;

// ====================
// Outer rectangle
// ====================
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// Boundary curves
Line(1) = {1, 2}; // Bottom
Line(2) = {2, 3}; // Right
Line(3) = {3, 4}; // Top
Line(4) = {4, 1}; // Left

// Outer loop
Curve Loop(1) = {1, 2, 3, 4};

// ====================
// Circular cylinder
// ====================

// Center
Point(5) = {xc, yc, 0, lc_cyl};

// Four points on circle
Point(6) = {xc - R, yc, 0, lc_cyl}; // Left
Point(7) = {xc, yc - R, 0, lc_cyl}; // Bottom
Point(8) = {xc + R, yc, 0, lc_cyl}; // Right
Point(9) = {xc, yc + R, 0, lc_cyl}; // Top

// Four circular arcs
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Inner loop (hole)
Curve Loop(2) = {-8, -7, -6, -5};


// ====================
// Fluid domain
// ====================

Plane Surface(1) = {1, 2};

// ====================
// Physical groups
// ====================

Physical Curve("Walls", 101) = {1, 3};
Physical Curve("Outlet", 102) = {2};
Physical Curve("Inlet", 103) = {4};

Physical Curve("Cylinder", 104) = {5, 6, 7, 8};

Physical Surface("Domain", 201) = {1};


// ====================
// Mesh settings
// ====================

Mesh.ElementOrder = 1;
Mesh.SecondOrderIncomplete = 0;

// Generate mesh
Mesh 2;
