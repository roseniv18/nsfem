lms = 0.1;

Point(1) = {0, 0, 0, lms};
Point(2) = {1, 0, 0, lms};
Point(3) = {1, 1, 0, lms};
Point(4) = {0, 1, 0, lms};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

Physical Curve("Bottom") = {1};
Physical Curve("Right") = {2};
Physical Curve("Top") = {3};
Physical Curve("Left") = {4};

Physical Surface("Domain") = {1};

Mesh 2;
