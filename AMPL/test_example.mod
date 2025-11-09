param n; # number of variables
param m; # number of constraints

set N:={1..n};
set M:={1..m};

param A{M, N};
param b{M};
param c{N};
var x{N} >= 0 ;

minimize f: sum{i in N} x[i]*c[i];

subject to Constraint1 {j in M}: 
    sum{i in N} A[j,i]*x[i] = b[j];