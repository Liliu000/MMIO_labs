param n; # number of rows and columns 9x9
param m; # possible integer numbers to fill sudoku 9 
param nn;
set N:={1..n};
set M:={1..m};
set NN:={1..nn}; #subset 3x3
param a{N, N} default 0;
var x{N,N,M} binary;

minimize f: sum{i in N, j in N, k in M} x[i,j,k];
subject to Constraint1 {i in N, j in N}: 
    sum{k in M} x[i,j,k] = 1;

subject to Constraint2 {k in M, j in N}: 
    sum{i in N} x[i,j,k] = 1;

subject to Constraint3 {k in M, i in N}: 
    sum{j in N} x[i,j,k] = 1;

subject to Constraint4 {i in NN, j in NN, k in M}:
    sum{ii in NN, jj in NN} x[ii+nn*(i-1),jj+nn*(j-1),k] = 1;

subject to Given {i in N, j in N: a[i,j] > 0}:
    x[i,j,a[i,j]] = 1;