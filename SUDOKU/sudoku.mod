set N := 1..9;
set R := 1..9;
set C := 1..9;
set B := 1..3;
param INITIAL{R,C} integer, >=0, <=9; #initial sudoku values
var X{N, R, C} binary; #decision variable
minimize OBJ: 0; #objective function
subject to UNIQUE_N {r in R, c in C}: 
    sum {n in N} X[n,r,c]=1;
subject to UNIQUE_C {n in N, c in C}: 
    sum {r in R} X[n,r,c]=1;
subject to UNIQUE_R {n in N, r in R}: 
    sum {c in C} X[n,r,c]=1;
subject to UNIQUE_B {n in N, i in B, j in B}:
    sum {r in 3*i-2 .. 3*i, c in 3*j-2 .. 3*j} X[n,r,c]=1;
subject to INITIAL_VAL {r in R, c in C: INITIAL[r,c] > 0}: 
    X[INITIAL[r,c],r,c] =1;