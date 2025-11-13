import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError


#due to matrix and vector operations, 
# sometimes decimals make it impossible to compare with exact numbers
# thats why we needed to add a certain tolerancy when comparing to 0
ROUND_TOL = 1e-7
ZERO_TOL = 1e-9
OBJ_TOL = 1e-12

#structure where to store the results of simplex
result = {
        "x": None,                  #solution vector
        "obj_value": None,          #final value of objective function
        "status": None,             #status in which simplex ended
        "iterations_phase1": 0,     #number of iterations of phase I
        "iterations_phase2": 0,     #number of iterations of 
        "error": None,              #extra information in case simplex did not end with optimal solution
    }


#input: basis, non_basis, A, c
#output: B, N, rN
#Function:
# -Recalculates basic matrix (B) and non basic matrix (N)  
# -Calculates reduced costs (rN)
def update(basis, non_basis, A, c):
    B = A[:, basis]             #Calculates B using basic variables
    N = A[:, non_basis]         #Calculates N using non basic variables
    cB = c[basis]               #Calculates cB using basic variables
    cN = c[non_basis]           #Calculates cN using non basic variables
    B_inv = np.linalg.inv(B)    #Calculates inverse of B
    lambda_T = cB @ B_inv       #Calculates solution to the dual problem  
    rN = cN - lambda_T @ N      #calculates reduced costs 
    return B, N, rN

#input: A, c
#output: basis, non_basis
#Function:
# -Finds trivial basic feasible solution 
def find_trivial_BFS(A: np.ndarray, b: np.ndarray):
    print("START FIND TRIVIAL BSF")
    #Gets the dimensions of the problem: n variables with m constraints
    m, n = A.shape                     
    #Initializes basic and non basic variables vectors 
    basis = []                         
    non_basis = list(range(n))
    
    #Loop to find the m variables' columns that form an identity matrix
    for i in range(m):
        #Loop to find a variable whose column can form an identity matrix 
        for j in range(n):
            col = A[:, j]
            #if Aj equals Ii then variable j can be basic variable
            is_identity_column = np.allclose(col.flatten(), np.identity(m)[i], atol=ROUND_TOL)
            if is_identity_column and j not in basis:
                basis.append(j)
                non_basis.remove(j)
                break

    #if we found m variables to form base, we already have a BFS
    if len(basis) == m:
        B = A[:, basis]
        x_B = np.linalg.solve(B, b)
            
        #Verify feasibility  (This should be always true)
        if np.all(x_B >= -ZERO_TOL):
            print(f"Base inicial factible encontrada: {basis}")
            print(f"x_B = {x_B}")
            return basis, non_basis

    #If weu reach this point, there's no basic feasible solution
    return None, None

#input: A, c
#output: basis, non_basis
#Function:
# -Loop to find an optimal or unbounded solution
def simplex_loop(A, b, c, basis, non_basis, phase):
    #initialize all the needed parameters: B, N, xB, rN
    B, N, rN= update(basis=basis, non_basis=non_basis, A=A, c=c)
    m, nm = A.shape
    xB = np.linalg.solve(B, b)
    iterations = 0

    #loop to find optimal solution.
    while not np.all(rN >= -OBJ_TOL):
        q = np.argmin(rN)               #Finds index of variable with min reduced cost 
        N_q= N[:, q]                    #Finds pivot column
        entering_var = non_basis[q]     #Finds variable that will eneter the base
        Y_q = np.linalg.solve(B, N_q)   #Calculates the transformed pivot column

        #if basic variables change negatively per unit of the entering variable 
        # then there is no specific minimum to reach
        if np.all(Y_q <= -ZERO_TOL):       
            #the problem doesn't have optimal solution, but an unlimited range of solutions in direction d
            d = np.zeros(nm)
            d[entering_var] = 1
            d[basis] = -1* Y_q.T
            x = np.zeros(nm)
            x[basis] = xB.T
            result["error"] =  f"{phase}: Y_q <=0 \n"+f"Yq, q: {non_basis[q]} \n" +f"direccion: {x} +alpha*{d}"
            result["status"] = f"{phase}: unbounded, no feasible"
            return None, None, None
        
        #loop to find the minimum ratio in order to know which variable leaves the base
        ratios = []
        for i in range(m):
            if Y_q[i] > ZERO_TOL:
                ratios.append((xB[i] / Y_q[i], i))

        #if all the values of the transformed pivot column are non positive, 
        # then there is no specific minimum to reach
        if not ratios:
            #the problem doesn't have optimal solution, 
            # but an unlimited range of solutions in direction d
            d = np.zeros(nm)
            d[entering_var] = 1
            d[basis] = -1* Y_q.T
            x = np.zeros(nm)
            x[basis] = xB.T
            result["error"] = f"{phase}: not ratios with q={non_basis[q]}\n"+ f"ratios: {ratios} \n"+ f"direccion: {x} +alpha*{d}"
            result["status"] = "unbounded, no feasible"
            return None, None, None

        _, p = min(ratios)          #find index of variable that will leave base
        leaving_var = basis[p]      #find variable that will leave base
        basis[p] = entering_var     #replace leaving variable for entering variable
        non_basis[q] = leaving_var  #replace entering variable for leaving variable
        #sort basic variables and non basic variables vectors 
        basis = sorted(basis)       
        non_basis = sorted(non_basis)

        #update all the needed parameters: B, N, xB, rN
        B, N, rN= update(basis=basis, non_basis=non_basis, A=A, c=c)
        detB = np.linalg.det(B)
        xB = np.linalg.solve(B, b)
        iterations+=1
    
    #loop reached a feasible and optimal solution
    if(phase == "PHASE1"):
        result["iterations_phase1"] = iterations
    else:
        result["iterations_phase2"] = iterations
    return basis, non_basis, xB

#input: A, c
#output: basis, non_basis
#Function:
# -Finds a basic feasible solution addind artificial variables
def phase1(A: np.matrix, b: np.matrix):
    #new optimization problem:
    #min sum a
    #Ax+I=b
    m, n = A.shape
    print("START PHASE1: Artificial variables")
    #initialize all the needed parameters: A, c, basis, non_basis
    A_extended= np.hstack((A, np.identity(m)))
    c_extended = np.hstack((np.zeros(n), np.ones(m)))
    basis = list(range(n, n + m))
    non_basis = list(range(n))
    
    #loop until find which combination of non artificial variables build the base of the problem 
    basis, non_basis, xB = simplex_loop(A=A_extended, b=b, c=c_extended, basis=basis, non_basis=non_basis, phase="PHASE1")
    cB = c_extended[basis]
    obj_value = cB @ xB

    #if there is no combination of non artificial variables that form a base
    # it means the problem has no solution, it is infeasible
    if abs(obj_value) >= OBJ_TOL:
        x = np.zeros(n+m)
        x[basis] = xB.T
        result["error"] = f"PHASE1: abs(obj_value) > 0 \n"+ f"x: {x} \n"+ f"artificial variables: {x[-m:]}\n"+f"obj_value: {obj_value} \n"
        result["status"] = "infeasible"
        return None, None 
    
    # Filter to keep only original variables
    basis = [i for i in basis if i < n]
    #if there is no combination of non artificial variables that form a base
    # it means the problem has no solution, it is infeasible
    if len(basis) < m:
        #should not get here
        result["error"] = f"PHASE1: Cannot form valid basis with original variables"
        result["status"] = "infeasible"
        return None, None

    #if we reach here, there is a combination of non artificial variables that form a base
    non_basis = [j for j in range(n) if j not in basis]
    return basis, non_basis

#input: A, b, c
#output: result
#Function:
# -implements simplex algorithm
def simplex(c: np.array, A: np.matrix, b: np.matrix):
    print("START SIMPLEX")
    n= len(c)
    m, t= b.shape
    if np.any(b < 0):
        raise ValueError("vector b must be non negative: b >= 0")
    num_rows, num_cols = A.shape  
    if num_cols != n:
        raise ValueError("matrix A must have as many columns as elements of x vector")
    if num_rows != m:
        raise ValueError("matrix A must have as many rows as elements of b vector")
    #initialize result
    result["x"]=result["obj_value"]=result["status"]=result["error"]=None 
    result["iterations_phase1"]=result["iterations_phase2"]=0
    
    #First we must find if problem has a trivial basic feasible solution
    basis, non_basis = find_trivial_BFS(A=A, b=b)
    print("END TRIVIAL BSF")
    if basis is None:
        # If there is no trivial basic feasible solution, 
        # execute Fase I to find basic feasible solution 
        basis, non_basis = phase1(A=A, b=b)
        print("END PHASE1")
        if basis is None:
            # If there is no basic feasible solution, then problem is infeasible
            return result

    #A basic feasible solution was found, 
    # so we have to find an optimal solution
    print("START PHASE2")
    basis, non_basis, xB = simplex_loop(A=A, b=b, c=c, basis=basis, non_basis=non_basis, phase="PHASE2")
    if basis is not None and xB is not None:
        #We found the solution of the problem
        x = np.zeros(n)
        x[basis] = xB.T
        #due to decimals, sometimes the solution was -0.0000000...1
        x[np.isclose(x, 0.0, atol=ROUND_TOL)] = 0.0 #forces to round to +0
        np.round(x, 7)                              #round to 7 digits
        obj_value = c @ x
        result["x"] = x
        result["obj_value"] = round(number=obj_value, ndigits=7) #round to 7 digits
        result["status"] = 'optimal'
    return result
    