import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

theorical_zero = -1e-10
result = {
        "x": None,
        "obj_value": None,
        "status": None,
        "iterations_phase1": 0,
        "iterations_phase2": 0,
        "error": None,
    }

def update(basis, non_basis, A, c):
    B = A[:, basis]
    N = A[:, non_basis]
    cB = c[basis]
    cN = c[non_basis]
    B_inv = np.linalg.inv(B)
    lambda_T = cB @ B_inv
    rN = cN - lambda_T @ N
    return B, N, rN

def find_trivial_BSF(A: np.ndarray, b: np.ndarray):
    print("START FIND TRIVIAL BSF")
    m, n = A.shape
    basis = []
    non_basis = list(range(n))
    
    #buscamos m columnas
    for i in range(m):
        # Buscar columna que sea el vector canónico e_i
        for j in range(n):
            col = A[:, j]
            if np.allclose(col.flatten(), np.identity(m)[i], atol=1e-6) and j not in basis:
                basis.append(j)
                non_basis.remove(j)
                break
    
    # Si encontramos m columnas linealmente independientes que forman base
    if len(basis) == m:
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
        x_B = B_inv @ b
            
        # Verificar si es factible
        if np.all(x_B >= theorical_zero):
            print(f"Base inicial factible encontrada: {basis}")
            print(f"x_B = {x_B}")
            return basis, non_basis
    #modificar , phase1 solo var a que necesite para len(basis == m)
    return None, None


def simplex_loop(A, b, c, basis, non_basis, phase):
    B, N, rN= update(basis=basis, non_basis=non_basis, A=A, c=c)
    m, nm = A.shape
    B_inv = np.linalg.inv(B)
    xB = B_inv @ b
    iterations = 0
    while not np.all(rN >= 0):
        #N_q= N[:, entering_idx]
        q = np.argmin(rN)
        N_q= N[:, q]
        entering_var = non_basis[q]

        B_inv = np.linalg.inv(B)
        Y_q = B_inv @ N_q
        if np.all(Y_q <= theorical_zero):
            d = np.zeros(nm)
            d[entering_var] = 1
            d[basis] = -1* Y_q.T
            x = np.zeros(nm)
            x[basis] = xB.T
            result["error"] =  f"{phase}: Y_q <=0 \n"+f"Yq, q: {non_basis[q]} \n" +f"direccion: {x} +alpha*{d}"
            result["status"] = f"{phase}: unbounded, no feasible"
            return None, None, None
        
        ratios = []
        for i in range(m):
            if Y_q[i] > 1e-9:
                ratios.append((xB[i] / Y_q[i], i))
     
        if not ratios:
            d = np.zeros(nm)
            d[entering_var] = 1
            d[basis] = -1* Y_q.T
            x = np.zeros(nm)
            x[basis] = xB.T
            result["error"] = f"{phase}: not ratios with q={non_basis[q]}\n"+ f"ratios: {ratios} \n"+ f"direccion: {x} +alpha*{d}"
            result["status"] = "unbounded, , no feasible"
            return None, None, None
                
        _, p = min(ratios)
        leaving_var = basis[p]   
        basis[p] = entering_var
        non_basis[q] = leaving_var
        basis = sorted(basis)
        non_basis = sorted(non_basis)
        B, N, rN= update(basis=basis, non_basis=non_basis, A=A, c=c)
        detB = np.linalg.det(B)
        B_inv = np.linalg.inv(B)
        xB = np.linalg.solve(B, b)
        iterations+=1
    if(phase == "PHASE1"):
        result["iterations_phase1"] = iterations
    else:
        result["iterations_phase2"] = iterations
    return basis, non_basis, xB

def phase1(A: np.matrix, b: np.matrix):
    m, n = A.shape
    #Ax+I=b
    print("START PHASE1: Artificial variables")
    A_extended= np.hstack((A, np.identity(m)))
    c_extended = np.hstack((np.zeros(n), np.ones(m)))
    basis = list(range(n, n + m))
    non_basis = list(range(n))
        
    basis, non_basis, xB = simplex_loop(A=A_extended, b=b, c=c_extended, basis=basis, non_basis=non_basis, phase="PHASE1")
    cB = c_extended[basis]
    obj_value = cB @ xB
    if abs(obj_value) > 1e-6:
        x = np.zeros(n+m)
        x[basis] = xB.T
        result["error"] = f"PHASE1: abs(obj_value) > 0 \n"+ f"x: {x} \n"+ f"artificial variables: {x[-m:]}\n"+f"obj_value: {obj_value} \n"
        result["status"] = "infeasible"
        return None, None 
    
    # Filter to keep only original variables
    basis = [i for i in basis if i < n]
    if len(basis) < m:
        #should not get here
        result["error"] = f"PHASE1: Cannot form valid basis with original variables"
        result["status"] = "infeasible"
        return None, None
    non_basis = [j for j in range(n) if j not in basis]
    return basis, non_basis

def simplex(c: np.array, A: np.matrix, b: np.matrix):
    print("START SIMPLEX")
    n= len(c)
    m, t= b.shape
    if np.any(b < 0):
        raise ValueError("Se requiere que b >= 0")
    num_rows, num_cols = A.shape  
    if num_cols != n:
        raise ValueError("matrix A must have as many columns as elements of x vector")
    if num_rows != m:
        raise ValueError("matrix A must have as many rows as elements of b vector")
    
    basis, non_basis = find_trivial_BSF(A=A, b=b)
    if basis is None:
        # Si no hay base trivial, usar Fase I completa con variables artificiales
        print("END TRIVIAL BSF")
        basis, non_basis = phase1(A=A, b=b)
        print("END PHASE1")
        if basis is None:
            return result

    print("START PHASE2")
    basis, non_basis, xB = simplex_loop(A=A, b=b, c=c, basis=basis, non_basis=non_basis, phase="PHASE2")
    # rn>= 0 (todos los costes reducidos >= 0) Condición de optimalidad -> se ha encontrado sol
    # Construir solución completa
    if basis is not None:
        x = np.zeros(n)
        x[basis] = xB.T
        x[np.isclose(x, 0.0, atol=1e-10)] = 0.0
        obj_value = c @ x
        result["x"] = x
        result["obj_value"] = obj_value
        result["status"] = 'optimal'
    return result
    

