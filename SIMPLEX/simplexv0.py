import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

theorical_zero = 0 #-1e-10
def calc_entering_idx(N, B, cN, cB):
    reduced_costs = {}
    entering_idx = None
    min_reduced_cost = theorical_zero

    for j in range(N.shape[1]):
        N_q = N[:, j]
        lambda_T = np.linalg.solve(B.T, cB)
        rN_q = cN[j] - lambda_T @ N_q
        reduced_costs[j] = rN_q
        if rN_q < min_reduced_cost:
            min_reduced_cost = rN_q
            entering_idx = j
    return entering_idx


def update(basis, non_basis, A, c):
    B = A[:, basis]
    cB = c[basis]
    cN = c[non_basis]
    N = A[:, non_basis]
    lambda_T = np.linalg.solve(B.T, cB)
    rN = cN - lambda_T @ N
    return B, N, cB, cN, rN

def find_trivial_BSF(A: np.ndarray, b: np.ndarray):
    print("START FIND TRIVIAL BSF")
    m, n = A.shape
    basis = []
    non_basis = list(range(n))
    
    for i in range(m):
        # Buscar columna que sea el vector canónico e_i
        for j in range(n):
            col = A[:, j]
            if np.allclose(col, np.eye(m)[i]) and j not in basis:
                basis.append(j)
                non_basis.remove(j)
                break
    
    # Si encontramos m columnas linealmente independientes que forman base
    if len(basis) == m:
        B = A[:, basis]
        x_B = np.linalg.solve(B, b)
            
        # Verificar si es factible
        if np.all(x_B >= -1e-10):
            print(f"Base inicial factible encontrada: {basis}")
            print(f"x_B = {x_B}")
            return basis, non_basis
    return None, None

def phase1(c: np.array, A: np.matrix, b: np.matrix, n: int, m: int):
    #Ax+I=b
    print("START PHASE1: Artificial variables")
    B=np.identity(m)
    N= A
    cN= np.zeros(n)
    cB= np.ones(m)
    A_extended= np.concatenate((N, B), axis=1)
    c_extended = np.concatenate((cN, cB))
    basis = list(range(n, n + m))
    non_basis = list(range(n))
    """print("A_extended")
    print(A_extended)
    print("c_extended")
    print(c_extended)
    print("basis:", basis)
    print("non_basis", non_basis)"""
    lambda_T = np.linalg.solve(B.T, cB)
    rN = cN - lambda_T @ N
    xB = np.linalg.solve(B, b)
        
    #entering_idx= calc_entering_idx(N=N, B=B, cN=cN, cB=cB)
    
    while not np.all(rN >= 0):
        #N_q= N[:, entering_idx]
        q = np.argmin(rN)
        N_q= N[:, q]
        entering_var = non_basis[q]

        Y_q = np.linalg.solve(B, N_q)
        if np.all(Y_q <= theorical_zero):
            print("PHASE1: Y_q <=0 >> unbounded")
            print("Yq, q:", non_basis[q])
            print(np.linalg.solve(B, N))
            return None, None  # Problema Fase I ilimitado -> infactible
        
        ratios = []
        for i in range(m):
            if Y_q[i] > theorical_zero:
                ratios.append((xB[i] / Y_q[i], i))
            """else:
                ratios.append((np.inf, i))"""
     
        if not ratios:
            print("PHASE1: not ratios >> unbounded")
            print("ratios", ratios)
            return None, None
        
        _, p = min(ratios)
        leaving_var = basis[p]   
        basis[p] = entering_var
        non_basis[q] = leaving_var
        basis = sorted(basis)
        non_basis = sorted(non_basis)
        B, N, cB, cN, rN= update(basis=basis, non_basis=non_basis, A=A_extended, c=c_extended)
        xB = np.linalg.solve(B, b)
        #entering_idx= calc_entering_idx(N=N, B=B, cN=cN, cB=cB)
    
    obj_value = cB @ xB
    if abs(obj_value) > theorical_zero:
        print("PHASE1: abs(obj_value) > 0  >> Infeasible")
        print("cB: ", cB)
        print("xB: ", xB)
        print("obj_value: ", obj_value)
        return None, None  # Infactible
    

    # Remover variables artificiales de la base si es posible
    new_basis = []
    for i, var in enumerate(basis):
        if var >= n:  # Variable artificial
            # Intentar reemplazarla con una variable original
            B_current = A_extended[:, basis]
            
            replaced = False
            for j in range(n):
                if j not in basis:
                    N_q = A_extended[:, j]
                    Y_q = np.linalg.solve(B_current, N_q.T)
                    if abs(Y_q[i]) > theorical_zero:
                        new_basis.append(j)
                        replaced = True
                        break
            
            if not replaced:
                new_basis.append(var)
        else:
            new_basis.append(var)
    
    basis = new_basis
    #base_indices_original = [i for i in result.base_indices if i < n]
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
    
    """print("A")
    print(A)
    print("c")
    print(c)
    print("b")
    print(b)"""
    basis, non_basis = find_trivial_BSF(A=A, b=b)
    if basis is None:
        # Si no hay base trivial, usar Fase I completa con variables artificiales
        print("END TRIVIAL BSF")
        basis, non_basis = phase1(c=c, A=A, b=b, n=n, m=m)
        print("END PHASE1")

        if basis is None:
            return None, 'infeasible'
    
    print("basis:", basis)
    print("non_basis", non_basis)

    return phase2(A=A, b=b, c=c, basis=basis, non_basis=non_basis, m=m, n=n)
    #return np.zeros(n+m), None



def phase2(A, b, c, basis, non_basis, m, n):
    print("START PHASE2")
    B, N, cB, cN, rN = update(basis=basis, non_basis=non_basis, A=A, c=c)
    xB = np.linalg.solve(B, b)
    print("N")
    print(N)
    print("B")
    print(B)
    print("cN")
    print(cN)
    print("cB")
    print(cB)

    #entering_idx = calc_entering_idx(N=N,B=B, cN=cN, cB=cB)
    while not np.all(rN >= 0):
        #N_q = N[:, entering_idx]
        q = np.argmin(rN)
        N_q= N[:, q]
        entering_var = non_basis[q]
        Y_q = np.linalg.solve(B, N_q)
        
        # Test de razón mínima
        if np.all(Y_q <= theorical_zero):
            print("PHASE2: Y_q <=0")
            print("Yq, q:", non_basis[q])
            print(np.linalg.solve(B, N))            
            return None, 'unbounded'

        ratios = [(xB[i] / Y_q[i], i) for i in range(m) if Y_q[i] > theorical_zero]
        if not ratios:
            print("PHASE2: not ratios >> unbounded")
            print("ratios", ratios)
            return None, None
        
        _, p = min(ratios)
        leaving_var = basis[p]   
        basis[p] = entering_var
        non_basis[q] = leaving_var
        basis = sorted(basis)
        non_basis = sorted(non_basis)
        B, N, cB, cN, rN = update(basis=basis, non_basis=non_basis, A=A, c=c)
        xB = np.linalg.solve(B, b)        

    # rn>= 0 (todos los costes reducidos >= 0) Condición de optimalidad -> se ha encontrado sol
    # Construir solución completa

    x = np.zeros(n+m)
    for i, var in enumerate(basis):
        if var < n:
            x[var] = xB[i]
    return x, 'optimal'
    #"""
    

