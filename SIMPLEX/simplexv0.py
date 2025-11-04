import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

def phase1(c: np.array, A: np.matrix):
    m, n= A.shape
    #Ax+I=b
    B=np.identity(m)
    N= A
    cn= np.ones(n)
    cb= np.zeros(m)
    A= np.concatenate((N, B), axis=1)
    c = np.concatenate((cn, cb))
    basis = list(range(n, n + m))
    non_basis = list(range(n))
    return A,c, basis, non_basis


def calc_rn(cn: np.array, cb: np.array, invB: np.matrix, N: np.matrix):
    print("op1",cb*invB*N)
    print("op2",cb@invB@N)
    return cn-cb@invB@N

def simplex(c: np.array, A: np.matrix, b: np.matrix):
    #x = np.array()
    n= len(c)
    m, t= b.shape
    if np.any(b < 0):
        raise ValueError("Se requiere que b >= 0")
    num_rows, num_cols = A.shape  
    print("num_rows:",num_rows)
    print("num_cols:",num_cols)
    if num_cols != n:
        raise ValueError("matrix A must have as many columns as elements of x vector")
    if num_rows != m:
        raise ValueError("matrix A must have as many rows as elements of b vector")
    A,c, basis, non_basis = phase1(c, A)
    
    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    cB = c[basis]
    cN = c[:, non_basis]
    N = A[:, non_basis]
    rn = cN- cB @ B_inv @ N

    rn_non_negative = True
    for i in range(rn.shape):
        if(rn[i]<0):
            rn_non_negative= False
    while not rn_non_negative:
        # Calcular rn (costes reducidos para variables no básicas)
        entering_var = None
        min_reduced_cost = -1e-10
        
        for j in non_basis:
            N_j = A[:, j]
            rc = c[j] - cB @ B_inv @ N_j
            if rc < min_reduced_cost:
                min_reduced_cost = rc
                entering_var = j
        xB = B_inv @ b
        # rn>= 0 (todos los costes reducidos >= 0) Condición de optimalidad -> se ha encontrado sol
        if entering_var is None:
            # Construir solución completa
            x = np.zeros(n)
            for i, var in enumerate(basis):
                if var < n:
                    x[var] = xB[i]
            return x, 'optimal'
        
        # Calcular dirección: Y_j = B^{-1} N_j
        N_j = A[:, entering_var]
        Y_j = B_inv @ N_j
        
        # Test de razón mínima
        if np.all(Y_j <= 1e-10):
            return None, 'unbounded'
        
        ratios = []
        for i in range(m):
            if Y_j[i] > 1e-10:
                ratios.append((xB[i] / Y_j[i], i))
            else:
                ratios.append((np.inf, i))
        
        alpha, leaving_idx = min(ratios)
        
        if alpha == np.inf:
            return None, 'unbounded'
        
        leaving_var = basis[leaving_idx]
        
        # Actualizar base
        basis[leaving_idx] = entering_var
        non_basis.remove(entering_var)
        non_basis.append(leaving_var)
        non_basis.sort()
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
        cB = c[basis]
        cN = c[:, non_basis]
        N = A[:, non_basis]
        rn = cN- cB @ B_inv @ N

        rn_non_negative = True
        for i in range(rn.shape):
            if(rn[i]<0):
                rn_non_negative= False

    raise RuntimeError("Máximo número de iteraciones alcanzado")


