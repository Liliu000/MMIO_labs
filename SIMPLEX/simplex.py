import numpy as np
from numpy.linalg import solve, LinAlgError

def simplex_revised(c, A, b):
    """
    Resuelve un problema de programación lineal usando el Simplex Revisado.
    
    Minimiza: c^T x
    Sujeto a: Ax = b, x >= 0
    
    Parámetros:
    -----------
    c : array_like, shape (N,)
        Vector de costes
    A : array_like, shape (M, N)
        Matriz de restricciones
    b : array_like, shape (M,)
        Términos independientes (se asume b >= 0)
    
    Retorna:
    --------
    x : ndarray or None
        Solución óptima si existe, None si no hay solución
    status : str
        'optimal': solución óptima encontrada
        'unbounded': problema ilimitado
        'infeasible': problema infactible
    """
    # Convertir a arrays de numpy
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    M, N = A.shape
    
    # Verificar que b >= 0
    if np.any(b < 0):
        raise ValueError("Se requiere que b >= 0")
    
    # Fase I: Encontrar una solución básica factible inicial
    basis, non_basis = _phase_1_revised(A, b, M, N)
    
    if basis is None:
        return None, 'infeasible'
    
    # Fase II: Optimizar con la función objetivo original
    x, status = _phase_2_revised(A, b, c, basis, non_basis, M, N)
    
    return x, status


def _phase_1_revised(A, b, M, N):
    """
    Fase I: Encuentra una solución básica factible usando variables artificiales.
    """
    # Extender A con variables artificiales: [A | I]
    A_extended = np.hstack([A, np.eye(M)])
    
    # Base inicial: variables artificiales (índices N a N+M-1)
    basis = list(range(N, N + M))
    non_basis = list(range(N))
    
    # Vector de costes para Fase I: minimizar suma de artificiales
    c_phase1 = np.zeros(N + M)
    c_phase1[N:] = 1
    
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        # Calcular B (matriz básica) y B_inv
        B = A_extended[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except LinAlgError:
            return None, None
        
        # Calcular solución básica actual
        x_B = B_inv @ b
        
        # Calcular precios duales: π^T = c_B^T B^{-1}
        c_B = c_phase1[basis]
        pi = c_B @ B_inv
        
        # Calcular costes reducidos para variables no básicas
        reduced_costs = {}
        entering_var = None
        min_reduced_cost = -1e-10
        
        for j in non_basis:
            A_j = A_extended[:, j]
            rc = c_phase1[j] - pi @ A_j
            reduced_costs[j] = rc
            if rc < min_reduced_cost:
                min_reduced_cost = rc
                entering_var = j
        
        # Condición de optimalidad
        if entering_var is None:
            break
        
        # Calcular dirección: d = B^{-1} A_j
        A_entering = A_extended[:, entering_var]
        d = B_inv @ A_entering
        
        # Test de razón mínima
        if np.all(d <= 1e-10):
            return None, None  # Problema Fase I ilimitado -> infactible
        
        ratios = []
        for i in range(M):
            if d[i] > 1e-10:
                ratios.append((x_B[i] / d[i], i))
            else:
                ratios.append((np.inf, i))
        
        theta, leaving_idx = min(ratios)
        leaving_var = basis[leaving_idx]
        
        # Actualizar base
        basis[leaving_idx] = entering_var
        non_basis.remove(entering_var)
        non_basis.append(leaving_var)
        non_basis.sort()
        
        iteration += 1
    
    # Verificar factibilidad
    B = A_extended[:, basis]
    B_inv = np.linalg.inv(B)
    x_B = B_inv @ b
    
    # Calcular valor objetivo Fase I
    c_B = c_phase1[basis]
    obj_value = c_B @ x_B
    
    if abs(obj_value) > 1e-10:
        return None, None  # Infactible
    
    # Remover variables artificiales de la base si es posible
    new_basis = []
    for i, var in enumerate(basis):
        if var >= N:  # Variable artificial
            # Intentar reemplazarla con una variable original
            B_current = A_extended[:, basis]
            B_inv_current = np.linalg.inv(B_current)
            
            replaced = False
            for j in range(N):
                if j not in basis:
                    A_j = A_extended[:, j]
                    d = B_inv_current @ A_j
                    if abs(d[i]) > 1e-10:
                        new_basis.append(j)
                        replaced = True
                        break
            
            if not replaced:
                new_basis.append(var)
        else:
            new_basis.append(var)
    
    basis = new_basis
    non_basis = [j for j in range(N) if j not in basis]
    
    return basis, non_basis


def _phase_2_revised(A, b, c, basis, non_basis, M, N):
    """
    Fase II: Optimiza con la función objetivo original usando Simplex Revisado.
    """
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        # Calcular B (matriz básica) y B_inv
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except LinAlgError:
            return None, 'infeasible'
        
        # Calcular solución básica actual
        x_B = B_inv @ b
        
        # Calcular precios duales: π^T = c_B^T B^{-1}
        c_B = c[basis]
        pi = c_B @ B_inv
        
        # Calcular costes reducidos para variables no básicas
        entering_var = None
        min_reduced_cost = -1e-10
        
        for j in non_basis:
            A_j = A[:, j]
            rc = c[j] - pi @ A_j
            if rc < min_reduced_cost:
                min_reduced_cost = rc
                entering_var = j
        
        # Condición de optimalidad: todos los costes reducidos >= 0
        if entering_var is None:
            # Construir solución completa
            x = np.zeros(N)
            for i, var in enumerate(basis):
                if var < N:
                    x[var] = x_B[i]
            return x, 'optimal'
        
        # Calcular dirección: d = B^{-1} A_j
        A_entering = A[:, entering_var]
        d = B_inv @ A_entering
        
        # Test de razón mínima
        if np.all(d <= 1e-10):
            return None, 'unbounded'
        
        ratios = []
        for i in range(M):
            if d[i] > 1e-10:
                ratios.append((x_B[i] / d[i], i))
            else:
                ratios.append((np.inf, i))
        
        theta, leaving_idx = min(ratios)
        
        if theta == np.inf:
            return None, 'unbounded'
        
        leaving_var = basis[leaving_idx]
        
        # Actualizar base
        basis[leaving_idx] = entering_var
        non_basis.remove(entering_var)
        non_basis.append(leaving_var)
        non_basis.sort()
        
        iteration += 1
    
    raise RuntimeError("Máximo número de iteraciones alcanzado")


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo 1: Problema con solución óptima
    print("=" * 50)
    print("Ejemplo 1: Problema estándar")
    print("=" * 50)
    c = np.array([-3, -2])
    A = np.array([[2, 1],
                  [1, 2]], dtype=float)
    b = np.array([4, 3], dtype=float)
    
    x, status = simplex_revised(c, A, b)
    print(f"Estado: {status}")
    if status == 'optimal':
        print(f"Solución óptima: x = {x}")
        print(f"Valor objetivo: {np.dot(c, x)}")
    print()
    
    # Ejemplo 2: Problema ilimitado
    print("=" * 50)
    print("Ejemplo 2: Problema ilimitado")
    print("=" * 50)
    c = np.array([-1, -1])
    A = np.array([[1, -1]], dtype=float)
    b = np.array([1], dtype=float)
    
    x, status = simplex_revised(c, A, b)
    print(f"Estado: {status}")
    print()
    
    # Ejemplo 3: Problema más grande
    print("=" * 50)
    print("Ejemplo 3: Problema más complejo")
    print("=" * 50)
    c = np.array([1, 1, 1])
    A = np.array([[1, 2, 1],
                  [2, 1, 3]], dtype=float)
    b = np.array([5, 6], dtype=float)
    
    x, status = simplex_revised(c, A, b)
    print(f"Estado: {status}")
    if status == 'optimal':
        print(f"Solución óptima: x = {x}")
        print(f"Valor objetivo: {np.dot(c, x)}")