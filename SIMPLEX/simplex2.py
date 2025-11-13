import numpy as np
from typing import Tuple, Optional, List

class SimplexResult:
    """Clase para almacenar los resultados del Simplex"""
    def __init__(self, status: str, x: Optional[np.ndarray] = None, 
                 obj_value: Optional[float] = None, iterations: int = 0):
        self.status = status  # 'optimal', 'unbounded', 'infeasible'
        self.x = x
        self.obj_value = obj_value
        self.iterations = iterations
    
    def __str__(self):
        if self.status == 'optimal':
            return (f"Estado: Óptimo\n"
                   f"Solución: {self.x}\n"
                   f"Valor objetivo: {self.obj_value:.6f}\n"
                   f"Iteraciones: {self.iterations}")
        elif self.status == 'unbounded':
            return f"Estado: Ilimitado\nIteraciones: {self.iterations}"
        else:
            return f"Estado: Infactible\nIteraciones: {self.iterations}"


def simplex_primal(c: np.ndarray, A: np.ndarray, b: np.ndarray, 
                   verbose: bool = False) -> SimplexResult:
    """
    Implementación del algoritmo Simplex Primal
    
    Resuelve: min c^T x
              s.a  Ax = b
                   x >= 0
    
    Parámetros:
    -----------
    c : array (n,) - Vector de costes
    A : array (m, n) - Matriz de restricciones
    b : array (m,) - Términos independientes (debe ser b >= 0)
    verbose : bool - Si True, muestra información de cada iteración
    
    Retorna:
    --------
    SimplexResult con el estado, solución y valor objetivo
    """
    
    # Convertir a arrays de numpy
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    m, n = A.shape
    
    # Verificar que b >= 0
    if np.any(b < 0):
        raise ValueError("El vector b debe ser no negativo (b >= 0)")
    
    # FASE I: Encontrar una base inicial factible
    if verbose:
        print("=" * 60)
        print("FASE I: Buscando solución básica factible inicial")
        print("=" * 60)
    
    base_indices, nonbase_indices = find_initial_basis(A, b, verbose)
    
    if base_indices is None:
        return SimplexResult('infeasible', iterations=0)
    
    # FASE II: Algoritmo Simplex
    if verbose:
        print("\n" + "=" * 60)
        print("FASE II: Algoritmo Simplex")
        print("=" * 60)
    
    iteration = 0
    MAX_ITERATIONS = 1000
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        # Matrices B y N
        B = A[:, base_indices]
        N = A[:, nonbase_indices]
        
        # Vectores de costes
        c_B = c[base_indices]
        c_N = c[nonbase_indices]
        
        # Calcular valores de las variables básicas
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return SimplexResult('infeasible', iterations=iteration)
        
        x_B = B_inv @ b
        
        # Calcular multiplicadores simplex (variables duales)
        lambda_T = c_B @ B_inv
        
        # Calcular costes reducidos
        r_N = c_N - lambda_T @ N
        
        if verbose:
            print(f"\n--- Iteración {iteration} ---")
            print(f"Base: {base_indices}")
            print(f"No base: {nonbase_indices}")
            print(f"x_B = {x_B}")
            print(f"Costes reducidos r_N = {r_N}")
            print(f"Valor objetivo = {c_B @ x_B:.6f}")
        
        # Condición de optimalidad: todos los costes reducidos >= 0
        if np.all(r_N >= -1e-10):  # Tolerancia numérica
            # Construir solución completa
            x = np.zeros(n)
            x[base_indices] = x_B
            obj_value = c @ x
            
            if verbose:
                print(f"\n*** SOLUCIÓN ÓPTIMA ENCONTRADA ***")
                print(f"Solución: {x}")
                print(f"Valor objetivo: {obj_value:.6f}")
            
            return SimplexResult('optimal', x, obj_value, iteration)
        
        # Seleccionar variable que entra en la base (coste reducido más negativo)
        q = np.argmin(r_N)
        entering_var = nonbase_indices[q]
        
        if verbose:
            print(f"Variable que entra: x_{entering_var} (coste reducido = {r_N[q]:.6f})")
        
        # Calcular dirección y_q = B^(-1) * N_q
        N_q = A[:, entering_var]
        y_q = B_inv @ N_q
        
        if verbose:
            print(f"Dirección y_q = {y_q}")
        
        # Verificar si el problema es ilimitado
        if np.all(y_q <= 1e-10):  # Tolerancia numérica
            if verbose:
                print("\n*** PROBLEMA ILIMITADO ***")
            return SimplexResult('unbounded', iterations=iteration)
        
        # Test de razón mínima (minimum ratio test)
        ratios = []
        for i in range(m):
            if y_q[i] > 1e-10:  # Solo considerar componentes positivas
                ratios.append((x_B[i] / y_q[i], i))
        
        if not ratios:
            if verbose:
                print("\n*** PROBLEMA ILIMITADO ***")
            return SimplexResult('unbounded', iterations=iteration)
        
        # Seleccionar variable que sale de la base
        alpha, p = min(ratios)
        leaving_var = base_indices[p]
        
        if verbose:
            print(f"Variable que sale: x_{leaving_var} (alpha = {alpha:.6f})")
        
        # Actualizar base
        base_indices[p] = entering_var
        nonbase_indices[q] = leaving_var
        
        # Ordenar para mantener consistencia
        base_indices = sorted(base_indices)
        nonbase_indices = sorted(nonbase_indices)
    
    if verbose:
        print(f"\n*** MÁXIMO DE ITERACIONES ALCANZADO ({MAX_ITERATIONS}) ***")
    
    return SimplexResult('max_iterations', iterations=iteration)


def find_initial_basis(A: np.ndarray, b: np.ndarray, 
                       verbose: bool = False) -> Tuple[Optional[List], Optional[List]]:
    """
    Encuentra una base inicial factible usando Fase I del Simplex
    
    Retorna:
    --------
    base_indices, nonbase_indices : listas de índices
    """
    m, n = A.shape
    
    # Buscar una base inicial trivial (matriz identidad en A)
    base_indices = []
    nonbase_indices = list(range(n))
    
    for i in range(m):
        # Buscar columna que sea el vector canónico e_i
        for j in range(n):
            col = A[:, j]
            if np.allclose(col, np.eye(m)[i]) and j not in base_indices:
                base_indices.append(j)
                nonbase_indices.remove(j)
                break
    
    # Si encontramos m columnas linealmente independientes que forman base
    if len(base_indices) == m:
        B = A[:, base_indices]
        try:
            B_inv = np.linalg.inv(B)
            x_B = B_inv @ b
            
            # Verificar si es factible
            if np.all(x_B >= -1e-10):
                if verbose:
                    print(f"Base inicial factible encontrada: {base_indices}")
                    print(f"x_B = {x_B}")
                return base_indices, nonbase_indices
        except np.linalg.LinAlgError:
            pass
    
    # Si no hay base trivial, usar Fase I completa con variables artificiales
    if verbose:
        print("No hay base trivial. Ejecutando Fase I completa...")
    
    return phase_one_simplex(A, b, verbose)


def phase_one_simplex(A: np.ndarray, b: np.ndarray, 
                      verbose: bool = False) -> Tuple[Optional[List], Optional[List]]:
    """
    Fase I del Simplex: encuentra una base inicial factible usando variables artificiales
    """
    m, n = A.shape
    
    # Crear problema de Fase I: min sum(a_i)
    # s.a. Ax + a = b, x >= 0, a >= 0
    
    A_phase1 = np.hstack([A, np.eye(m)])
    c_phase1 = np.hstack([np.zeros(n), np.ones(m)])
    
    # Base inicial: variables artificiales
    base_indices = list(range(n, n + m))
    nonbase_indices = list(range(n))
    
    # Resolver problema de Fase I
    result = simplex_phase1(c_phase1, A_phase1, b, base_indices, nonbase_indices, verbose)
    
    if result.status != 'optimal':
        return None, None
    
    # Verificar si todas las variables artificiales son 0
    if result.obj_value > 1e-6:
        if verbose:
            print("Problema original es infactible (variables artificiales > 0)")
        return None, None
    
    # Extraer base factible del problema original
    base_indices_original = [i for i in result.base_indices if i < n]
    nonbase_indices_original = [i for i in range(n) if i not in base_indices_original]
    
    if verbose:
        print(f"Base factible encontrada en Fase I: {base_indices_original}")
    
    return base_indices_original, nonbase_indices_original


def simplex_phase1(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                   base_indices: List, nonbase_indices: List,
                   verbose: bool = False) -> SimplexResult:
    """Simplex para Fase I (versión simplificada)"""
    
    m, n = A.shape
    iteration = 0
    MAX_ITERATIONS = 1000
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        B = A[:, base_indices]
        N = A[:, nonbase_indices]
        c_B = c[base_indices]
        c_N = c[nonbase_indices]
        
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return SimplexResult('infeasible', iterations=iteration)
        
        x_B = B_inv @ b
        lambda_T = c_B @ B_inv
        r_N = c_N - lambda_T @ N
        
        if np.all(r_N >= -1e-10):
            result = SimplexResult('optimal', None, c_B @ x_B, iteration)
            result.base_indices = base_indices
            return result
        
        q = np.argmin(r_N)
        entering_var = nonbase_indices[q]
        
        N_q = A[:, entering_var]
        y_q = B_inv @ N_q
        
        if np.all(y_q <= 1e-10):
            return SimplexResult('unbounded', iterations=iteration)
        
        ratios = [(x_B[i] / y_q[i], i) for i in range(m) if y_q[i] > 1e-10]
        if not ratios:
            return SimplexResult('unbounded', iterations=iteration)
        
        _, p = min(ratios)
        leaving_var = base_indices[p]
        
        base_indices[p] = entering_var
        nonbase_indices[q] = leaving_var
        base_indices = sorted(base_indices)
        nonbase_indices = sorted(nonbase_indices)
    
    return SimplexResult('max_iterations', iterations=iteration)


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("EJEMPLO 1: Problema del documento (página 34)")
    print("=" * 70)
    
    # min -x1 - 4x2
    # s.a. x1 - x2 + x3 = 5
    #      3x1 + 2x2 + x4 = 20
    #      -3x1 + 5x2 + x5 = 3
    #      xi >= 0
    
    c1 = np.array([-1, -4, 0, 0, 0])
    A1 = np.array([
        [1, -1, 1, 0, 0],
        [3, 2, 0, 1, 0],
        [-3, 5, 0, 0, 1]
    ])
    b1 = np.array([5, 20, 3])
    
    result1 = simplex_primal(c1, A1, b1, verbose=True)
    print("\n" + str(result1))
    
    print("\n\n" + "=" * 70)
    print("EJEMPLO 2: Problema ilimitado (página 40)")
    print("=" * 70)
    
    # min -x1 - x2
    # s.a. x1 - x2 + x3 = 1
    #      -3x1 + x2 + x4 = 3
    #      xi >= 0
    
    c2 = np.array([-1, -1, 0, 0])
    A2 = np.array([
        [1, -1, 1, 0],
        [-3, 1, 0, 1]
    ])
    b2 = np.array([1, 3])
    
    result2 = simplex_primal(c2, A2, b2, verbose=True)
    print("\n" + str(result2))
    
    print("\n\n" + "=" * 70)
    print("EJEMPLO 3: Problema simple")
    print("=" * 70)
    
    # min 2x1 + 3x2
    # s.a. x1 + x2 + s1 = 12
    #      -2x1 + x2 + s2 = 2
    #      x1 + 3x2 + s3 = 30
    #      xi >= 0
    
    c3 = np.array([2, 3, 0, 0, 0])
    A3 = np.array([
        [1, 1, 1, 0, 0],
        [-2, 1, 0, 1, 0],
        [1, 3, 0, 0, 1]
    ])
    b3 = np.array([12, 2, 30])
    
    result3 = simplex_primal(c3, A3, b3, verbose=True)
    print("\n" + str(result3))