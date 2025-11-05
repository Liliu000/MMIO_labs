import numpy as np
from SIMPLEX.simplexv0 import simplex
from SUDOKU.sudoku import generate_dat, load_ampl_results, run_ampl_model
import pandas as pd

def generate_x(K):
    df = pd.read_csv("SUDOKU/sudoku_unsolved.csv")
    df = df.drop(df.columns[0], axis=1)
    print(df)
    N,M = df.shape
    print(df.shape)

    matrix = np.zeros((N, M, K))
    print(matrix.shape)
    print(df.iloc[N-1, M-1])  # Using iloc to access by integer position
    for j in range(N):
        for i in range(M):
            num = df.iloc[i, j]
            matrix[i][j][num-1] = 1
    return matrix

if __name__ == '__main__':
    
    """generate_dat(K=9)
    run_ampl_model()
    sol_df = load_ampl_results()
    """
    #generate_A(9)
    
    c= np.array([1,2,1]) #M all 0
    A= np.matrix([[-2,1,1],[-1, 2, 3]]) # NxNxM
    b= np.array([[1],[3]]) #N all 1
    X, status = simplex(c, A, b)
    print("X:", X)
    print("status", status)  
