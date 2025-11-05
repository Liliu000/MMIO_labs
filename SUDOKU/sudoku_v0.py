import subprocess
import time
import re
import pandas as pd
import numpy as np


def generate_dat(K, filename='SUDOKU/sudoku.dat'):

    df = pd.read_csv("SUDOKU/sudoku_unsolved.csv")
    df = df.drop(df.columns[0], axis=1)
    print(df)
    N,M = df.shape
    print(df.shape)

    with open(filename, 'w') as f:
        f.write("#Define sets\n")
        f.write(f"param n := {N};\n")
        f.write(f"param m := {K};\n")
        f.write(f"param nn := {3};\n")

        f.write("#Define parameters\n")
        f.write(f"param a: {'\t'.join([str(j+1) for j in range(N)])} :=\n")
        for i in range(N):
            f.write(f" {i+1} " + ' '.join(f"{df.iloc[i, j]:.0f}" for j in range(N)) + "\n")
        f.write(";\n")



def run_ampl_model():
    
    # Run AMPL
    result = subprocess.run(['ampl', 'sudoku.run'],cwd='AMPL', 
                          capture_output=True, text=True)
    

def load_ampl_results(filepath="SUDOKU/x_matrix.csv"):
    df = pd.read_csv(filepath, header=None, names=['i','j','k','x'])
    I = df["i"].max()
    J = df["j"].max()
    K = df["k"].max()

    # Initialize 3D NumPy array
    matrix = np.zeros((I, J, K))

    # Fill in values
    for _, row in df.iterrows():
        matrix[int(row["i"]) - 1, int(row["j"]) - 1, int(row["k"]) - 1] = row["x"]
    
    I, J, K = matrix.shape
    result = np.zeros((I, J))
    for j in range(J):
        for i in range(I):
            for k in range(K):
                if matrix[i][j][k] !=0:
                    result[i][j]=k+1
            
    res = pd.DataFrame(data=result)
    res = res.astype(int)
    res.to_csv("SUDOKU/sudoku_solved_ampl.csv")
    return res




