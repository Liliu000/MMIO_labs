import subprocess
import time
import re
import pandas as pd
import numpy as np


def generate_dat(A: np.matrix, b: np.matrix, c: np.array, filename='AMPL/test_example.dat'):

    """matrix = np.zeros((N, M, K))
    print(matrix.shape)
    print(df.iloc[N-1, M-1])  # Using iloc to access by integer position
    for j in range(N):
        for i in range(M):
            num = df.iloc[i, j]
            matrix[i][j][num-1] = 1"""
    m, n = A.shape
    with open(filename, 'w') as f:
        f.write("#Define sets\n")
        f.write(f"param n := {n};\n")
        f.write(f"param m := {m};\n")

        f.write("#Define parameters\n")
        f.write(f"param A: {'\t'.join([str(i+1) for i in range(n)])} :=\n")
        for j in range(m):
            f.write(f" {j+1} " + ' '.join(f"{A[j, i]:f}" for i in range(n)) + "\n")
        f.write(";\n")

        f.write(f"param c :=\n")
        for i in range(n):
            f.write(f" {i+1} " + f"{c[i]:f}" + "\n")
        f.write(";\n")

        f.write(f"param b :=\n")
        for j in range(m):
            f.write(f" {j+1} " + ' '.join(f"{b[j, i]:f}" for i in range(1)) + "\n")
        f.write(";\n")





def run_ampl_model():
    
    # Run AMPL
    result = subprocess.run(['ampl', 'test_example.run'],cwd='AMPL', 
                          capture_output=True, text=True)
    
    print("AMPL output:")
    print(result.stdout)
    if result.stderr:
        return None, result.stderr
    return result.stdout, None



def load_ampl_results(filepath="AMPL/x.csv"):
    df = pd.read_csv(filepath, header=None, names=['i','x'])
    I = df["i"].max()

    # Initialize 3D NumPy array
    x = np.zeros(I)

    # Fill in values
    for _, row in df.iterrows():
        x[int(row["i"])-1] = row["x"]
    
    return x
