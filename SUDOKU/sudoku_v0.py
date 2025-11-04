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

    """matrix = np.zeros((N, M, K))
    print(matrix.shape)
    print(df.iloc[N-1, M-1])  # Using iloc to access by integer position
    for j in range(N):
        for i in range(M):
            num = df.iloc[i, j]
            matrix[i][j][num-1] = 1"""

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
    
    """print("AMPL output:")
    print(result.stdout)
    if result.stderr:
        print("AMPL errors:")
        print(result.stderr)
    
    simplex_iterations = None
    solve_elapsed_time = None
    simplex_match = re.search(r'(\d+)\s+simplex iterations', result.stdout)
    if simplex_match:
        simplex_iterations = int(simplex_match.group(1))
    
    time_match = re.search(r'_solve_elapsed_time\s*=\s*([\d.]+)', result.stdout)
    if time_match:
        solve_elapsed_time = float(time_match.group(1))

    return simplex_iterations, solve_elapsed_time"""



