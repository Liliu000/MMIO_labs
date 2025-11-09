import numpy as np
from SIMPLEX.simplexv0 import simplex
from AMPL.test_example import generate_dat, load_ampl_results, run_ampl_model
import pandas as pd

def optimize(A: np.matrix, b: np.matrix, c:np.array, filename: str):
    result = simplex(c, A, b)

    generate_dat(A=A, b=b, c=c)
    ampl_output, ampl_error = run_ampl_model()
    ampl_x = load_ampl_results()
    lines= []
    lines.append("SIMPLEX")
    if result["error"] is not None:
        lines.append(result["error"])
    if result["status"] is not None:
        lines.append(result["status"])
    if result["status"] == "optimal":
        lines.append(
        f"SIMPLEX: x={result["x"]}; f={result["obj_value"]}; iterations_phase1={result["iterations_phase1"]}; iterations_phase2: {result["iterations_phase2"]}"
        )
    lines.append("AMPL")
    if(ampl_error) is not None:
        lines.append(ampl_error)
    else:
        lines.append(ampl_output)
        lines.append(f"AMPL:    x={ampl_x}")

    output = "\n".join(lines) 
    print(output)
    with open(filename, 'w') as f:
        f.write(output)
    

if __name__ == '__main__':
    """ Con solucion, BSF TRIVIAL
    min -x1 -4x2
    s.a x1 -x2 + x3 = 5
    3x1 + 2x2 + x4 = 20
    -3x1 + 5x2 + x5 = 3
    xi >= 0
    """
    c= np.array([-1.0,-4.,0., 0., 0.]) 
    A= np.matrix([[1.,-1., 1., 0., 0.],[3., 2.,0., 1., 0.], [-3., 5., 0., 0., 1.]]) 
    b= np.array([[5.],[20.], [3.]]) 

    optimize(A=A,b=b,c=c, filename="test_output/optimum_trivialBSF.txt")

    """
    ILIMITADO, BSF TRIVIAL
    min -x1 -x2
    s.a x1 - x2 + x3= 1
    -3x1 + x2 +x4= 3
    xi >= 0
    """
    c= np.array([-1,-1,0, 0]) 
    A= np.matrix([[1,-1,1, 0],[-3, 1, 0, 1]]) 
    b= np.array([[1],[3]]) 

    optimize(A=A,b=b,c=c, filename="test_output/unbounded_trivialBSF.txt")
    """
    Con solucion, PHASE1
    min -x1 +2x2 + x3
    s.a -2x1 + x2 + x3= 1
    -1x1 + 2x2 + 3x3= 3
    xi >= 0
    """
    c= np.array([1,2,1]) 
    A= np.matrix([[-2,1,1],[-1, 2, 3]]) 
    b= np.array([[1],[3]]) 

    optimize(A=A,b=b,c=c, filename="test_output/optimum_phase1.txt")

    """
    INFACTIBLE
    min -x1-x2
    s.a 
    """
    A = np.matrix([[1., 1., 1., 0. ], [1., 1., 0., -1.]])
    c = np.array([-1., -1., 0., 0.])
    b= np.matrix([[1.], [3.]])

    optimize(A=A,b=b,c=c, filename="test_output/infeasible_phase1.txt")


    
    
