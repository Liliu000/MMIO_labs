# MMIO_labs


steps:
1. create virtual environment python:
windows:  python -m venv venv
2. activate environment: <br/>.\venv\Scripts\Activate.ps1
3. install needed packages:<br/> pip install -r requirements.txt
4. Execute simplex: from directory MMIO_labs, execute: <br/> python .\compare.py <br/>
    resutls will be stored in test_output directory
5. Execute SUDOKU: from directory MMIO_labs\SUDOKU, execute: <br/> ampl .\sudoku.run <br/>
    If want to try other examples, modify sudoku.run <br/>
    results will be stored in sudoku_solution.txt<br/>
