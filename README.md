# MMIO_labs


steps:
1. create virtual environment python:
windows:  python -m venv venv
2. activate environment: .\venv\Scripts\Activate.ps1
3. install needed packages: pip install -r requirements.txt
4. Execute simplex: from directory MMIO_labs, execute: python .\compare.py
    resutls will be stored in test_output directory
5. Execute SUDOKU: from directory MMIO_labs\SUDOKU, execute:  ampl .\sudoku.run
    If want to try other examples, modify sudoku.run
    results will be stored in sudoku_solution.txt