"""Execute the notebook using this environment without registering a user kernel.

Run from the repository root: uv run python member1/verify_notebook.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[1]
    # A failed fresh run must not leave an older success report looking current.
    report_path = root / 'member1/results/notebook_validation.json'
    report_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='dsa5102-member1-') as temporary:
        temporary = Path(temporary)
        kernels = temporary / 'kernels' / 'member1-validation'
        kernels.mkdir(parents=True)
        (kernels / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Member1 validation', 'language': 'python'}))
        os.environ['JUPYTER_PATH'] = str(temporary)
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temporary / 'runtime')
        os.environ['IPYTHONDIR'] = str(temporary / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(temporary / 'matplotlib')
        os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
        import nbformat
        from nbclient import NotebookClient
        notebook_path = root / 'member1_eda_ridge.ipynb'
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        started = time.monotonic()
        client = NotebookClient(notebook, timeout=600, kernel_name='member1-validation',
                                resources={'metadata': {'path': str(root)}}, allow_errors=False)
        client.execute()
        code_cells = [c for c in notebook.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
        # Keep portable, conventional metadata, not the temporary validation kernel name.
        notebook.metadata.kernelspec = {'display_name': 'Python (DSA5102 Group 8)',
                                       'language': 'python', 'name': 'python3'}
        nbformat.validate(notebook)
        nbformat.write(notebook, notebook_path)
        elapsed = time.monotonic() - started
        report = {'executed_from_clean_kernel': True, 'python': sys.version.split()[0],
                  'code_cells': len(code_cells), 'error_outputs': 0,
                  'elapsed_seconds': round(elapsed, 2), 'notebook': notebook_path.name}
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
