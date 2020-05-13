import datetime
import os
import re
import subprocess
import sys
import unittest
# for i in sys.path:
#     if 'pyross' in i or i == '':
#         sys.path.remove(i)


def run_notebook_tests():
    """
    Runs Jupyter notebook tests. Exits if they fail.
    """
    basepath = os.path.dirname(__file__)
    nbpath = os.path.abspath(os.path.join(basepath, "..", "examples"))
    # Ignore books with deliberate errors, but check they still exist
    ignore_list = []

    for ignored_book in ignore_list:
        if not os.path.isfile(ignored_book):
            raise Exception('Ignored notebook not found: ' + ignored_book)

    # Scan and run
    print('Testing notebooks')
    ok = True
    for notebook in list_notebooks(nbpath, True, ignore_list):
        ok &= test_notebook(notebook)
        # print(notebook)
    if not ok:
        print('\nErrors encountered in notebooks')
        sys.exit(1)
    print('\nOK')


def list_notebooks(root, recursive=True, ignore_list=None, notebooks=None):
    """
    Returns a list of all notebooks in a directory.
    """
    if notebooks is None:
        notebooks = []
    if ignore_list is None:
        ignore_list = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if path in ignore_list:
            print('Skipping ignored notebook: ' + path)
            continue

        # Add notebooks
        if os.path.splitext(path)[1] == '.ipynb':
            notebooks.append(path)

        # Recurse into subdirectories
        elif recursive and os.path.isdir(path):
            # Ignore hidden directories
            if filename[:1] == '.':
                continue
            list_notebooks(path, recursive, ignore_list, notebooks)

    return notebooks


def test_notebook(path):
    """
    Tests a notebook in a subprocess, exists if it doesn't finish.
    """
    import nbconvert
    print('Running ' + path + ' ... ', end='')
    sys.stdout.flush()

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter()
    code, __ = e.from_filename(path)

    # Remove coding statement, if present
    code = '\n'.join([x for x in code.splitlines() if "ipython" not in x])
    # print(code)

    # Tell matplotlib not to produce any figures
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Template'

    # Run in subprocess
    cmd = [sys.executable, '-c', code]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        stdout, stderr = p.communicate()
        # TODO: Use p.communicate(timeout=3600) if Python3 only
        if p.returncode != 0:
            # Show failing code, output and errors before returning
            print('ERROR')
            print('-- script ' + '-' * (79 - 10))
            for i, line in enumerate(code.splitlines()):
                j = str(1 + i)
                print(j + ' ' * (5 - len(j)) + line)
            print('-- stdout ' + '-' * (79 - 10))
            print(stdout)
            print('-- stderr ' + '-' * (79 - 10))
            print(stderr)
            print('-' * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print('ABORTED')
        sys.exit(1)

    # Successfully run
    print('ok')
    return True


def export_notebook(ipath, opath):
    """
    Exports the notebook at `ipath` to a python file at `opath`.
    """
    import nbconvert
    from traitlets.config import Config

    # Create nbconvert configuration to ignore text cells
    c = Config()
    c.TemplateExporter.exclude_markdown = True

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter(config=c)
    code, __ = e.from_filename(ipath)

    # Remove "In [1]:" comments
    r = re.compile(r'(\s*)# In\[([^]]*)\]:(\s)*')
    code = r.sub('\n\n', code)

    # Store as executable script file
    with open(opath, 'w') as f:
        f.write('#!/usr/bin/env python')
        f.write(code)
    os.chmod(opath, 0o775)


if __name__ == '__main__':
    run_notebook_tests()
