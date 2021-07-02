import datetime
import os
import re
import subprocess
import sys
import unittest
import argparse
import time
# for i in sys.path:
#     if 'pyross' in i or i == '':
#         sys.path.remove(i)


def run_notebook_tests(path, recursive=False):
    """
    Runs Jupyter notebook tests. Exits if they fail.
    """
    basepath = os.path.dirname(__file__)
    nbpath = os.path.abspath(os.path.join(basepath, "..", path))

    '''
    Ignore notebooks which take longer or have deliberate errors, 
    but check they still exists
    '''
    os.chdir('../examples/')

    cwd =os.getcwd()
    ignore_list = [os.path.join(cwd, 'control/ex04-SIR-optimal_control.ipynb'),
                   os.path.join(cwd, 'inference/ex06_inference_latent_SEIR.ipynb'),
                   os.path.join(cwd, 'inference/ex07-latent-hessian-and-sensitivity.ipynb'),
                   os.path.join(cwd, 'inference/ex08-optimal_design.ipynb'),
                   os.path.join(cwd, 'inference/ex09a_calibration_SIR.ipynb'),
                   os.path.join(cwd, 'inference/ex09b_calibration_latent_SIR.ipynb'),
                   os.path.join(cwd, 'inference/ex11-evidence.ipynb'),
                   os.path.join(cwd, 'inference/ex12-fastest-growing-mode-inference.ipynb'),
                   os.path.join(cwd, 'stochastic/ex05-SEAIRQ.ipynb'),
                   os.path.join(cwd, 'stochastic/ex03-SIkR.ipynb'),
                   os.path.join(cwd, 'stochastic/ex07-Spp-overdispersion.ipynb'),
                    ]

    for ignored_book in ignore_list:
        if not os.path.isfile(ignored_book):
            raise Exception('Ignored notebook not found: ' + ignored_book)

    # Scan and run
    print('Testing notebooks')
    ok = True
    for notebook, cwd in list_notebooks(nbpath, recursive, ignore_list):
        os.chdir(cwd) # necessary for relative imports in notebooks
        ok &= test_notebook(notebook)
        # print(notebook)
    if not ok:
        print('\nErrors encountered in notebooks')
        sys.exit(1)
    print('\nOK')


def list_notebooks(root, recursive=False, ignore_list=None, notebooks=None):
    """
    Returns a list of all notebooks in a directory.
    """
    if notebooks is None:
        notebooks = []
    if ignore_list is None:
        ignore_list = []
    try:
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            cwd = os.path.dirname(path)
            if path in ignore_list:
                print('Skipping ignored notebook: ' + path)
                continue
    
            # Add notebooks
            if os.path.splitext(path)[1] == '.ipynb':
                notebooks.append((path,cwd))
    
            # Recurse into subdirectories
            elif recursive and os.path.isdir(path):
                # Ignore hidden directories
                if filename[:1] == '.':
                    continue
                list_notebooks(path, recursive, ignore_list, notebooks)
    except NotADirectoryError:
        path = root
        cwd = os.path.dirname(path)
        return [(path,cwd)]

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
    ipylines = ['ipython', 'show(']
    code = '\n'.join([x for x in code.splitlines() if not 'ipython' in x])
    for x in code.splitlines():
        if not any(s in ipylines for s in x):
            code += '\n'.join([x])
    # print(code)

    # Tell matplotlib not to produce any figures
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Template'

    # Run in subprocess
    start = time.time()
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
            # print('-- script ' + '-' * (79 - 10))
            # for i, line in enumerate(code.splitlines()):
            #     j = str(1 + i)
            #     print(j + ' ' * (5 - len(j)) + line)
            print('-- stdout ' + '-' * (79 - 10))
            print(stdout)
            print('-- stderr ' + '-' * (79 - 10))
            print(stderr)
            print('-' * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        stop = time.time()
        print('ABORTED after', round(stop-start,4), "s")
        sys.exit(1)

    # Successfully run
    stop = time.time()
    print('ok. Run took ', round(stop-start,4), "s")
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
    # Set up argument parsing
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(
        description='Run notebook unit tests for PyRoss.',
    )
    # Unit tests
    parser.add_argument(
        '--path', default = '.',
        help='Run specific notebook or folder containing notebooks',)
    parser.add_argument(
        '--recursive', default = True, type=str2bool,
        help='Wheither or not subfolders are searched',)

    # Parse!
    args = parser.parse_args()
    print(args)
    run_notebook_tests(args.path, recursive=args.recursive)
