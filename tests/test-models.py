#!/usr/bin/env python3
#
# Tests all models' syntax, meta data, and derivatives
#
import os
import re
import sys
import traceback

from pathlib import Path

import myokit

# Natural sort regex
_natural_sort_regex = re.compile(r'([0-9]+)')

# Test directories
_test_root = Path(__file__).parent
_mmt_root = _test_root.parent / 'models-mmt'
_cm1_root = _test_root.parent / 'models-cellml1'
_cm2_root = _test_root.parent / 'models-cellml2'

# Overwrite expected output files (after prompt)
_write_expected = False


# Test data: state vectors, bound variables, derivative vectors
class DTest:
    """
    Tests a model by comparing calculated derivatives against a known
    reference.

    Arguments:

    ``state``
        A state vector or a path to a file containing one
    ``derivatives``
        A derivatives vector or a path to a file containing one
    ``inputs``
        A dictionary mapping bound variables to new values
    ``prepare=None``
        An optional callable that modifies models before testing (e.g. changing
        parameters). If given, models will be cloned before ``prepare`` is
        called.

    """

    def __init__(self, state, derivatives, inputs={}, prepare=None):
        self._state = state
        self._inputs = inputs
        self._expected = derivatives
        self._cached_vectors = False
        self._prepare = prepare
        self._org_expected = False

    def run(self, model, allow_write_expected=False):
        """
        Runs this test on the given model.

        Returns ``None`` if succesful, else a formatted multi-line string
        describing the numberical differences (see :meth:`myokit.step()`).
        """
        # Parse constructor arguments on first run (need a model for this)
        if not self._cached_vectors:

            # Prepare model first, in case number of states changes
            if callable(self._prepare):
                model = model.clone()
                self._prepare(model)

            # Load or check state and derivatives
            if isinstance(self._state, str):
                path = _test_root / 'data' / self._state
                self._state = myokit.load_state(path, model)
            else:
                self._state = model.map_to_state(self._state)

            if isinstance(self._expected, str):
                self._org_expected = _test_root / 'data' / self._expected
                self._expected = myokit.load_state(self._org_expected, model)
            else:
                self._expected = model.map_to_state(self._expected)

        # Check if derivatives match
        dy = model.evaluate_derivatives(
            state=self._state, inputs=self._inputs,
            ignore_unbound_inputs=False)
        if all([myokit.float.eq(a, b) for (a, b) in zip(dy, self._expected)]):
            return None

        # Get long error message
        err = myokit.step(model, state=self._state, inputs=self._inputs,
                          reference=self._expected)

        # Allow cheeky write or overwrite
        if _write_expected and allow_write_expected and self._org_expected:
            print()
            print(err)
            print()
            y = input('Overwrite test output file {self._org_expected} (y/n)?')
            if y.strip().lower() == 'y':
                myokit.save_state(self._org_expected, dy)
                print('New output written to file.')
            print()
            return None
        return err


# Derivative tests
dtests = {
    'vc-level-0': {
        'default': DTest(
            'vc-level-0-default-in.txt',  'vc-level-0-default-out.txt'),
        'steadier': DTest(
            'vc-level-0-steadier-in.txt',  'vc-level-0-steadier-out.txt',
            {'pace': -120}),
        'moving': DTest(
            'vc-level-0-moving-in.txt',  'vc-level-0-moving-out.txt',
            {'pace': 40}),

        },
    'vc-level-1': {},
}


def test_models():
    """ Scans for and syntax-checks Myokit models. """

    # Scan directory, running models as we find them.
    def scan(root, failed=None):
        if failed is None:
            failed = []

        for path in sorted(root.iterdir(), key=natural_sort_path):
            if path.suffix == '.mmt':
                fancy = str(path.relative_to(_mmt_root.parent))
                print(fancy + '.' * (max(0, 70 - len(fancy))), end='')
                sys.stdout.flush()

                res = test(path)
                if res is None:
                    print('ok')
                else:
                    print('FAIL')
                    failed.append((path, res))
            elif path.is_dir():
                # Ignore hidden directories
                if path.name[:1] == '.':
                    continue
                scan(path, failed)

        return failed

    failed = scan(_mmt_root)
    if failed:
        for path, e in failed:
            fancy = str(path.relative_to(_mmt_root.parent))
            fancy = f'Error output for: {fancy}'
            print(f'== {fancy} ' + '=' * (79 - (4 + len(fancy))))
            print(f'\n{e}\n')
        print('=' * 79)
        print(f'Test failed ({len(failed)}) error(s).')
        return False
    print('Test passed.')
    return True


def natural_sort_path(path):
    """
    Function to use as ``key`` in a sort, to get natural sorting of paths
    (e.g. "2" before "10").
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in _natural_sort_regex.split(path.name)]


def test(path):
    """
    Runs all tests for a model.

    - Syntax and meta data in mmt file
    - Reference states and derivatives are found and well-formed
    -
    - CellML files are found and well-formed

    Testing ends when the first test fails.
    """
    model, err = test_syntax_and_meta(path)
    if err is not None:
        return err

    # Isolate model name
    model_name = path.stem

    # Find derivative tests
    tests = dtests.get(model_name, {})
    if len(tests) == 0:
        return f'No derivative tests found for {model_name}'

    # Run derivative tests for Myokit models
    for test_name, test in tests.items():
        try:
            err = test.run(model, allow_write_expected=True)
        except Exception as e:
            return (f'Exception during derivative test {test_name}: {e}\n' +
                    traceback.format_exc())
        if err is not None:
            return err



    # Get CellML files
    cellml1, cellml2, err = find_cellml_files(path)
    if err is not None:
        return err




    return


def test_syntax_and_meta(path):
    """
    Tests a Myokit model's syntax and derivatives.

    Returns ``(model, err)`` where ``model`` is set if loaded correctly, and
    ``err`` is a string if an error occurred.
    """

    # Check that we can load it
    try:
        m = myokit.load_model(path)
    except myokit.ParseError as e:
        return None, str(e)

    # Check its meta-data
    tags = [
        'name',
        'version',
        'mmt_authors',
        'display_name',
        'desc',
    ]
    for tag in tags:
        if tag not in m.meta:
            return m, f'Missing meta data annotation: {tag}'
        if m.meta[tag] == '':
            return m, f'Empty meta data annotation: {tag}'

    # Check for trailing whitespace
    with open(path, 'r') as f:
        trailing = []
        for i, line in enumerate(f.readlines()):
            if line.rstrip() != line.rstrip('\n\r'):
                trailing.append(i)
        if trailing:
            trailing = ', '.join(str(1 + i) for i in trailing)
            return None, f'Trailing whitespace on line(s): {trailing}'

    return m, None


'''
def find_state_files(model, model_name):
    """
    Finds a list of tuples (in, out) for the given model name, where ``in`` is
    the ``Path`` to a file containing state values, and ``out`` is the ``Path``
    to the corresponding derivatives.

    Files must take the form ``model-name-(in|out)-(id).txt``, where the input
    files (``model-name-in-``) must contain states and match an output file
    with the same ``id``, containing the corresponding derivatives.

    Returns ``(tuples, err)``, where ``err`` is either ``None`` or a string
    indicating an error occurred.
    """
    path = _test_root / 'data'

    # Gather input and output files
    inputs, outputs = {}, {}
    reg = re.compile(rf'^{model_name}-(in|out)-([^.]+).txt$')
    for path in (_test_root / 'data').glob(f'{model_name}-*-*.txt'):
        m = reg.match(path.name)
        if m is not None:
            d = inputs if m.group(1) == 'in' else outputs
            i = m.group(2)
            if i in d:
                return (f'Duplicate index in path {path}: {i} is already used'
                        f' by {d[i]}.')
            d[i] = path

    # Match inputs with outputs
    if inputs.keys() != outputs.keys():
        a = inputs.keys() - outputs.keys()
        b = outputs.keys() - inputs.keys()
        x = '\n  '.join(
            [f'{model_name}-in-{x}.txt' for x in a] +
            [f'{model_name}-out-{x}.txt' for x in b])
        return None, (f'Mismatched input and output files for {model_name}.\n'
                      f'No matches found for\n  {x}.')
    if len(inputs) == 0:
        return None, f'No test states found for {model_name}.mmt'

    # Join dictionaries and return
    io_pairs = {}
    for k, v in inputs.items():
        try:
            a = myokit.load_state(v, model)
            v = outputs[k]
            b = myokit.load_state(v, model)
        except (ValueError, TypeError, IOError) as e:
            return None, f'Unable to parse state/derivatives from {v}: {e}'
        io_pairs[k] = (a, b)

    return io_pairs, None
'''


def test_mmt_derivatives(model, pairs):
    """
    Tests if the ``model`` produces the correct derivatives for all given
    states.

    The dict ``pairs`` maps identifier strings to
    ``(state_vector, derivative_vector)`` pairs.
    """
    for idx, (states, expected) in pairs.items:
        model.set_state(states)





def find_cellml_files(path):
    return None, None, None










if __name__ == '__main__':
    print('Syntax-checking all model files!')
    print('This is used for regular online testing.')
    print('If you are not interested in testing the models,')
    print()
    print('  Press Ctrl+C to abort.')
    print()
    if '--write-expected' in sys.argv:
        _write_expected = True
    if not test_models():
        sys.exit(1)
