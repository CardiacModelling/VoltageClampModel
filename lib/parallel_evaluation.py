#
# Utility classes to perform multiple model evaluations sequentially or in
# parallell.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import gc
import os
import sys
import time
import traceback
import multiprocessing
try:
    # Python 3
    import queue
except ImportError:
    import Queue as queue


def evaluate(f, x, parallel=False, args=None, nworkers=None):
    """
    Evaluates the list of functions ``f`` on every value present in ``x`` and
    returns a sequence of evaluations ``f[i](x[i])``.

    To run the evaluations on all available cores, set ``parallel=True``. For
    details see :class:`ParallelEvaluator`.

    Extra arguments to pass to ``f`` can be given in the optional list of 
    tuple ``args``. If used, each ``f`` will be called as 
    ``f[i](x[i], *args[i])``.
    """
    if parallel:
        evaluator = ParallelEvaluator(f, args, nworkers=nworkers)
    else:
        evaluator = SequentialEvaluator(f, args)
    return evaluator.evaluate(x)


class Evaluator(object):
    """
    *Abstract class*

    Interface for classes that take a list of functions (or callable objects)
    ``f(x)`` and evaluate it for list of input values ``x``. This interface is
    shared by a parallel and a sequential implementation, allowing easy
    switching between parallel or sequential implementations of the same
    algorithm.

    Arguments:

    ``function``
        A list of functions or other callable objects ``f`` that takes a list 
        of value ``x`` and returns an evaluation ``f[i](x[i])``.
    ``args``
        An optional tuple containing extra arguments to each ``f``. If 
        ``args`` is specified, ``f`` will be called as 
        ``f[i](x[i], *args[i])``.

    """
    def __init__(self, function, args=None):
        for f in function:
            if not callable(f):
                raise ValueError('The given function must be callable.')
        self._function = function
        if args is None:
            self._args = [()] * len(function)
        else:
            for a in args:
                if type(a) != tuple:
                    raise ValueError(
                        'The argument `args` must be either None or a tuple.')
            if len(args) != len(function):
                raise ValueError(
                        'The argument `args` must have the same length as'
                        ' the given function.')
            self._args = args

    def evaluate(self, positions):
        """
        Evaluate the function for every value in the sequence ``positions``.

        Returns a list with the returned evaluations.
        """
        try:
            len(positions)
        except TypeError:
            raise ValueError(
                'The argument `positions` must be a sequence of input values'
                ' to the evaluator\'s function.')
        if len(positions) != len(self._function):
            raise ValueError(
                'The argument `positions` must have the same length as the'
                ' evaluator\'s function.')
        return self._evaluate(positions)

    def _evaluate(self, positions):
        """
        Internal version of :meth:`evaluate()`.
        """
        raise NotImplementedError


class ParallelEvaluator(Evaluator):
    """
    *Extends:* :class:`Evaluator`

    Evaluates a single-valued function object for any set of input values
    given, using all available cores.

    Shares an interface with the :class:`SequentialEvaluator`, allowing
    parallelism to be switched on and off with minimal hassle. Parallelism
    takes a little time to be set up, so as a general rule of thumb it's only
    useful for if the total run-time is at least ten seconds (anno 2015).

    By default, the number of processes ("workers") used to evaluate the
    function is set equal to the number of CPU cores reported by python's
    ``multiprocessing`` module. To override the number of workers used, set
    ``nworkers`` to some integer greater than ``0``.

    There are two important caveats for using multiprocessing to evaluate
    functions:

      1. Processes don't share memory. This means the function to be
         evaluated will be duplicated (via pickling) for each process (see
         `Avoid shared state <http://docs.python.org/2/library/\
multiprocessing.html#all-platforms>`_ for details).
      2. On windows systems your code should be within an
         ``if __name__ == '__main__':`` block (see `Windows
         <https://docs.python.org/2/library/multiprocessing.html#windows>`_
         for details).

    Arguments:

    ``function``
        The function to evaluate
    ``nworkers``
        The number of worker processes to use. If left at the default value
        ``nworkers=None`` the number of workers will equal the number of CPU
        cores in the machine this is run on. In many cases this will provide
        good performance.
    ``max_tasks_per_worker``
        Python garbage collection does not seem to be optimized for
        multi-process function evaluation. In many cases, some time can be
        saved by refreshing the worker processes after every
        ``max_tasks_per_worker`` evaluations. This number can be tweaked for
        best performance on a given task / system.
    ``args``
        An optional tuple containing extra arguments to the objective function.

    The evaluator will keep it's subprocesses alive and running until it is
    tidied up by garbage collection.

    Note that while this class uses multiprocessing, it is not thread/process
    safe itself: It should not be used by more than a single thread/process at
    a time.
    """
    def __init__(
            self, function, nworkers=None, max_tasks_per_worker=500,
            args=None):
        super(ParallelEvaluator, self).__init__(function, args)
        # Determine number of workers
        if nworkers is None:
            self._nworkers = max(1, multiprocessing.cpu_count())
        else:
            self._nworkers = int(nworkers)
            if self._nworkers < 1:
                raise ValueError(
                    'Number of workers must be an integer greater than 0 or'
                    ' `None` to use the default value.')
        # Create empty set of workers
        self._workers = []
        # Maximum tasks per worker (for some reason, this saves memory)
        self._max_tasks = int(max_tasks_per_worker)
        if self._max_tasks < 1:
            raise ValueError(
                'Maximum tasks per worker should be at least 1 (but probably'
                ' much greater).')
        # Queue with tasks
        self._tasks = multiprocessing.Queue()
        # Queue with results
        self._results = multiprocessing.Queue()
        # Queue used to add an exception object and context to
        self._errors = multiprocessing.Queue()
        # Flag set if an error is encountered
        self._error = multiprocessing.Event()

    def __del__(self):
        # Cancel everything
        try:
            self._stop()
        except Exception:
            pass

    def _clean(self):
        """
        Cleans up any dead workers & return the number of workers tidied up.
        """
        cleaned = 0
        for k in range(len(self._workers) - 1, -1, -1):
            w = self._workers[k]
            if w.exitcode is not None:
                w.join()
                cleaned += 1
                del(self._workers[k])
        if cleaned:
            gc.collect()
        return cleaned

    def _populate(self):
        """
        Populates (but usually repopulates) the worker pool.
        """
        for k in range(self._nworkers - len(self._workers)):
            w = _Worker(
                self._function,
                self._args,
                self._tasks,
                self._results,
                self._max_tasks,
                self._errors,
                self._error,
            )
            self._workers.append(w)
            w.start()

    def _evaluate(self, positions):
        """
        Evaluate all tasks in parallel, in batches of size self._max_tasks.
        """
        # Ensure task and result queues are empty
        # For some reason these lines block when running on windows
        # if not (self._tasks.empty() and self._results.empty()):
        #    raise Exception('Unhandled tasks/results left in queues.')
        # Clean up any dead workers
        self._clean()
        # Ensure worker pool is populated
        self._populate()
        # Start
        try:
            # Enqueue all tasks (non-blocking)
            for k, x in enumerate(positions):
                self._tasks.put((k, x))
            # Collect results (blocking)
            n = len(positions)
            m = 0
            results = [0] * n
            while m < n and not self._error.is_set():
                time.sleep(0.001)   # This is really necessary
                # Retrieve all results
                try:
                    while True:
                        i, f = self._results.get(block=False)
                        results[i] = f
                        m += 1
                except queue.Empty:
                    pass
                # Clean dead workers
                if self._clean():
                    # Repolate
                    self._populate()
        except (IOError, EOFError):
            # IOErrors can originate from the queues as a result of issues in
            # the subprocesses. Check if the error flag is set. If it is, let
            # the subprocess exception handling deal with it. If it isn't,
            # handle it here.
            if not self._error.is_set():
                self._stop()
                raise
            # TODO: Maybe this should be something like while(error is not set)
            # wait for it to be set, then let the subprocess handle it...
        except (Exception, SystemExit, KeyboardInterrupt):
            # All other exceptions, including Ctrl-C and user triggered exits
            # should (1) cause all child processes to stop and (2) bubble up to
            # the caller.
            self._stop()
            raise
        # Error in worker threads
        if self._error.is_set():
            errors = self._stop()
            # Raise exception
            if errors:
                pid, trace = errors[0]
                raise Exception('Exception in subprocess:' + trace)
            else:
                raise Exception('Unknown exception in subprocess.')
        # Return results
        return results

    def _stop(self):
        """
        Forcibly halts the workers
        """
        time.sleep(0.1)

        # Terminate workers
        for w in self._workers:
            if w.exitcode is None:
                w.terminate()
        for w in self._workers:
            if w.is_alive():
                w.join()
        self._workers = []

        # Free memory
        gc.collect()

        # Clear queues
        def clear(q):
            items = []
            try:
                while True:
                    items.append(q.get(timeout=0.1))
            except (queue.Empty, IOError, EOFError):
                pass
            return items

        clear(self._tasks)
        clear(self._results)
        errors = clear(self._errors)

        # Create new queues & error event
        self._tasks = multiprocessing.Queue()
        self._results = multiprocessing.Queue()
        self._errors = multiprocessing.Queue()
        self._error = multiprocessing.Event()

        # Return errors
        return errors


class SequentialEvaluator(Evaluator):
    """
    *Extends:* :class:`Evaluator`

    Evaluates a list of functions (or callable objects) for a list of input
    values.

    Runs sequentially, but shares an interface with the
    :class:`ParallelEvaluator`, allowing parallelism to be switched on/off.

    Arguments:

    ``function``
        The list of functions to evaluate.
    ``args``
        An optional list of tuple containing extra arguments to ``f``. If 
        ``args`` is specified, ``f`` will be called as 
        ``f[i](x[i], *args[i])``.

    Returns a list containing the calculated functions evaluations.
    """
    def __init__(self, function, args=None):
        super(SequentialEvaluator, self).__init__(function, args)

    def _evaluate(self, positions):
        scores = [0] * len(positions)
        for k, x in enumerate(positions):
            scores[k] = self._function[k](x, *self._args[k])
        return scores


#
# Note: For Windows multiprocessing to work, the _Worker can never be a nested
# class!
#
class _Worker(multiprocessing.Process):
    """
    *Extends:* ``multiprocessing.Process``

    Worker class for use with :class:`ParallelEvaluator`.

    Evaluates a single-valued function for every point in a ``tasks`` queue
    and places the results on a ``results`` queue.

    Keeps running until it's given the string "stop" as a task.

    Arguments:

    ``function``
        The list of functions to optimize.
    ``args``
        A list of (possibly empty) tuples containing extra input arguments to
        the objective list of functions.
    ``tasks``
        The queue to read tasks from. Tasks are stored as tuples
        ``(i, p)`` where ``i`` is a task id and ``p`` is the
        position to evaluate.
    ``results``
        The queue to store results in. Results are stored as
        tuples ``(i, p, r)`` where ``i`` is the task id, ``p`` is
        the position evaluated (which can be updated by the
        refinement method!) and ``r`` is the result at ``p``.
    ``max_tasks``
        The maximum number of tasks to perform before dying.
    ``errors``
        A queue to store exceptions on
    ``error``
        This flag will be set by the worker whenever it encounters an
        error.

    """
    def __init__(
            self, function, args, tasks, results, max_tasks, errors, error):
        super(_Worker, self).__init__()
        self.daemon = True
        self._function = function
        self._args = args
        self._tasks = tasks
        self._results = results
        self._max_tasks = max_tasks
        self._errors = errors
        self._error = error

    def run(self):
        # Worker processes should never write to stdout or stderr.
        # This can lead to unsafe situations if they have been redicted to
        # a GUI task such as writing to the IDE console.
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            for k in range(self._max_tasks):
                i, x = self._tasks.get()
                f = self._function[i](x, *self._args[i])
                self._results.put((i, f))
                # Check for errors in other workers
                if self._error.is_set():
                    return
        except (Exception, KeyboardInterrupt, SystemExit):
            self._errors.put((self.pid, traceback.format_exc()))
            self._error.set()

