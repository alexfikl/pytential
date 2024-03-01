__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2023 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import sumpy.symbolic as sym


def sort_arrays_together(
        *arys: Any,
        key: Optional[Callable[[Any], Any]] = None
        ) -> Iterable[Tuple[Any, ...]]:
    """Sort a sequence of arrays by considering them
    as an array of sequences using the given sorting key

    :param key: a function that takes in a tuple of values
                and returns a value to compare.
    """
    return zip(*sorted(zip(*arys), key=key))


def pytest_teardown_function():
    from pyopencl.tools import clear_first_arg_caches
    clear_first_arg_caches()

    from sympy.core.cache import clear_cache
    clear_cache()

    import sumpy
    sumpy.code_cache.clear_in_mem_cache()

    from loopy import clear_in_mem_caches
    clear_in_mem_caches()

    import gc
    gc.collect()

    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)


def chop(expr: sym.Basic, rtol: float) -> sym.Basic:
    """Chop numeric values to zero or the nearest integer.

    This replaces all floating points numbers that are close to a given integer
    (in relative tolerance *rtol*) with that integer.

    :returns: a modified *expr* with all floating point values chopped to their
        closest integer.
    """
    nums = expr.atoms(sym.Number)
    replace_dict: Dict[sym.Number, Union[int, float]] = {}

    for num in nums:
        new_num = float(num)
        if abs(new_num) < rtol:
            replace_dict[num] = 0
        else:
            new_num_int = int(new_num)
            if abs(new_num_int - new_num) < rtol * abs(new_num):
                replace_dict[num] = new_num_int
            else:
                replace_dict[num] = new_num

    return expr.xreplace(replace_dict)


def forward_substitution(
        L: sym.Matrix,
        b: sym.Matrix,
        postprocess_division: Callable[[sym.Basic], sym.Basic],
        ) -> sym.Matrix:
    """Solve a lower triangular linear system :math:`L x = b`.

    This applies the callable *postprocess_division* after each division.
    """
    n = len(b)
    res = sym.Matrix(b)

    for i in range(n):
        for j in range(i):
            res[i] -= L[i, j] * res[j]

        res[i] = postprocess_division(res[i] / L[i, i])

    return res


def backward_substitution(
        U: sym.Matrix,
        b: sym.Matrix,
        postprocess_division: Callable[[sym.Basic], sym.Basic],
        ) -> sym.Matrix:
    """Solve a lower triangular linear system :math:`U x = b`.

    This applies the callable *postprocess_division* after each division.
    """
    n = len(b)
    res = sym.Matrix(b)

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            res[i] -= U[i, j] * res[j]

        res[i] = postprocess_division(res[i] / U[i, i])

    return res


def solve_from_lu(
        L: sym.Matrix,
        U: sym.Matrix,
        perm: Iterable[Tuple[int, int]],
        b: sym.Matrix,
        postprocess_division: Callable[[sym.Basic], sym.Basic]
        ) -> sym.Matrix:
    """Solve a linear system with a given :math:`(L, U, P)` factorization.

    Intermediate results are expanded to avoid an explosion of the expression
    trees. This calls :func:`forward_substitution` and :func:`backward_substitution`
    to solve the triangular systems.

    :param L: lower triangular matrix.
    :param U: upper triangular matrix.
    :param perm: permutation matrix.
    :param b: a column vector to solve for.
    :param postprocess_division: callable that is called after each division.
    """
    # Permute first
    res = sym.Matrix(b)
    for p, q in perm:
        res[p], res[q] = res[q], res[p]

    return backward_substitution(
        U,
        forward_substitution(L, res, postprocess_division),
        postprocess_division,
        )

# vim: foldmethod=marker
