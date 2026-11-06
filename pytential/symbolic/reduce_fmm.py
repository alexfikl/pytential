from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
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

import logging
from collections import defaultdict
from math import prod
from typing import TYPE_CHECKING

import sympy as sp

import pymbolic.primitives as prim
from pymbolic.geometric_algebra.mapper import WalkMapper
from pymbolic.mapper import Mapper
from sumpy.kernel import AxisSourceDerivative, AxisTargetDerivative, KernelWrapper


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymbolic.typing import ArithmeticExpression

    from pytential.symbolic.primitives import IntG

logger = logging.getLogger(__name__)


__all__ = (
    "reduce_number_of_fmms",
)

__doc__ = """
.. autofunction:: reduce_number_of_fmms
"""


# {{{ Reduce number of FMMs - main routine


def reduce_number_of_fmms(
        int_gs: Sequence[IntG], *,
        source_dependent_variables: Sequence[ArithmeticExpression] | None = None,
    ):
    """
    Reduce the number of FMMs needed for a system of
    :class:`~pytential.symbolic.primitives.IntG` objects.

    This is done by converting the ``IntG`` object to a matrix of polynomials
    with :math:`d` variables corresponding to :math:`d` dimensions, where each
    variable represents a (target) derivative operator along one of the axes.
    All the properties of derivative operator that we want are reflected in the
    properties of the polynomial including addition, multiplication and exact
    polynomial division.

    This matrix is factored into two matrices, where the first matrix
    represents a transformation at the target, and the second matrix represents
    a transformation at the source.

    .. note::

        There is no argument for target-dependent variables as the algorithm
        assumes that there are no target-dependent variables passed to this
        function. (where a "source/target-dependent variable" is a symbolic
        variable that evaluates to a vector discretized on the sources/targets)

    :arg source_dependent_variables: list of :class:`pymbolic.primitives.Expression`
        objects. When reducing FMMs, consider only these variables as dependent
        on source. For eg: densities, source derivative vectors.
    """
    # {{{ check inputs

    # NOTE: might want to refactor this into its own function?

    if not int_gs:
        return int_gs

    ref_int_g = int_gs[0]
    target_kernel = ref_int_g.target_kernel
    dim = target_kernel.dim
    base_kernel = target_kernel.get_base_kernel()

    if not all(int_g.target_kernel.dim == dim for int_g in int_gs):
        raise ValueError(f"all IntGs must have the same dimension: {dim}")

    kernel_arguments = dict(ref_int_g.kernel_arguments)
    for i, int_g in enumerate(int_gs[1:]):
        # NOTE: in each IntG, source_kernels are guaranteed to have the same
        # base_kernel as the target_kernel, so we don't need to check more
        if int_g.target_kernel != target_kernel:
            raise ValueError(
                f"IntG[{i}] has a mismatched `target_kernel`: "
                f"{int_g.target_kernel!r} (expected {target_kernel!r})"
            )

        if ref_int_g.qbx_forced_limit != int_g.qbx_forced_limit:
            raise ValueError(
                f"IntG[{i}] has a mismatched 'qbx_forced_limit': "
                f"{int_g.qbx_forced_limit!r} (expected {ref_int_g.qbx_forced_limit!r})"
            )

        if ref_int_g.source != int_g.source:
            raise ValueError(
                f"IntG[{i}] has a mismatched 'source' geometry: "
                f"{int_g.source!r} (expected {ref_int_g.source!r})"
            )

        # NOTE: check that all common kernel arguments agree
        for k, v in int_g.kernel_arguments.items():
            if (ref := kernel_arguments.get(k)) is not None and v != ref:
                raise ValueError(
                    f"IntG[{i}] has a mismatched kernel argument '{k}': "
                    f"{ref} (expected {v})"
                )
            else:
                kernel_arguments[k] = v

    if source_dependent_variables is None:
        source_dependent_variables = get_all_source_dependent_variables(int_gs)

    # }}}

    # convert to a matrix
    x = sp.symbols(f"_x0:{dim}")
    mat, source_exprs = make_int_g_poly_matrix(
        int_gs, source_dependent_variables, x
    )
    mat = sp.nsimplify(mat)

    # create polynomial ring
    poly_ring = sp.EX.old_poly_ring(*x, order=sp.grevlex)

    # try to quotient out the base kernel PDE to avoid.. what??
    try:
        pde = base_kernel.get_pde_as_diff_op()
        if len(pde.eqs) > 1:
            ring = poly_ring
        else:
            eq = next(iter(pde.eqs))
            sym_pde = sum(
                coeff * prod(x[i]**ident.mi[i] for i in range(dim))
                for ident, coeff in eq.items()
            )
            ring = poly_ring / [sym_pde]
    except NotImplementedError:
        ring = poly_ring

    # Factor the matrix into two
    try:
        left_factor, right_factor = _make_syzygy_factorization(mat, x, ring)
    except ValueError:
        logger.debug("could not find a factorization for %s", mat)
        return int_gs

    # If there are n inputs and m outputs,
    #
    #  - matrix: R^{m x n},
    #  - LHS: R^{m x k},
    #  - RHS: R^{k x n}.
    #
    # If k is greater than or equal to n we are gaining nothing.
    # Return as is.
    if right_factor.shape[0] >= mat.shape[1]:
        return int_gs

    base_kernel = int_gs[0].source_kernels[0].get_base_kernel()
    base_int_g = int_gs[0].copy(target_kernel=base_kernel,
            source_kernels=(base_kernel,), densities=(1,))

    # Convert polynomials back to IntGs with source derivatives
    source_int_gs = [[_convert_source_poly_to_int_g_derivs(
        as_poly(expr, x), base_int_g,
        x) for expr in row] for row in right_factor.tolist()]

    # For each row in the right factor, merge the IntGs to one IntG
    # to get a total of k IntGs.
    source_int_gs_merged = []
    for i in range(right_factor.shape[0]):
        source_kernels = []
        densities = []
        for j in range(right_factor.shape[1]):
            if source_int_gs[i][j] == 0:
                continue
            new_densities = [density * source_exprs[j] for density in
                    source_int_gs[i][j].densities]
            source_kernels.extend(source_int_gs[i][j].source_kernels)
            densities.extend(new_densities)
            nonzero_intg = source_int_gs[i][j]
        source_int_gs_merged.append(nonzero_intg.copy(
            source_kernels=tuple(source_kernels), densities=tuple(densities)))

    # Now that we have the IntG expressions depending on the source
    # we now have to attach the target dependent derivatives.
    res = [0]*left_factor.shape[0]
    for i in range(left_factor.shape[0]):
        for j in range(left_factor.shape[1]):
            res[i] += _convert_target_poly_to_int_g_derivs(
                    left_factor[i, j].as_poly(*x, domain=sp.EX),
                    int_gs[i], source_int_gs_merged[j])

    return res


def as_poly(expr, axis_vars):
    res = expr.as_poly(*axis_vars, domain=sp.EX)
    if res is None:
        return expr.simplify().as_poly(*axis_vars, domain=sp.EX)
    else:
        return res


# }}}


# {{{ get_all_source_dependent_variables

class GatherAllSourceDependentVariables(WalkMapper):
    def __init__(self):
        self.vars = {}

    def map_variable(self, expr):
        from sumpy.symbolic import SpatialConstant
        if not isinstance(expr, SpatialConstant):
            self.vars[expr] = True

    def map_list(self, exprs):
        for expr in exprs:
            self.rec(expr)

    def map_int_g(self, expr):
        self.vars[expr] = True
        for density in expr.densities:
            self.rec(density)

    def map_node_coordinate_component(self, expr):
        self.vars[expr] = True

    map_num_reference_derivative = map_node_coordinate_component
    map_interpolation = map_node_coordinate_component


def get_all_source_dependent_variables(
        int_gs: Sequence[IntG]
    ) -> Sequence[ArithmeticExpression]:
    # FIXME: what about kernel arguments? normal, parameters, etc.
    mapper = GatherAllSourceDependentVariables()
    for int_g in int_gs:
        for density in int_g.densities:
            mapper(density)

    return tuple(mapper.vars)


# }}}


# {{{ convert IntG expressions to a matrix


# FIXME: make pymbolic.CoefficientCollector more general?

class CoefficientCollector(Mapper):
    """From a density expression, extracts expressions that need to be
    evaluated for each source and coefficients for each expression.

    For eg: when this mapper is given as ``s*(s + 2) + 3`` input,
    it returns {s**2: 1, s: 2, 1: 3}.

    This is more general than
    :class:`pymbolic.mapper.coefficient.CoefficientCollector` as that deals
    only with linear expressions, but this collector works for polynomial
    expressions too.
    """

    def __init__(
            self,
            source_dependent_variables: Sequence[ArithmeticExpression]
        ) -> None:
        self.source_dependent_variables = source_dependent_variables

    def __call__(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        return super().__call__(expr)

    def map_sum(self, expr):
        stride_dicts = [self.rec(ch) for ch in expr.children]

        result = defaultdict(lambda: 0)
        for stride_dict in stride_dicts:
            for var, stride in stride_dict.items():
                result[var] += stride
        return dict(result)

    def map_algebraic_leaf(self, expr):
        if expr in self.source_dependent_variables:
            return {expr: 1}
        else:
            return {1: expr}

    def map_node_coordinate_component(self, expr):
        return {expr: 1}

    map_num_reference_derivative = map_node_coordinate_component

    def map_common_subexpression(self, expr):
        return {expr: 1}

    def map_subscript(self, expr):
        if expr in self.source_dependent_variables or \
                expr.aggregate in self.source_dependent_variables:
            return {expr: 1}
        else:
            return {1: expr}

    def map_constant(self, expr):
        return {1: expr}

    def map_product(self, expr):
        if len(expr.children) > 2:
            # rewrite products of more than two children as a nested
            # product and recurse to make it easier to handle.
            left = expr.children[0]
            right = prim.Product(tuple(expr.children[1:]))
            new_prod = prim.Product((left, right))
            return self.rec(new_prod)
        elif len(expr.children) == 1:
            return self.rec(expr.children[0])
        elif len(expr.children) == 0:
            return {1: 1}
        left, right = expr.children
        d_left = self.rec(left)
        d_right = self.rec(right)
        d = defaultdict(lambda: 0)
        for var_left, coeff_left in d_left.items():
            for var_right, coeff_right in d_right.items():
                d[var_left*var_right] += coeff_left*coeff_right
        return dict(d)

    def map_quotient(self, expr):
        d_num = self.rec(expr.numerator)
        d_den = self.rec(expr.denominator)
        if len(d_den) > 1:
            raise ValueError
        den_var, den_coeff = next(iter(d_den.items()))

        return {num_var/den_var: num_coeff/den_coeff for
            num_var, num_coeff in d_num.items()}

    def map_power(self, expr):
        d_base = self.rec(expr.base)
        d_exponent = self.rec(expr.exponent)
        # d_exponent should look like {1: k}
        if len(d_exponent) > 1 or 1 not in d_exponent:
            raise RuntimeError("nonlinear expression")
        exp, = d_exponent.values()
        if exp == 1:
            return d_base
        if len(d_base) > 1:
            raise NotImplementedError("powers are not implemented")
        (var, coeff), = d_base.items()
        return {var**exp: coeff**exp}

    rec = __call__


def _kernel_source_derivatives_as_poly(kernel, x):
    r"""Converts a :class:`sumpy.kernel.Kernel` object to a polynomial.

    A :class:`~sumpy.kernel.Kernel` represents a derivative operator and the
    derivative operator is converted to a polynomial with variables given by
    *x*. For example, for source *x*, the derivative operator, is converted as
    follows

    .. math::

        \frac{\mathrm{d}}{\mathrm{d} x_1}
        \frac{\mathrm{d}}{\mathrm{d} x_2}
        +
        \frac{\mathrm{d}}{\mathrm{d} x_1}
        \quad \implies \quad
        x_1 x_2 + x_1.
    """

    if isinstance(kernel, AxisSourceDerivative):
        poly = _kernel_source_derivatives_as_poly(kernel.inner_kernel, x)
        return -x[kernel.axis] * poly

    if isinstance(kernel, KernelWrapper):
        raise TypeError(f"{type(kernel).__name__!r} not supported")

    return 1


def make_int_g_poly_matrix(
        int_gs: Sequence[IntG],
        source_dependent_variables: Sequence[ArithmeticExpression],
        x: Sequence[sp.Symbol],
    ) -> tuple[sp.Matrix, Sequence[ArithmeticExpression]]:
    """Create a matrix from a list of :class:`~pytential.symbolic.primitives.IntG`
    objects.

    Each expression contains ``source_dependent_variables``. Each element in
    the matrix is a multi-variate polynomial and the variables in the
    polynomial are the *x* input variables. Each polynomial represents a
    derivative operator.

    Number of rows of the returned matrix is equal to the number of ``int_gs`` and
    the number of columns is equal to the number of input source dependent
    expressions.
    """
    from sumpy.symbolic import PymbolicToSympyMapper

    to_sympy = PymbolicToSympyMapper()
    coefficient_collector = CoefficientCollector(source_dependent_variables)

    source_exprs = []
    matrix = []

    for int_g in int_gs:
        row = [0] * len(source_exprs)
        for density, source_kernel in zip(int_g.densities, int_g.source_kernels,
                                          strict=True):
            d = coefficient_collector(density)
            for source_expr, coeff in d.items():
                if source_expr not in source_exprs:
                    source_exprs.append(source_expr)
                    row += [0]

                poly = _kernel_source_derivatives_as_poly(source_kernel, x)
                row[source_exprs.index(source_expr)] += poly * to_sympy(coeff)

        matrix.append(row)

    # NOTE: pad the matrix so that all the rows have equal length
    for row in matrix:
        row += [0]*(len(source_exprs) - len(row))

    return sp.Matrix(matrix), source_exprs


# }}}


# {{{ factor the matrix

def minimal_generating_set(m):
    """Computes a module with a minimal generating set as its generators
    from an input module with possibly redundant generators. The output
    does not necessarily have the smallest minimal generating set.
    """
    gens = list(m.gens)
    nonzero = [x for x in gens if any(y != m.ring.zero for y in x)]
    basis = nonzero.copy()
    for x in nonzero:
        others = basis[:]
        others.remove(x)
        if x in m.container.submodule(*others):
            basis = others
    return m.container.submodule(*basis)


def _convert_to_matrix(module, *generators):
    result = []
    for syzygy in module:
        row = []
        for dmp in syzygy:
            try:
                d = dmp.data.to_dict()
            except AttributeError:
                d = dmp.to_dict()
            row.append(sp.Poly(d, *generators,
                domain=sp.EX).as_expr())
        result.append(row)
    return sp.Matrix(result)


def syzygy_module(m, generators, ring):
    """Takes as input a module of polynomials with domain :class:`sympy.EX`
    represented as a matrix and returns the syzygy module.
    The syzygy module *S* satisfies S m = 0 and is a left nullspace of the matrix.
    Using :class:`sympy.EX` because that represents the domain with any symbolic
    element. Usually we need an Integer or Rational domain, but since there can be
    unrelated symbols like *mu* in the expression, we need to use a symbolic domain.
    """

    module = ring.free_module(m.shape[1]).submodule(
        *[m[i, :] for i in range(m.shape[0])])
    return minimal_generating_set(module.syzygy_module())


def _make_syzygy_factorization(mat, axis_vars, ring):
    """Return a "rank-revealing" factorisation of the matrix.

    For a matrix M, we want to find a factorisation such that M = L R
    with minimum number of columns of L. The polynomials represent
    derivative operators and therefore division is not well defined.
    To avoid divisions, we work in a polynomial ring which doesn't
    have division either. The revealed rank might not be the actual
    rank.

    To get a good factorisation, what we do is first find a matrix
    such that S M.T = 0 where S is the syzygy module converted to a matrix.
    It can also be referred to as the left nullspace of the matrix.
    Then, M S.T = 0 which implies that M is in the space spanned by
    the syzygy module of S.T and to get R we get the transpose of that.
    """
    if mat.shape[0] < mat.shape[1]:
        # For sympy performance, we use a tall and skinny matrix
        L, R = _make_syzygy_factorization(mat.T, axis_vars, ring)
        return R.T, L.T

    S_module = syzygy_module(mat.T, axis_vars, ring)
    S = _convert_to_matrix(S_module.gens, *axis_vars)

    if len(S) == 0:
        return mat, sp.eye(mat.shape[1])
    R_module = syzygy_module(S.T, axis_vars, ring)
    R = _convert_to_matrix(R_module.gens, *axis_vars)
    L_module = [R_module.in_terms_of_generators(mat[i, :])
           for i in range(mat.shape[0])]
    L = _convert_to_matrix(L_module, *axis_vars)

    if 0:
        L2 = R.LUsolve(sp.Matrix(mat),
                iszerofunc=lambda x: x.simplify() == 0)
        L2 = L2.applyfunc(lambda x: x.simplify())
        return L2, R

    return L, R

# }}}


# {{{ convert factors back into IntGs

def _convert_source_poly_to_int_g_derivs(poly, orig_int_g, axis_vars):
    """This does the opposite of :func:`_kernel_source_derivs_as_poly`
    and converts a polynomial back to a source derivative
    operator. First it is converted to a :class:`sumpy.kernel.Kernel`
    and then to a :class:`~pytential.symbolic.primitives.IntG`.
    """
    if poly == 0:
        return 0

    from sumpy.symbolic import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    orig_kernel = orig_int_g.source_kernels[0]
    source_kernels = []
    densities = []
    for monom, coeff in poly.terms():
        kernel = orig_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisSourceDerivative(idim, kernel)
        source_kernels.append(kernel)
        # (-1) below is because d/dx f(c - x) = - f'(c - x)
        densities.append(to_pymbolic(coeff) * (-1)**sum(monom))

    from pytential.symbolic.mappers import flatten
    return orig_int_g.copy(
        source_kernels=tuple(source_kernels),
        densities=tuple(flatten(densities)),
    )


def _convert_target_poly_to_int_g_derivs(poly, orig_int_g, rhs_int_g):
    """This does the opposite of :func:`_kernel_source_derivs_as_poly`
    and converts a polynomial back to a target derivative
    operator. It is applied to a :class:`~pytential.symbolic.primitives.IntG`
    object and returns a new instance.
    """
    from sumpy.symbolic import SympyToPymbolicMapper
    to_pymbolic = SympyToPymbolicMapper()

    result = 0
    for monom, coeff in poly.terms():
        kernel = orig_int_g.target_kernel
        for idim, rep in enumerate(monom):
            for _ in range(rep):
                kernel = AxisTargetDerivative(idim, kernel)
        result += orig_int_g.copy(target_kernel=kernel,
                source_kernels=rhs_int_g.source_kernels,
                densities=rhs_int_g.densities) * to_pymbolic(coeff)

    return result

# }}}

# vim: fdm=marker
