__copyright__ = """
Copyright (C) 2017 Natalie Beams
Copyright (C) 2022 Isuru Fernando
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

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pytools.obj_array import make_obj_array
from sumpy.kernel import (
    AxisSourceDerivative, AxisTargetDerivative, BiharmonicKernel, ElasticityKernel,
    Kernel, LaplaceKernel, StokesletKernel, StressletKernel, TargetPointMultiplier)
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel
from pytential.symbolic.typing import ExpansionLimitType, ExpressionT


__doc__ = """
.. autoclass:: RepresentationType
.. autofunction:: make_elasticity_wrapper
.. autofunction:: make_elasticity_double_layer_wrapper

.. autoclass:: ElasticityWrapperBase
.. autoclass:: ElasticityDoubleLayerWrapperBase

.. autoclass:: ElasticityWrapperNaive
.. autoclass:: ElasticityDoubleLayerWrapperNaive

.. autoclass:: ElasticityWrapperBiharmonic
.. autoclass:: ElasticityDoubleLayerWrapperBiharmonic

.. autoclass:: ElasticityWrapperYoshida
.. autoclass:: ElasticityDoubleLayerWrapperYoshida
"""

# {{{ entrypoints


class RepresentationType(enum.Enum):
    """Base kernel used in elasticity representations."""

    #: Use the standard diadic or triadic kernel.
    Naive = enum.auto()
    #: Use the Laplace kernel as a base kernel.
    Laplace = enum.auto()
    #: Use the biharmonic kernel as a base kernel.
    Biharmonic = 3


def make_elasticity_wrapper(
        dim: int,
        mu: Union[ExpressionT, str] = "mu",
        nu: Union[ExpressionT, str] = "nu",
        repr_type: RepresentationType = RepresentationType.Naive,
        ) -> "ElasticityWrapperBase":
    """Creates a :class:`ElasticityWrapperBase` object for the given inputs.

    If *nu* is :math:`0.5` (as a literal), this can also create an appropriate
    :func:`~pytential.symbolic.stokes.StokesletWrapperBase`.

    :param repr_type: representation to use in the elasticity kernel.
    :return: an appropriate :class:`ElasticityWrapperBase` for the given dimension
        or parameters.
    """
    if not isinstance(repr_type, RepresentationType):
        raise TypeError("'repr_type' must be a 'RepresentationType' enum value")

    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if isinstance(nu, str):
        nu = SpatialConstant(nu)

    if nu == 0.5:
        from pytential.symbolic.stokes import make_stokeslet_wrapper
        return make_stokeslet_wrapper(dim=dim, mu=mu, repr_type=repr_type)

    if repr_type == RepresentationType.Naive:
        return ElasticityWrapperNaive(dim=dim, mu=mu, nu=nu)
    elif repr_type == RepresentationType.Biharmonic:
        return ElasticityWrapperBiharmonic(dim=dim, mu=mu, nu=nu)
    elif repr_type == RepresentationType.Laplace:
        return ElasticityWrapperYoshida(dim=dim, mu=mu, nu=nu)
    else:
        raise AssertionError()


def make_elasticity_double_layer_wrapper(
        dim: int,
        mu: Union[ExpressionT, str] = "mu",
        nu: Union[ExpressionT, str] = "nu",
        repr_type: RepresentationType = RepresentationType.Naive
        ) -> "ElasticityDoubleLayerWrapperBase":
    """Creates a :class:`ElasticityWrapperBase` object for the given inputs.

    If *nu* is :math:`0.5` (as a literal), this can also create an appropriate
    :func:`~pytential.symbolic.stokes.StokesletWrapperBase`.

    :param repr_type: representation to use in the elasticity kernel.
    :return: an appropriate :class:`ElasticityWrapperBase` for the given dimension
        or parameters.
    """
    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if isinstance(nu, str):
        nu = SpatialConstant(nu)

    if nu == 0.5:
        from pytential.symbolic.stokes import make_stresslet_wrapper
        return make_stresslet_wrapper(dim=dim, mu=mu, repr_type=repr_type)

    if repr_type == RepresentationType.Naive:
        return ElasticityDoubleLayerWrapperNaive(dim=dim, mu=mu, nu=nu)
    elif repr_type == RepresentationType.Biharmonic:
        return ElasticityDoubleLayerWrapperBiharmonic(dim=dim, mu=mu, nu=nu)
    elif repr_type == RepresentationType.Laplace:
        return ElasticityDoubleLayerWrapperYoshida(dim=dim, mu=mu, nu=nu)
    else:
        raise AssertionError()

# }}}


# {{{ ABCs

@dataclass(frozen=True)
class ElasticityWrapperBase(ABC):
    """Wrapper class for the single-layer of the
    :class:`~sumpy.kernel.ElasticityKernel` kernel.

    The :meth:`apply` function returns the integral expressions needed for
    the displacement fields resulting from the convolution with the vector density.
    It is meant to work similarly to calling :func:`~pytential.symbolic.primitives.S`
    (which is a :class:`~pytential.symbolic.primitives.IntG`).

    .. autoattribute:: dim
    .. autoattribute:: mu
    .. autoattribute:: nu

    .. automethod:: apply
    .. automethod:: apply_derivative
    """

    #: Ambient dimension.
    dim: int
    #: Expression or value for the shear modulus.
    mu: ExpressionT
    #: Expression or value for Poisson's ratio.
    nu: ExpressionT

    @abstractmethod
    def apply(self,
              density_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        """Symbolic expressions the single-layer potential.

        We construct an object array of symbolic expressions for the vector
        resulting from integrating the dyadic kernel with *density_vec_sym*
        as the density.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        :arg extra_deriv_dirs: adds target derivatives to all the integral
            objects with the given derivative axis. Setting this to ``(i,)``
            is equivalent to calling :meth:`apply_derivative`.
        """

    def apply_derivative(self,
                         deriv_dir: int,
                         density_vec_sym: np.ndarray,
                         qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        """Symbolic derivative the single-layer potential.

        We construct an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        dyadic kernel with *density_vec_sym* as the density. This is equivalent
        to calling :meth:`apply` with ``(deriv_dir,)``.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        return self.apply(density_vec_sym, qbx_forced_limit,
                          extra_deriv_dirs=(deriv_dir,))


@dataclass(frozen=True)
class ElasticityDoubleLayerWrapperBase(ABC):
    """Wrapper class for the double-layer of the
    :class:`~sumpy.kernel.ElasticityKernel` kernel.

    The :meth:`apply` function returns the integral expressions needed for
    convolving the kernel with a vector density, and is meant to work
    similarly to :func:`~pytential.symbolic.primitives.D` (which is
    :class:`~pytential.symbolic.primitives.IntG`).

    .. autoattribute:: dim
    .. autoattribute:: mu
    .. autoattribute:: nu

    .. automethod:: apply
    .. automethod:: apply_derivative
    """

    #: Ambient dimension.
    dim: int
    #: Expression or value for the shear modulus.
    mu: ExpressionT
    #: Expression or value for Poisson's ratio.
    nu: ExpressionT

    @abstractmethod
    def apply(self,
              density_vec_sym: np.ndarray,
              dir_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        """Symbolic expressions for integrating double-layer potential.

        We construct an object array of symbolic expressions for the vector
        resulting from integrating the triadic kernel with density
        *density_vec_sym* and source direction vector *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the direction vector.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        :arg extra_deriv_dirs: adds target derivatives to all the integral
            objects with the given derivative axis.
        """

    def apply_derivative(self,
                         deriv_dir: int,
                         density_vec_sym: np.ndarray,
                         dir_vec_sym: np.ndarray,
                         qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        """Symbolic derivative of the double-layer potential.

        We construct an object array of symbolic expressions for the vector
        resulting from integrating the *deriv_dir* target derivative of the
        triadic kernel with density *density_vec_sym* and source direction
        vector *dir_vec_sym*.

        :arg deriv_dir: integer denoting the axis direction for the derivative.
        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector variable for the normal direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        return self.apply(density_vec_sym, dir_vec_sym, qbx_forced_limit,
                          extra_deriv_dirs=(deriv_dir,))

# }}}


# {{{ Naive and Biharmonic helpers

def _make_int_g(
        knl: Kernel,
        density_sym: Any,
        *,
        extra_deriv_dirs: Tuple[int, ...],
        **kwargs: Any) -> np.ndarray:
    for deriv_dir in extra_deriv_dirs:
        knl = AxisTargetDerivative(deriv_dir, knl)

    kernel_arg_names = {
        arg.loopy_arg.name
        for arg in (knl.get_args() + knl.get_source_args())
    }

    # When the kernel is Laplace, mu and nu are not kernel arguments
    # Also when nu==0.5, it's not a kernel argument to StokesletKernel
    for var_name in ["mu", "nu"]:
        if var_name not in kernel_arg_names:
            kwargs.pop(var_name)

    return sym.int_g_vec(knl, density_sym, **kwargs)


@dataclass(frozen=True)
class _ElasticityWrapperWithKernel(ElasticityWrapperBase):
    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError(
                f"Unsupported dimension for '{type(self).__name__}': {self.dim}")

    @property
    @abstractmethod
    def base_kernel(self) -> Optional[Kernel]:
        """The base kernel used in representing the vector kernel."""

    @cached_property
    def kernel_dict(self) -> Dict[Tuple[int, int], Kernel]:
        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{01}` is identical to :math:`T_{10}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.

        d = {}
        for i in range(self.dim):
            for j in range(i, self.dim):
                if self.nu == 0.5:
                    d[(i, j)] = StokesletKernel(dim=self.dim, icomp=i, jcomp=j)
                else:
                    d[(i, j)] = ElasticityKernel(dim=self.dim, icomp=i, jcomp=j)

                d[(j, i)] = d[(i, j)]

        return d

    def apply(self,
              density_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        sym_expr: List[ExpressionT] = [0] * self.dim
        for comp in range(self.dim):
            for i in range(self.dim):
                intg = _make_int_g(
                    self.kernel_dict[comp, i],
                    density_vec_sym[i],
                    extra_deriv_dirs=extra_deriv_dirs,
                    qbx_forced_limit=qbx_forced_limit,
                    mu=self.mu,
                    nu=self.nu)

                sym_expr[comp] += intg / (2 * (1 - self.nu))

        return make_obj_array(
            rewrite_using_base_kernel(sym_expr, base_kernel=self.base_kernel)
            )


ELASTICITY_DLP_LAPLACE_IDX = (-1, -1, -1)


@dataclass(frozen=True)
class _ElasticityDoubleLayerWrapperWithKernel(ElasticityDoubleLayerWrapperBase):
    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError(
                f"Unsupported dimension for '{type(self).__name__}': {self.dim}")

    @property
    @abstractmethod
    def base_kernel(self) -> Optional[Kernel]:
        """The base kernel used in representing the vector kernel."""

    @cached_property
    def kernel_dict(self) -> Dict[Tuple[int, int, int], Kernel]:
        d = {}

        for i in range(self.dim):
            for j in range(i, self.dim):
                for k in range(j, self.dim):
                    d[i, j, k] = (
                        StressletKernel(dim=self.dim, icomp=i, jcomp=j, kcomp=k))

        # The dictionary allows us to exploit symmetry -- that
        # :math:`T_{012}` is identical to :math:`T_{120}` -- and avoid creating
        # multiple expansions for the same kernel in a different ordering.
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if (i, j, k) in d:
                        continue

                    i0, j0, k0 = sorted([i, j, k])
                    d[i, j, k] = d[i0, j0, k0]

        # For elasticity (nu != 0.5), we need the Laplacian of the
        # BiharmonicKernel which is the LaplaceKernel.
        d[ELASTICITY_DLP_LAPLACE_IDX] = LaplaceKernel(self.dim)

        return d

    def _get_int_g(self,
                   idx: Tuple[int, int, int],
                   density_sym: Any,
                   dir_vec_sym: np.ndarray,
                   *,
                   qbx_forced_limit: ExpansionLimitType,
                   extra_deriv_dirs: Tuple[int, ...]):
        """
        Returns the convolution of the double layer of the elasticity kernel
        given by `idx` and its derivatives.
        """

        nu = self.nu
        kernel_indices = [idx] + [ELASTICITY_DLP_LAPLACE_IDX] * 3
        dir_vec_indices = [idx[-1], idx[1], idx[0], idx[2]]
        coeffs = [1, (1 - 2*nu)/self.dim, -(1 - 2*nu)/self.dim, -(1 - 2*nu)]
        extra_deriv_dirs_vec = [(), (idx[0],), (idx[1],), (idx[2],)]

        if idx[0] != idx[1]:
            coeffs[-1] = 0

        result: ExpressionT = 0
        for kernel_idx, dir_vec_idx, coeff, extra_deriv_dirs in (
                zip(kernel_indices, dir_vec_indices, coeffs, extra_deriv_dirs_vec)
                ):
            if coeff == 0:
                continue

            knl = self.kernel_dict[kernel_idx]
            result += _make_int_g(
                knl,
                density_sym * dir_vec_sym[dir_vec_idx],
                extra_deriv_dirs=extra_deriv_dirs + extra_deriv_dirs,
                qbx_forced_limit=qbx_forced_limit,
                mu=self.mu,
                nu=self.nu) * coeff

        return result / (2 * (1 - nu))

    def apply(self,
              density_vec_sym: np.ndarray,
              dir_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        sym_expr: List[ExpressionT] = [0] * self.dim

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sym_expr[comp] += self._get_int_g(
                        (comp, i, j),
                        density_vec_sym[i],
                        dir_vec_sym,
                        qbx_forced_limit=qbx_forced_limit,
                        extra_deriv_dirs=extra_deriv_dirs)

        return make_obj_array(
            rewrite_using_base_kernel(sym_expr, base_kernel=self.base_kernel)
            )

# }}}


# {{{ Naive

@dataclass(frozen=True)
class ElasticityWrapperNaive(_ElasticityWrapperWithKernel):
    """
    This method uses uses the base elasticity kernel and corresponds to
    :attr:`RepresentationType.Naive`.

    .. autoattribute:: base_kernel
    """

    @property
    def base_kernel(self) -> Optional[Kernel]:
        return None


class ElasticityDoubleLayerWrapperNaive(_ElasticityDoubleLayerWrapperWithKernel):
    """
    This method uses uses the base elasticity kernel and corresponds to
    :attr:`RepresentationType.Naive`.

    .. autoattribute:: base_kernel
    """

    @property
    def base_kernel(self) -> Optional[Kernel]:
        return None


# }}}


# {{{ Biharmonic

@dataclass(frozen=True)
class ElasticityWrapperBiharmonic(_ElasticityWrapperWithKernel):
    """
    This method uses uses the biharmonic kernel and corresponds to
    :attr:`RepresentationType.Biharmonic`.

    .. autoattribute:: base_kernel
    """

    @property
    def base_kernel(self) -> Optional[Kernel]:
        return BiharmonicKernel(self.dim)


class ElasticityDoubleLayerWrapperBiharmonic(
        _ElasticityDoubleLayerWrapperWithKernel):
    """
    This method uses uses the biharmonic kernel and corresponds to
    :attr:`RepresentationType.Biharmonic`.

    .. autoattribute:: base_kernel
    """

    @property
    def base_kernel(self) -> Optional[Kernel]:
        return BiharmonicKernel(self.dim)

# }}}


# {{{ Yoshida

def _apply_yoshida_single_and_double_layer(
        slp_density_vec_sym: np.ndarray,
        dlp_density_vec_sym: np.ndarray,
        dir_vec_sym: np.ndarray,
        *,
        qbx_forced_limit: ExpansionLimitType,
        slp_weight: float,
        dlp_weight: float,
        mu: ExpressionT,
        nu: ExpressionT,
        extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
    dim = dir_vec_sym.size
    if dim != 3:
        raise ValueError(f"Unsupported dimension: {dim}")

    if slp_density_vec_sym.shape != (dim,):
        raise ValueError(f"Single-layer density is not {dim}d")

    if dlp_density_vec_sym.shape != (dim,):
        raise ValueError(f"Double-layer density is not {dim}d")

    lame_lambda = 2 * nu * mu / (1 - 2 * nu)
    slp_weight *= -1

    def C(i: int, j: int, k: int, l: int) -> ExpressionT:   # noqa: E741
        result: ExpressionT = 0
        if i == j and k == l:
            result += lame_lambda
        if i == k and j == l:
            result += mu
        if i == l and j == k:
            result += mu
        return result * dlp_weight

    def add_extra_deriv_dirs(target_kernel: Kernel) -> Kernel:
        for deriv_dir in extra_deriv_dirs:
            target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

        return target_kernel

    def P(i: int, j: int, int_g: sym.IntG) -> sym.IntG:
        target_kernel = AxisTargetDerivative(i, int_g.target_kernel)
        deriv_target_kernel = (
            add_extra_deriv_dirs(TargetPointMultiplier(j, target_kernel)))

        result = -int_g.copy(target_kernel=deriv_target_kernel)
        if i == j:
            target_kernel = add_extra_deriv_dirs(int_g.target_kernel)
            result += (3 - 4 * nu) * int_g.copy(target_kernel=target_kernel)

        return result / (4 * mu * (1 - nu))

    def Q(i: int, int_g: sym.IntG) -> sym.IntG:
        assert isinstance(int_g, sym.IntG)

        target_kernel = add_extra_deriv_dirs(
            AxisTargetDerivative(i, int_g.target_kernel))

        res = int_g.copy(target_kernel=target_kernel)
        return res / (4 * mu * (1 - nu))

    kernel = LaplaceKernel(dim)
    source = sym.nodes(dim).as_vector()
    normal = dir_vec_sym
    sigma = dlp_density_vec_sym

    source_kernels = [None] * (dim + 1)
    for i in range(dim):
        source_kernels[i] = AxisSourceDerivative(i, kernel)
    source_kernels[dim] = kernel

    from itertools import product

    sym_expr: np.ndarray = np.zeros(dim, dtype=object)
    for i in range(dim):
        for k in range(dim):
            densities = [0] * (dim + 1)
            for l, j, m in product(range(dim), repeat=3):  # noqa: E741
                densities[l] += C(k, l, m, j)*normal[m]*sigma[j]
            densities[dim] += slp_weight * slp_density_vec_sym[k]

            int_g = sym.IntG(
                target_kernel=kernel,
                source_kernels=tuple(source_kernels),
                densities=tuple(densities),
                qbx_forced_limit=qbx_forced_limit)
            sym_expr[i] += P(i, k, int_g)

        densities = [0] * (dim + 1)
        for k in range(dim):
            for m, j, l in product(range(dim), repeat=3):   # noqa: E741
                densities[l] += C(k, l, m, j) * normal[m] * sigma[j] * source[k]
                if k == l:
                    densities[dim] += C(k, l, m, j) * normal[m] * sigma[j]

            densities[dim] += slp_weight * source[k] * slp_density_vec_sym[k]

        int_g = sym.IntG(
            target_kernel=kernel,
            source_kernels=tuple(source_kernels),
            densities=tuple(densities),
            qbx_forced_limit=qbx_forced_limit)
        sym_expr[i] += Q(i, int_g)

    return sym_expr


@dataclass(frozen=True)
class ElasticityWrapperYoshida(ElasticityWrapperBase):
    r"""Elasticity single-layer using Yoshida et al's method [Yoshida2001]_.

    This method uses uses Laplace derivatives and corresponds to
    :attr:`RepresentationType.Laplace`.

    .. [Yoshida2001] K.-I. Yoshida, N. Nishimura, S. Kobayashi,
        *Application of Fast Multipole Galerkin Boundary Integral Equation
        Method to Elastostatic Crack Problems in 3D*,
        International Journal for Numerical Methods in Engineering, Vol. 50,
        pp. 525--547, 2001,
        `DOI <https://doi.org/10.1002/1097-0207(20010130)50:3%3C525::aid-nme34%3E3.0.co;2-4>`__.
    """  # noqa: E501

    def __post_init__(self):
        if self.dim != 3:
            raise ValueError(
                f"Unsupported dimension for '{type(self).__name__}': {self.dim}")

    def apply(self,
              density_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        slp_density_vec_sym = density_vec_sym
        dlp_density_vec_sym = np.zeros(self.dim)
        dir_vec_sym = np.zeros(self.dim)

        return _apply_yoshida_single_and_double_layer(
            slp_density_vec_sym,
            dlp_density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            slp_weight=1,
            dlp_weight=0,
            mu=self.mu,
            nu=self.nu,
            extra_deriv_dirs=extra_deriv_dirs)


@dataclass(frozen=True)
class ElasticityDoubleLayerWrapperYoshida(ElasticityDoubleLayerWrapperBase):
    r"""Elasticity double-layer using Yoshida et al's method [Yoshida2001]_.

    This method uses uses Laplace derivatives and corresponds to
    :attr:`RepresentationType.Laplace`.
    """

    def __post_init__(self):
        if self.dim != 3:
            raise ValueError(
                f"Unsupported dimension for '{type(self).__name__}': {self.dim}")

    def apply(self,
              density_vec_sym: np.ndarray,
              dir_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        slp_density_vec_sym = np.zeros(self.dim)
        dlp_density_vec_sym = density_vec_sym

        return _apply_yoshida_single_and_double_layer(
            slp_density_vec_sym,
            dlp_density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            slp_weight=0,
            dlp_weight=1,
            mu=self.mu,
            nu=self.nu,
            extra_deriv_dirs=extra_deriv_dirs)


# }}}
