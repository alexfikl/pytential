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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from sumpy.kernel import (
    AxisSourceDerivative, AxisTargetDerivative, BiharmonicKernel, Kernel,
    LaplaceKernel, TargetPointMultiplier)
from sumpy.symbolic import SpatialConstant

from pytential import sym
from pytential.symbolic.elasticity import (
    ElasticityDoubleLayerWrapperBase, ElasticityWrapperBase, RepresentationType,
    _ElasticityDoubleLayerWrapperWithKernel, _ElasticityWrapperWithKernel)
from pytential.symbolic.pde.system_utils import rewrite_using_base_kernel
from pytential.symbolic.typing import ExpansionLimitType, ExpressionT


__doc__ = """
.. automethod:: make_stokeslet_wrapper
.. automethod:: make_stresslet_wrapper

.. autoclass:: StokesletWrapperBase
.. autoclass:: StressletWrapperBase

.. autoclass:: StokesOperator
.. autoclass:: HsiaoKressExteriorStokesOperator
.. autoclass:: HebekerExteriorStokesOperator
"""


# {{{ entrypoints

def make_stokeslet_wrapper(
        dim: int,
        mu: Union[ExpressionT, str] = "mu",
        repr_type: RepresentationType = RepresentationType.Naive,
        ) -> "StokesletWrapperBase":
    """Creates a :class:`StokesletWrapperBase` object for the given inputs.

    :param repr_type: representation to use in the elasticity kernel.
    :return: an appropriate :class:`StokesletWrapperBase` for the given dimension
        or parameters.
    """
    if not isinstance(repr_type, RepresentationType):
        raise TypeError("'repr_type' must be a 'RepresentationType' enum value")

    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if repr_type == RepresentationType.Naive:
        return StokesletWrapperNaive(dim=dim, mu=mu)
    elif repr_type == RepresentationType.Biharmonic:
        return StokesletWrapperBiharmonic(dim=dim, mu=mu)
    elif repr_type == RepresentationType.Laplace:
        return StokesletWrapperTornberg(dim=dim, mu=mu)
    else:
        raise AssertionError()


def make_stresslet_wrapper(
        dim: int,
        mu: Union[ExpressionT, str] = "mu",
        repr_type: RepresentationType = RepresentationType.Naive
        ) -> "StressletWrapperBase":
    """Creates a :class:`StressletWrapperBase` object for the given inputs.

    :param repr_type: representation to use in the elasticity kernel.
    :return: an appropriate :class:`StressletWrapperBase` for the given dimension
        or parameters.
    """
    if not isinstance(repr_type, RepresentationType):
        raise TypeError("'repr_type' must be a 'RepresentationType' enum value")

    if isinstance(mu, str):
        mu = SpatialConstant(mu)

    if repr_type == RepresentationType.Naive:
        return StressletWrapperNaive(dim=dim, mu=mu)
    elif repr_type == RepresentationType.Biharmonic:
        return StressletWrapperBiharmonic(dim=dim, mu=mu)
    elif repr_type == RepresentationType.Laplace:
        return StressletWrapperTornberg(dim=dim, mu=mu)
    else:
        raise AssertionError()

# }}}


# {{{ ABCs

def _stokeslet_apply_pressure(
        density_vec_sym: np.ndarray,
        qbx_forced_limit: ExpansionLimitType,
        extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
    dim = density_vec_sym.size

    # Pressure representation doesn't differ depending on the implementation
    # and is implemented in base class here.
    lknl = LaplaceKernel(dim=dim)

    sym_expr: ExpressionT = 0
    for i in range(dim):
        deriv_dirs = tuple(extra_deriv_dirs) + (i,)
        knl = lknl
        for deriv_dir in deriv_dirs:
            knl = AxisTargetDerivative(deriv_dir, knl)
        sym_expr += sym.int_g_vec(knl, density_vec_sym[i],
                                  qbx_forced_limit=qbx_forced_limit)

    return sym_expr


@dataclass(frozen=True, init=False)
class StokesletWrapperBase(ElasticityWrapperBase):
    """Wrapper class for the :class:`~sumpy.kernel.StokesletKernel` kernel.

    In addition to the methods in
    :class:`~pytential.symbolic.elasticity.ElasticityWrapperBase`, this class
    also provides :meth:`apply_stress` which applies symmetric viscous stress tensor
    in the requested direction and :meth:`apply_pressure`.

    .. automethod:: apply
    .. automethod:: apply_derivative
    .. automethod:: apply_pressure
    .. automethod:: apply_stress
    """

    def __init__(self, dim: int, mu: ExpressionT) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    def apply_pressure(self,
                       density_vec_sym: np.ndarray,
                       qbx_forced_limit: ExpansionLimitType,
                       extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
        """Symbolic expression for pressure field associated with the Stokeslet."""
        return _stokeslet_apply_pressure(
            density_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            extra_deriv_dirs=extra_deriv_dirs)

    def apply_stress(self,
                     density_vec_sym: np.ndarray,
                     dir_vec_sym: np.ndarray,
                     qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        Note that this computation is very similar to computing
        a double-layer potential with the Stresslet kernel in
        :class:`StressletWrapperBase`. The difference is that here the direction
        vector is applied at the target points, while in the Stresslet the
        direction is applied at the source points.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement apply_stress")


def _stresslet_apply_pressure(density_vec_sym: np.ndarray,
                              dir_vec_sym: np.ndarray,
                              qbx_forced_limit: ExpansionLimitType,
                              mu: ExpressionT,
                              extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
    dim = density_vec_sym.dim
    if dir_vec_sym.shape != (dim,):
        raise ValueError(f"Direction vector is not {dim}d")

    from itertools import product

    lknl = LaplaceKernel(dim=dim)
    sym_expr: ExpressionT = 0

    for i, j in product(range(dim), repeat=2):
        deriv_dirs = tuple(extra_deriv_dirs) + (i, j)
        knl = lknl
        for deriv_dir in deriv_dirs:
            knl = AxisTargetDerivative(deriv_dir, knl)
        sym_expr += 2.0 * mu * sym.int_g_vec(knl,
                                             density_vec_sym[i] * dir_vec_sym[j],
                                             qbx_forced_limit=qbx_forced_limit)

    return sym_expr


@dataclass(frozen=True, init=False)
class StressletWrapperBase(ElasticityDoubleLayerWrapperBase):
    """Wrapper class for the :class:`~sumpy.kernel.StressletKernel` kernel.

    In addition to the methods in
    :class:`pytential.symbolic.elasticity.ElasticityDoubleLayerWrapperBase`, this
    class also provides :meth:`apply_stress` which applies symmetric viscous stress
    tensor in the requested direction and :meth:`apply_pressure`.

    .. automethod:: apply
    .. automethod:: apply_derivative
    .. automethod:: apply_pressure
    .. automethod:: apply_stress
    """

    def __init__(self, dim: int, mu: ExpressionT) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    def apply_pressure(self,
                       density_vec_sym: np.ndarray,
                       dir_vec_sym: np.ndarray,
                       qbx_forced_limit: ExpansionLimitType,
                       extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
        """Symbolic expression for pressure field associated with the Stresslet.
        """
        return _stresslet_apply_pressure(
            density_vec_sym,
            dir_vec_sym,
            mu=self.mu,
            qbx_forced_limit=qbx_forced_limit,
            extra_deriv_dirs=extra_deriv_dirs)

    def apply_stress(self,
                     density_vec_sym: np.ndarray,
                     normal_vec_sym: np.ndarray,
                     dir_vec_sym: np.ndarray,
                     qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        r"""Symbolic expression for viscous stress applied to a direction.

        Returns a vector of symbolic expressions for the force resulting
        from the viscous stress

        .. math::

            -p \delta_{ij} + \mu (\nabla_i u_j + \nabla_j u_i)

        applied in the direction of *dir_vec_sym*.

        :arg density_vec_sym: a symbolic vector variable for the density vector.
        :arg normal_vec_sym: a symbolic vector variable for the normal vectors
            (outward facing normals at source locations).
        :arg dir_vec_sym: a symbolic vector for the application direction.
        :arg qbx_forced_limit: the *qbx_forced_limit* argument to be passed on
            to :class:`~pytential.symbolic.primitives.IntG`.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement apply_stress")

# }}}


# {{{ Stokeslet/StressletWrapper Naive and Biharmonic

@dataclass(frozen=True, init=False)
class _StokesletWrapperWithKernel(_ElasticityWrapperWithKernel):
    def __init__(self, dim: int, mu: ExpressionT) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    @property
    @abstractmethod
    def stresslet_obj(self) -> "_StressletWrapperWithKernel":
        pass

    def apply_pressure(self,
                       density_vec_sym: np.ndarray,
                       qbx_forced_limit: ExpansionLimitType,
                       extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
        sym_expr = _stokeslet_apply_pressure(
            density_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            extra_deriv_dirs=extra_deriv_dirs)

        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

    def apply_stress(self,
                     density_vec_sym: np.ndarray,
                     dir_vec_sym: np.ndarray,
                     qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        sym_expr: np.ndarray = np.zeros((self.dim,), dtype=object)
        stresslet_obj = self.stresslet_obj

        for comp in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    int_g = _make_int_g(
                        stresslet_obj.kernel_dict[comp, i, j],
                        density_vec_sym[j],
                        qbx_forced_limit=qbx_forced_limit,
                        mu=self.mu,
                        nu=self.nu)

                    sym_expr[comp] += dir_vec_sym[i] * int_g

        return sym_expr


@dataclass(frozen=True, init=False)
class _StressletWrapperWithKernel(_ElasticityDoubleLayerWrapperWithKernel):
    def __init__(self, dim: int, mu: ExpressionT) -> None:
        super().__init__(dim=dim, mu=mu, nu=0.5)

    def apply_pressure(self,
                       density_vec_sym: np.ndarray,
                       dir_vec_sym: np.ndarray,
                       qbx_forced_limit: ExpansionLimitType,
                       extra_deriv_dirs: Tuple[int, ...] = ()) -> ExpressionT:
        sym_expr = _stresslet_apply_pressure(
            density_vec_sym,
            dir_vec_sym,
            mu=self.mu,
            qbx_forced_limit=qbx_forced_limit,
            extra_deriv_dirs=extra_deriv_dirs)

        res, = rewrite_using_base_kernel([sym_expr], base_kernel=self.base_kernel)
        return res

    def apply_stress(self,
                     density_vec_sym: np.ndarray,
                     normal_vec_sym: np.ndarray,
                     dir_vec_sym: np.ndarray,
                     qbx_forced_limit: ExpansionLimitType) -> np.ndarray:
        sym_expr = np.empty((self.dim,), dtype=object)

        # Build velocity derivative matrix
        sym_grad_matrix = np.empty((self.dim, self.dim), dtype=object)
        for i in range(self.dim):
            sym_grad_matrix[:, i] = self.apply_derivative(i, density_vec_sym,
                                     normal_vec_sym, qbx_forced_limit)

        for comp in range(self.dim):

            # First, add the pressure term:
            sym_expr[comp] = - dir_vec_sym[comp] * self.apply_pressure(
                                            density_vec_sym, normal_vec_sym,
                                            qbx_forced_limit)

            # Now add the velocity derivative components
            for j in range(self.dim):
                sym_expr[comp] = sym_expr[comp] + (
                                    dir_vec_sym[j] * self.mu * (
                                        sym_grad_matrix[comp][j]
                                        + sym_grad_matrix[j][comp])
                                        )
        return sym_expr

# }}}


# {{{ Naive

@dataclass(frozen=True, init=False)
class StokesletWrapperNaive(_StokesletWrapperWithKernel):
    @property
    def base_kernel(self) -> Optional[Kernel]:
        return None

    @property
    def stresslet_obj(self) -> "StressletWrapperNaive":
        return StressletWrapperNaive(dim=self.dim, mu=self.mu)


@dataclass(frozen=True, init=False)
class StressletWrapperNaive(_StressletWrapperWithKernel):
    @property
    def base_kernel(self) -> Optional[Kernel]:
        return None


StokesletWrapperBase.register(StokesletWrapperNaive)
StressletWrapperBase.register(StressletWrapperNaive)

# }}}


# {{{ Biharmonic

@dataclass(frozen=True, init=False)
class StokesletWrapperBiharmonic(_StokesletWrapperWithKernel):
    @property
    def base_kernel(self) -> Optional[Kernel]:
        return BiharmonicKernel(self.dim)

    @property
    def stresslet_obj(self) -> "StressletWrapperBiharmonic":
        return StressletWrapperBiharmonic(dim=self.dim, mu=self.mu)


@dataclass(frozen=True, init=False)
class StressletWrapperBiharmonic(_StressletWrapperWithKernel):
    @property
    def base_kernel(self) -> Optional[Kernel]:
        return BiharmonicKernel(self.dim)


StokesletWrapperBase.register(StokesletWrapperBiharmonic)
StressletWrapperBase.register(StressletWrapperBiharmonic)

# }}}


# {{{ Tornberg

def _make_int_g(
        target_kernel: Kernel,
        source_kernels: Tuple[Kernel, ...],
        densities: Tuple[ExpressionT],
        qbx_forced_limit: ExpansionLimitType) -> ExpressionT:
    if len(source_kernels) != len(densities):
        raise ValueError(
            f"'source_kernels' have length '{len(source_kernels)}' and "
            f"'densities' have length '{len(densities)}'")

    new_densities = tuple([density for density in densities if density != 0])
    if not new_densities:
        return 0

    return sym.IntG(
        target_kernel=target_kernel,
        source_kernels=tuple([
            kernel for kernel, d in zip(source_kernels, densities) if d != 0
            ]),
        densities=new_densities,
        qbx_forced_limit=qbx_forced_limit)


def _apply_tornberg_single_and_double_layer(
        stokeslet_density_vec_sym: np.ndarray,
        stresslet_density_vec_sym: np.ndarray,
        dir_vec_sym: np.ndarray,
        *,
        qbx_forced_limit: ExpansionLimitType,
        stokeslet_weight: float,
        stresslet_weight: float,
        mu: ExpressionT,
        extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
    dim, = dir_vec_sym.shape
    if stokeslet_density_vec_sym.shape != (dim,):
        raise ValueError(f"Single-layer density is not {dim}d")

    if stresslet_density_vec_sym.shape != (dim,):
        raise ValueError(f"Double-layer density is not {dim}d")

    sym_expr = np.zeros((dim,), dtype=object)
    source = sym.nodes(dim).as_vector()

    # The paper in [1] ignores the scaling we use in the Stokeslet/Stresslet
    # and gives formulae for the kernel expression only
    # stokeslet_weight = StokesletKernel.global_scaling_const /
    #    LaplaceKernel.global_scaling_const
    # stresslet_weight = StressletKernel.global_scaling_const /
    #    LaplaceKernel.global_scaling_const
    stresslet_weight *= 3.0
    stokeslet_weight *= -0.5*mu**(-1)

    laplace_kernel = LaplaceKernel(dim=dim)
    common_source_kernels = tuple([
        AxisSourceDerivative(k, laplace_kernel) for k in range(dim)
        ] + [laplace_kernel])

    for i in range(dim):
        for j in range(dim):
            densities = tuple([
                (stresslet_weight / 6.0) * (
                    stresslet_density_vec_sym[k] * dir_vec_sym[j]
                    + stresslet_density_vec_sym[j] * dir_vec_sym[k])
                for k in range(dim)
                ] + [stokeslet_weight * stokeslet_density_vec_sym[j]])

            target_kernel = (
                TargetPointMultiplier(j, AxisTargetDerivative(i, laplace_kernel)))
            for deriv_dir in extra_deriv_dirs:
                target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

            sym_expr[i] -= _make_int_g(
                target_kernel=target_kernel,
                source_kernels=common_source_kernels,
                densities=densities,
                qbx_forced_limit=qbx_forced_limit)

            if i == j:
                target_kernel = laplace_kernel
                for deriv_dir in extra_deriv_dirs:
                    target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

                sym_expr[i] += _make_int_g(
                    target_kernel=target_kernel,
                    source_kernels=common_source_kernels,
                    densities=densities,
                    qbx_forced_limit=qbx_forced_limit)

        common_density0 = sum(
            source[k] * stresslet_density_vec_sym[k] for k in range(dim))
        common_density1 = sum(
            source[k] * dir_vec_sym[k] for k in range(dim))
        common_density2 = sum(
            source[k] * stokeslet_density_vec_sym[k] for k in range(dim))
        densities = tuple([
            (stresslet_weight / 6.0) * (
                common_density0 * dir_vec_sym[k]
                + common_density1 * stresslet_density_vec_sym[k])
            for k in range(dim)
            ] + [stokeslet_weight * common_density2])

        target_kernel = AxisTargetDerivative(i, laplace_kernel)
        for deriv_dir in extra_deriv_dirs:
            target_kernel = AxisTargetDerivative(deriv_dir, target_kernel)

        sym_expr[i] += _make_int_g(
            target_kernel=target_kernel,
            source_kernels=common_source_kernels,
            densities=densities,
            qbx_forced_limit=qbx_forced_limit)

    return sym_expr


@dataclass(frozen=True, init=False)
class StokesletWrapperTornberg(StokesletWrapperBase):
    """A Stokeslet wrapper using Tornberg and Greengard method [Tornberg2008]_.

    This method uses uses Laplace derivatives and corresponds to
    :attr:`~pytential.symbolic.elasticity.RepresentationType.Laplace`.

    .. [Tornberg2008] A.-K. Tornberg, L. Greengard,
        *A Fast Multipole Method for the Three-Dimensional Stokes Equations*,
        Journal of Computational Physics, Vol. 227, pp. 1613--1619, 2008,
        `DOI <https://doi.org/10.1016/j.jcp.2007.06.029>`__.
    """

    def apply(self,
              density_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        stokeslet_density_vec_sym = density_vec_sym
        stresslet_density_vec_sym = np.zeros(self.dim)
        dir_vec_sym = np.zeros(self.dim)

        return _apply_tornberg_single_and_double_layer(
            stokeslet_density_vec_sym,
            stresslet_density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=1,
            stresslet_weight=0,
            mu=self.mu,
            extra_deriv_dirs=extra_deriv_dirs)


@dataclass(frozen=True, init=False)
class StressletWrapperTornberg(StressletWrapperBase):
    """A Stresslet wrapper using Tornberg and Greengard's method [Tornberg2008]_.

    This method uses uses Laplace derivatives and corresponds to
    :attr:`~pytential.symbolic.elasticity.RepresentationType.Laplace`.
    """

    def apply(self,
              density_vec_sym: np.ndarray,
              dir_vec_sym: np.ndarray,
              qbx_forced_limit: ExpansionLimitType,
              extra_deriv_dirs: Tuple[int, ...] = ()) -> np.ndarray:
        stokeslet_density_vec_sym = np.zeros(self.dim)
        stresslet_density_vec_sym = density_vec_sym

        return _apply_tornberg_single_and_double_layer(
            stokeslet_density_vec_sym,
            stresslet_density_vec_sym,
            dir_vec_sym,
            qbx_forced_limit=qbx_forced_limit,
            stokeslet_weight=0,
            stresslet_weight=1,
            mu=self.mu,
            extra_deriv_dirs=extra_deriv_dirs)

# }}}


# {{{ base Stokes operator

class StokesOperator:
    """
    .. attribute:: ambient_dim
    .. attribute:: side

    .. automethod:: __init__
    .. automethod:: get_density_var
    .. automethod:: prepare_rhs
    .. automethod:: operator

    .. automethod:: velocity
    .. automethod:: pressure
    """

    def __init__(self, ambient_dim, side, stokeslet, stresslet, mu):
        """
        :arg ambient_dim: dimension of the ambient space.
        :arg side: :math:`+1` for exterior or :math:`-1` for interior.
        """
        if side not in [+1, -1]:
            raise ValueError(f"invalid evaluation side: {side}")

        self.ambient_dim = ambient_dim
        self.side = side

        if stresslet is None:
            stresslet = make_stresslet_wrapper(dim=self.ambient_dim, mu=mu)

        if stokeslet is None:
            stokeslet = make_stokeslet_wrapper(dim=self.ambient_dim, mu=mu)

        self.stokeslet = stokeslet
        self.stresslet = stresslet

    @property
    def dim(self):
        return self.ambient_dim - 1

    def get_density_var(self, name="sigma"):
        """
        :returns: a symbolic vector corresponding to the density.
        """
        return sym.make_sym_vector(name, self.ambient_dim)

    def prepare_rhs(self, b):
        """
        :returns: a (potentially) modified right-hand side *b* that matches
            requirements of the representation.
        """
        return b

    @abstractmethod
    def operator(self, sigma):
        """
        :returns: the integral operator that should be solved to obtain the
            density *sigma*.
        """
        raise NotImplementedError

    @abstractmethod
    def velocity(self, sigma, *, normal, qbx_forced_limit=None):
        """
        :returns: a representation of the velocity field in the Stokes flow.
        """
        raise NotImplementedError

    @abstractmethod
    def pressure(self, sigma, *, normal, qbx_forced_limit=None):
        """
        :returns: a representation of the pressure in the Stokes flow.
        """
        raise NotImplementedError

# }}}


# {{{ exterior Stokes flow

class HsiaoKressExteriorStokesOperator(StokesOperator):
    """Representation for 2D Stokes Flow based on [HsiaoKress1985]_.

    Inherits from :class:`StokesOperator`.

    .. [HsiaoKress1985] G. C. Hsiao and R. Kress, *On an Integral Equation for
        the Two-Dimensional Exterior Stokes Problem*,
        Applied Numerical Mathematics, Vol. 1, 1985,
        `DOI <https://doi.org/10.1016/0168-9274(85)90029-7>`__.

    .. automethod:: __init__
    """

    def __init__(self, *, omega, alpha=1.0, eta=1.0,
                 stokeslet=None, stresslet=None, mu=None):
        r"""
        :arg omega: farfield behaviour of the velocity field, as defined
            by :math:`A` in [HsiaoKress1985]_ Equation 2.3.
        :arg alpha: real parameter :math:`\alpha > 0`.
        :arg eta: real parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning.
        """
        super().__init__(ambient_dim=2, side=+1, stokeslet=stokeslet,
                stresslet=stresslet, mu=mu)

        # NOTE: in [hsiao-kress], there is an analysis on a circle, which
        # recommends values in
        #   1/2 <= alpha <= 2 and max(1/alpha, 1) <= eta <= min(2, 2/alpha)
        # so we choose alpha = eta = 1, which seems to be in line with some
        # of the presented numerical results too.

        self.omega = omega
        self.alpha = alpha
        self.eta = eta

    def _farfield(self, qbx_forced_limit):
        source_dofdesc = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        length = sym.integral(self.ambient_dim, self.dim, 1, dofdesc=source_dofdesc)
        result = self.stresslet.apply_single_and_double_layer(
                -self.omega / length, [0]*self.ambient_dim, [0]*self.ambient_dim,
                qbx_forced_limit=qbx_forced_limit, stokeslet_weight=1,
                stresslet_weight=0)
        return result

    def _operator(self, sigma, normal, qbx_forced_limit):
        # NOTE: we set a dofdesc here to force the evaluation of this integral
        # on the source instead of the target when using automatic tagging
        # see :meth:`pytential.symbolic.mappers.LocationTagger._default_dofdesc`
        dd = sym.DOFDescriptor(None, discr_stage=sym.QBX_SOURCE_STAGE1)
        int_sigma = sym.integral(self.ambient_dim, self.dim, sigma, dofdesc=dd)

        meanless_sigma = sym.cse(sigma - sym.mean(self.ambient_dim,
            self.dim, sigma, dofdesc=dd))

        result = self.eta * self.alpha / (2.0 * np.pi) * int_sigma
        result += self.stresslet.apply_single_and_double_layer(meanless_sigma,
                sigma, normal, qbx_forced_limit=qbx_forced_limit,
                stokeslet_weight=-self.eta, stresslet_weight=1)

        return result

    def prepare_rhs(self, b):
        return b + self._farfield(qbx_forced_limit=+1)

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. K. 1985 Equation 2.18
        return -0.5 * self.side * sigma - self._operator(
            sigma, normal, qbx_forced_limit)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. K. 1985 Equation 2.16
        return -self._farfield(qbx_forced_limit) \
                - self._operator(sigma, normal, qbx_forced_limit)

    def pressure(self, sigma, *, normal, qbx_forced_limit=2):
        # FIXME: H. K. 1985 Equation 2.17
        raise NotImplementedError


class HebekerExteriorStokesOperator(StokesOperator):
    """Representation for 3D Stokes Flow based on [Hebeker1986]_.

    Inherits from :class:`StokesOperator`.

    .. [Hebeker1986] F. C. Hebeker, *Efficient Boundary Element Methods for
        Three-Dimensional Exterior Viscous Flow*, Numerical Methods for
        Partial Differential Equations, Vol. 2, 1986,
        `DOI <https://doi.org/10.1002/num.1690020404>`__.

    .. automethod:: __init__
    """

    def __init__(self, *, eta=None, stokeslet=None, stresslet=None, mu=None):
        r"""
        :arg eta: a parameter :math:`\eta > 0`. Choosing this parameter well
            can have a non-trivial effect on the conditioning of the operator.
        """

        super().__init__(ambient_dim=3, side=+1, stokeslet=stokeslet,
                stresslet=stresslet, mu=mu)

        # NOTE: eta is chosen here based on H. 1986 Figure 1, which is
        # based on solving on the unit sphere
        if eta is None:
            eta = 0.75

        self.eta = eta
        self.laplace_kernel = LaplaceKernel(3)

    def _operator(self, sigma, normal, qbx_forced_limit):
        result = self.stresslet.apply_single_and_double_layer(sigma,
                sigma, normal, qbx_forced_limit=qbx_forced_limit,
                stokeslet_weight=self.eta, stresslet_weight=1,
                extra_deriv_dirs=())
        return result

    def operator(self, sigma, *, normal, qbx_forced_limit="avg"):
        # NOTE: H. 1986 Equation 17
        return -0.5 * self.side * sigma - self._operator(sigma,
            normal, qbx_forced_limit)

    def velocity(self, sigma, *, normal, qbx_forced_limit=2):
        # NOTE: H. 1986 Equation 16
        return -self._operator(sigma, normal, qbx_forced_limit)

    def pressure(self, sigma, *, normal, qbx_forced_limit=2):
        # FIXME: not given in H. 1986, but should be easy to derive using the
        # equivalent single-/double-layer pressure kernels
        raise NotImplementedError

# }}}
