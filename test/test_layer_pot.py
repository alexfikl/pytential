from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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


import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pytools import Record
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)
from sumpy.visualization import FieldPlotter
from pytential import bind, sym, norm

import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


def make_circular_point_group(ambient_dim, npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x):
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = center[:, np.newaxis] + radius*np.vstack((np.cos(t), np.sin(t)))
    return result


# {{{ geometry test

def test_geometry(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 30
    order = 5

    mesh = make_curve_mesh(partial(ellipse, 1),
            np.linspace(0, 1, nelements+1),
            order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr = Discretization(cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    import pytential.symbolic.primitives as prim
    area_sym = prim.integral(2, 1, 1)

    area = bind(discr, area_sym)(queue)

    err = abs(area-2*np.pi)
    print(err)
    assert err < 1e-3

# }}}


# {{{ ellipse eigenvalues

@pytest.mark.parametrize(["ellipse_aspect", "mode_nr", "qbx_order"], [
    # Run with FMM
    (1, 5, 3),
    (1, 6, 3),
    # (2, 5, 3), /!\ FIXME: Does not achieve sufficient FMM precision

    # Run without FMM
    (1, 5, 4),
    (1, 6, 4),
    (1, 7, 5),
    (2, 5, 4),
    (2, 6, 4),
    (2, 7, 5),
    ])
def test_ellipse_eigenvalues(ctx_getter, ellipse_aspect, mode_nr, qbx_order):
    logging.basicConfig(level=logging.INFO)

    print("ellipse_aspect: %s, mode_nr: %d, qbx_order: %d" % (
            ellipse_aspect, mode_nr, qbx_order))

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    target_order = 8

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from pytential.qbx import QBXLayerPotentialSource
    from pytools.convergence import EOCRecorder

    s_eoc_rec = EOCRecorder()
    d_eoc_rec = EOCRecorder()
    sp_eoc_rec = EOCRecorder()

    if ellipse_aspect != 1:
        nelements_values = [60, 100, 150, 200]
    else:
        nelements_values = [30, 70]

    # See
    #
    # [1] G. J. Rodin and O. Steinbach, "Boundary Element Preconditioners
    # for Problems Defined on Slender Domains", SIAM Journal on Scientific
    # Computing, Vol. 24, No. 4, pg. 1450, 2003.
    # http://dx.doi.org/10.1137/S1064827500372067

    for nelements in nelements_values:
        mesh = make_curve_mesh(partial(ellipse, ellipse_aspect),
                np.linspace(0, 1, nelements+1),
                target_order)

        fmm_order = qbx_order + 3
        if fmm_order > 6:
            # FIXME: for now
            fmm_order = False

        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx, _ = QBXLayerPotentialSource(pre_density_discr, 4*target_order,
                qbx_order, fmm_order=fmm_order).with_refinement()

        density_discr = qbx.density_discr
        nodes = density_discr.nodes().with_queue(queue)

        if 0:
            # plot geometry, centers, normals

            from pytential.qbx.utils import get_centers_on_side
            centers = get_centers_on_side(qbx, 1)

            nodes_h = nodes.get()
            centers_h = [centers[0].get(), centers[1].get()]
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            pt.plot(centers_h[0], centers_h[1], "o")
            normal = bind(qbx, sym.normal())(queue).as_vector(np.object)
            pt.quiver(nodes_h[0], nodes_h[1],
                    normal[0].get(), normal[1].get())
            pt.gca().set_aspect("equal")
            pt.show()

        angle = cl.clmath.atan2(nodes[1]*ellipse_aspect, nodes[0])

        ellipse_fraction = ((1-ellipse_aspect)/(1+ellipse_aspect))**mode_nr

        # (2.6) in [1]
        J = cl.clmath.sqrt(  # noqa
                cl.clmath.sin(angle)**2
                + (1/ellipse_aspect)**2 * cl.clmath.cos(angle)**2)

        from sumpy.kernel import LaplaceKernel
        lap_knl = LaplaceKernel(2)

        # {{{ single layer

        sigma = cl.clmath.cos(mode_nr*angle)/J

        s_sigma_op = bind(qbx, sym.S(lap_knl, sym.var("sigma")))
        s_sigma = s_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        s_eigval = 1/(2*mode_nr) * (1 + (-1)**mode_nr * ellipse_fraction)

        # (2.12) in [1]
        s_sigma_ref = s_eigval*J*sigma

        if 0:
            #pt.plot(s_sigma.get(), label="result")
            #pt.plot(s_sigma_ref.get(), label="ref")
            pt.plot((s_sigma_ref-s_sigma).get(), label="err")
            pt.legend()
            pt.show()

        s_err = (
                norm(density_discr, queue, s_sigma - s_sigma_ref)
                /
                norm(density_discr, queue, s_sigma_ref))
        s_eoc_rec.add_data_point(qbx.h_max, s_err)

        # }}}

        # {{{ double layer

        sigma = cl.clmath.cos(mode_nr*angle)

        d_sigma_op = bind(qbx,
                sym.D(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
        d_sigma = d_sigma_op(queue=queue, sigma=sigma)

        # SIGN BINGO! :)
        d_eigval = -(-1)**mode_nr * 1/2*ellipse_fraction

        d_sigma_ref = d_eigval*sigma

        if 0:
            pt.plot(d_sigma.get(), label="result")
            pt.plot(d_sigma_ref.get(), label="ref")
            pt.legend()
            pt.show()

        if ellipse_aspect == 1:
            d_ref_norm = norm(density_discr, queue, sigma)
        else:
            d_ref_norm = norm(density_discr, queue, d_sigma_ref)

        d_err = (
                norm(density_discr, queue, d_sigma - d_sigma_ref)
                /
                d_ref_norm)
        d_eoc_rec.add_data_point(qbx.h_max, d_err)

        # }}}

        if ellipse_aspect == 1:
            # {{{ S'

            sigma = cl.clmath.cos(mode_nr*angle)

            sp_sigma_op = bind(qbx,
                    sym.Sp(lap_knl, sym.var("sigma"), qbx_forced_limit="avg"))
            sp_sigma = sp_sigma_op(queue=queue, sigma=sigma)
            sp_eigval = 0

            sp_sigma_ref = sp_eigval*sigma

            sp_err = (
                    norm(density_discr, queue, sp_sigma - sp_sigma_ref)
                    /
                    norm(density_discr, queue, sigma))
            sp_eoc_rec.add_data_point(qbx.h_max, sp_err)

            # }}}

    print("Errors for S:")
    print(s_eoc_rec)
    required_order = qbx_order + 1
    assert s_eoc_rec.order_estimate() > required_order - 1.5

    print("Errors for D:")
    print(d_eoc_rec)
    required_order = qbx_order
    assert d_eoc_rec.order_estimate() > required_order - 1.5

    if ellipse_aspect == 1:
        print("Errors for S':")
        print(sp_eoc_rec)
        required_order = qbx_order
        assert sp_eoc_rec.order_estimate() > required_order - 1.5

# }}}


# {{{ integral equation test backend

def run_int_eq_test(cl_ctx, queue, case, resolution):
    mesh = case.get_mesh(resolution, case.target_order)
    print("%d elements" % mesh.nelements)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

    source_order = 4*case.target_order

    refiner_extra_kwargs = {}

    if case.k != 0:
        refiner_extra_kwargs["kernel_length_scale"] = 5/case.k

    if case.fmm_backend is None:
        fmm_order = False
    else:
        fmm_order = case.qbx_order + 5

    qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=source_order, qbx_order=case.qbx_order,
            fmm_order=fmm_order, fmm_backend=case.fmm_backend)

    if case.use_refinement:
        qbx, _ = qbx.with_refinement(**refiner_extra_kwargs)

    density_discr = qbx.density_discr

    # {{{ plot geometry

    if 0:
        if mesh.ambient_dim == 2:
            # show geometry, centers, normals
            nodes_h = density_discr.nodes().get(queue=queue)
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            normal = bind(density_discr, sym.normal(2))(queue).as_vector(np.object)
            pt.quiver(nodes_h[0], nodes_h[1],
                    normal[0].get(queue), normal[1].get(queue))
            pt.gca().set_aspect("equal")
            pt.show()

        elif mesh.ambient_dim == 3:
            from meshmode.discretization.visualization import make_visualizer
            bdry_vis = make_visualizer(queue, density_discr, case.target_order)

            bdry_normals = bind(density_discr, sym.normal(3))(queue)\
                    .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("bdry_normals", bdry_normals),
                ])

        else:
            raise ValueError("invalid mesh dim")

    # }}}

    # {{{ set up operator

    from pytential.symbolic.pde.scalar import (
            DirichletOperator,
            NeumannOperator)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if case.k:
        knl = HelmholtzKernel(mesh.ambient_dim)
        knl_kwargs = {"k": sym.var("k")}
        concrete_knl_kwargs = {"k": case.k}
    else:
        knl = LaplaceKernel(mesh.ambient_dim)
        knl_kwargs = {}
        concrete_knl_kwargs = {}

    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    if case.bc_type == "dirichlet":
        op = DirichletOperator(knl, case.loc_sign, use_l2_weighting=True,
                kernel_arguments=knl_kwargs)
    elif case.bc_type == "neumann":
        op = NeumannOperator(knl, case.loc_sign, use_l2_weighting=True,
                 use_improved_operator=False, kernel_arguments=knl_kwargs,
                 alpha=case.neumann_alpha)
    else:
        assert False

    op_u = op.operator(sym.var("u"))

    # }}}

    # {{{ set up test data

    if case.loc_sign < 0:
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.inner_radius
    else:
        test_src_geo_radius = case.inner_radius
        test_tgt_geo_radius = case.outer_radius

    point_sources = make_circular_point_group(
            mesh.ambient_dim, 10, test_src_geo_radius,
            func=lambda x: x**1.5)
    test_targets = make_circular_point_group(
            mesh.ambient_dim, 20, test_tgt_geo_radius)

    np.random.seed(22)
    source_charges = np.random.randn(point_sources.shape[1])
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1e-15

    source_charges_dev = cl.array.to_device(queue, source_charges)

    # }}}

    # {{{ establish BCs

    from pytential.source import PointPotentialSource
    from pytential.target import PointsTarget

    point_source = PointPotentialSource(cl_ctx, point_sources)

    pot_src = sym.IntG(
        # FIXME: qbx_forced_limit--really?
        knl, sym.var("charges"), qbx_forced_limit=None, **knl_kwargs)

    test_direct = bind((point_source, PointsTarget(test_targets)), pot_src)(
            queue, charges=source_charges_dev, **concrete_knl_kwargs)

    if case.bc_type == "dirichlet":
        bc = bind((point_source, density_discr), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs)

    elif case.bc_type == "neumann":
        bc = bind(
                (point_source, density_discr),
                sym.normal_derivative(
                    qbx.ambient_dim, pot_src, where=sym.DEFAULT_TARGET)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)

    # }}}

    # {{{ solve

    bound_op = bind(qbx, op_u)

    rhs = bind(density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u", dtype, **concrete_knl_kwargs),
            rhs,
            tol=case.gmres_tol,
            progress=True,
            hard_failure=True)

    print("gmres state:", gmres_result.state)
    u = gmres_result.solution

    # }}}

    # {{{ build matrix for spectrum check

    if 0:
        from sumpy.tools import build_matrix
        mat = build_matrix(bound_op.scipy_op("u", dtype=dtype, k=case.k))
        w, v = la.eig(mat)
        if 0:
            pt.imshow(np.log10(1e-20+np.abs(mat)))
            pt.colorbar()
            pt.show()

        #assert abs(s[-1]) < 1e-13, "h
        #assert abs(s[-2]) > 1e-7
        #from pudb import set_trace; set_trace()

    # }}}

    # {{{ error check

    bound_tgt_op = bind((qbx, PointsTarget(test_targets)),
            op.representation(sym.var("u")))

    test_via_bdry = bound_tgt_op(queue, u=u, k=case.k)

    err = test_direct-test_via_bdry

    err = err.get()
    test_direct = test_direct.get()
    test_via_bdry = test_via_bdry.get()

    # {{{ remove effect of net source charge

    if case.k == 0 and case.bc_type == "neumann" and case.loc_sign == -1:

        # remove constant offset in interior Laplace Neumann error
        tgt_ones = np.ones_like(test_direct)
        tgt_ones = tgt_ones/la.norm(tgt_ones)
        err = err - np.vdot(tgt_ones, err)*tgt_ones

    # }}}

    rel_err_2 = la.norm(err)/la.norm(test_direct)
    rel_err_inf = la.norm(err, np.inf)/la.norm(test_direct, np.inf)

    # }}}

    print("rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf))

    # {{{ test tangential derivative

    if case.check_tangential_deriv:
        bound_t_deriv_op = bind(qbx,
                op.representation(
                    sym.var("u"),
                    map_potentials=lambda pot: sym.tangential_derivative(2, pot),
                    qbx_forced_limit=case.loc_sign))

        #print(bound_t_deriv_op.code)

        tang_deriv_from_src = bound_t_deriv_op(
                queue, u=u, **concrete_knl_kwargs).as_scalar().get()

        tang_deriv_ref = (bind(
                (point_source, density_discr),
                sym.tangential_derivative(2, pot_src)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)
                .as_scalar().get())

        if 0:
            pt.plot(tang_deriv_ref.real)
            pt.plot(tang_deriv_from_src.real)
            pt.show()

        td_err = (tang_deriv_from_src - tang_deriv_ref)

        rel_td_err_inf = la.norm(td_err, np.inf)/la.norm(tang_deriv_ref, np.inf)

        print("rel_td_err_inf: %g" % rel_td_err_inf)

    else:
        rel_td_err_inf = None

    # }}}

    # {{{ 3D plotting

    if 0:
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(queue, density_discr, case.target_order)

        bdry_normals = bind(density_discr, sym.normal(3))(queue)\
                .as_vector(dtype=object)

        bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
            ("u", u),
            ("bc", bc),
            ("bdry_normals", bdry_normals),
            ])

        from meshmode.mesh.processing import find_bounding_box
        bbox_min, bbox_max = find_bounding_box(mesh)
        bbox_center = 0.5*(bbox_min+bbox_max)
        bbox_size = max(bbox_max-bbox_min) / 2
        fplot = FieldPlotter(
                bbox_center, extent=2*2*bbox_size, npoints=(150, 150, 1))

        qbx_stick_out = qbx.copy(target_stick_out_factor=0.15)
        from pytential.target import PointsTarget
        from pytential.qbx import QBXTargetAssociationFailedException

        try:
            solved_pot = bind(
                    (qbx_stick_out, PointsTarget(fplot.points)),
                    op.representation(sym.var("u"))
                    )(queue, u=u, k=case.k)
        except QBXTargetAssociationFailedException as e:
            fplot.write_vtk_file(
                    "failed-targets.vts",
                    [
                        ("failed_targets", e.failed_target_flags.get(queue))
                        ])
            raise

        solved_pot = solved_pot.get()

        true_pot = bind((point_source, PointsTarget(fplot.points)), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs).get()

        #fplot.show_scalar_in_mayavi(solved_pot.real, max_val=5)
        fplot.write_vtk_file(
                "potential.vts",
                [
                    ("solved_pot", solved_pot),
                    ("true_pot", true_pot),
                    ("pot_diff", solved_pot-true_pot),
                    ]
                )

    # }}}

    # {{{ 2D plotting

    if 0:
        fplot = FieldPlotter(np.zeros(2),
                extent=1.25*2*max(test_src_geo_radius, test_tgt_geo_radius),
                npoints=200)

        #pt.plot(u)
        #pt.show()

        fld_from_src = bind((point_source, PointsTarget(fplot.points)),
                pot_src)(queue, charges=source_charges_dev, **concrete_knl_kwargs)
        fld_from_bdry = bind(
                (qbx, PointsTarget(fplot.points)),
                op.representation(sym.var("u"))
                )(queue, u=u, k=case.k)
        fld_from_src = fld_from_src.get()
        fld_from_bdry = fld_from_bdry.get()

        nodes = density_discr.nodes().get(queue=queue)

        def prep():
            pt.plot(point_sources[0], point_sources[1], "o",
                    label="Monopole 'Point Charges'")
            pt.plot(test_targets[0], test_targets[1], "v",
                    label="Observation Points")
            pt.plot(nodes[0], nodes[1], "k-",
                    label=r"$\Gamma$")

        from matplotlib.cm import get_cmap
        cmap = get_cmap()
        cmap._init()
        if 0:
            cmap._lut[(cmap.N*99)//100:, -1] = 0  # make last percent transparent?

        prep()
        if 1:
            pt.subplot(131)
            pt.title("Field error (loc_sign=%s)" % case.loc_sign)
            log_err = np.log10(1e-20+np.abs(fld_from_src-fld_from_bdry))
            log_err = np.minimum(-3, log_err)
            fplot.show_scalar_in_matplotlib(log_err, cmap=cmap)

            #from matplotlib.colors import Normalize
            #im.set_norm(Normalize(vmin=-6, vmax=1))

            cb = pt.colorbar(shrink=0.9)
            cb.set_label(r"$\log_{10}(\mathdefault{Error})$")

        if 1:
            pt.subplot(132)
            prep()
            pt.title("Source Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_src.real, max_val=3)

            pt.colorbar(shrink=0.9)
        if 1:
            pt.subplot(133)
            prep()
            pt.title("Solved Field")
            fplot.show_scalar_in_matplotlib(
                    fld_from_bdry.real, max_val=3)

            pt.colorbar(shrink=0.9)

        # total field
        #fplot.show_scalar_in_matplotlib(
        #fld_from_src.real+fld_from_bdry.real, max_val=0.1)

        #pt.colorbar()

        pt.legend(loc="best", prop=dict(size=15))
        from matplotlib.ticker import NullFormatter
        pt.gca().xaxis.set_major_formatter(NullFormatter())
        pt.gca().yaxis.set_major_formatter(NullFormatter())

        pt.gca().set_aspect("equal")

        if 0:
            border_factor_top = 0.9
            border_factor = 0.3

            xl, xh = pt.xlim()
            xhsize = 0.5*(xh-xl)
            pt.xlim(xl-border_factor*xhsize, xh+border_factor*xhsize)

            yl, yh = pt.ylim()
            yhsize = 0.5*(yh-yl)
            pt.ylim(yl-border_factor_top*yhsize, yh+border_factor*yhsize)

        #pt.savefig("helmholtz.pdf", dpi=600)
        pt.show()

        # }}}

    class Result(Record):
        pass

    return Result(
            h_max=qbx.h_max,
            rel_err_2=rel_err_2,
            rel_err_inf=rel_err_inf,
            rel_td_err_inf=rel_td_err_inf,
            gmres_result=gmres_result)

# }}}


# {{{ integral equation test frontend

class IntEqTestCase:
    def __init__(self, helmholtz_k, bc_type, loc_sign):
        self.helmholtz_k = helmholtz_k
        self.bc_type = bc_type
        self.loc_sign = loc_sign

    @property
    def k(self):
        return self.helmholtz_k

    def __str__(self):
        return ("name: %s, bc_type: %s, loc_sign: %s, "
                "helmholtz_k: %s, qbx_order: %d, target_order: %d"
            % (self.name, self.bc_type, self.loc_sign, self.helmholtz_k,
                self.qbx_order, self.target_order))

    fmm_backend = "sumpy"
    gmres_tol = 1e-14


class CurveIntEqTestCase(IntEqTestCase):
    resolutions = [30, 40, 50]

    def get_mesh(self, resolution, target_order):
        return make_curve_mesh(
                self.curve_func,
                np.linspace(0, 1, resolution+1),
                target_order)

    fmm_backend = None
    use_refinement = True
    neumann_alpha = None  # default

    inner_radius = 0.1
    outer_radius = 2

    qbx_order = 5
    target_order = qbx_order

    check_tangential_deriv = True


class EllipseIntEqTestCase(CurveIntEqTestCase):
    name = "3-to-1 ellipse"

    def curve_func(self, x):
        return ellipse(3, x)


class EllipsoidIntEqTestCase(IntEqTestCase):
    resolutions = [2, 1]
    name = "ellipsoid"

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource("ellipsoid.step"), 2, order=2,
                other_options=[
                    "-string",
                    "Mesh.CharacteristicLengthMax = %g;" % resolution])

        from meshmode.mesh.processing import perform_flips
        # Flip elements--gmsh generates inside-out geometry.
        return perform_flips(mesh, np.ones(mesh.nelements))

    fmm_backend = "fmmlib"
    use_refinement = False
    neumann_alpha = 0  # no double layers in FMMlib backend yet

    inner_radius = 0.4
    outer_radius = 5

    qbx_order = 2
    target_order = qbx_order
    check_tangential_deriv = False

    # We're only expecting three digits based on FMM settings. Who are we
    # kidding?
    gmres_tol = 1e-5


@pytest.mark.parametrize("case", [
    EllipseIntEqTestCase(helmholtz_k=helmholtz_k, bc_type=bc_type,
        loc_sign=loc_sign)
    for helmholtz_k in [0, 1.2]
    for bc_type in ["dirichlet", "neumann"]
    for loc_sign in [-1, +1]
    ] + [
    EllipsoidIntEqTestCase(0.7, "neumann", +1)
    ])
# Sample test run:
# 'test_integral_equation(cl._csc, EllipseIntEqTestCase(0, "dirichlet", +1), 5)'  # noqa: E501
def test_integral_equation(ctx_getter, case):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    if case.fmm_backend == "fmmlib":
        pytest.importorskip("pyfmmlib")

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    from pytools.convergence import EOCRecorder
    print("qbx_order: %d, %s" % (case.qbx_order, case))

    eoc_rec_target = EOCRecorder()
    eoc_rec_td = EOCRecorder()

    for resolution in case.resolutions:
        result = run_int_eq_test(cl_ctx, queue, case, resolution)

        eoc_rec_target.add_data_point(result.h_max, result.rel_err_2)

        if result.rel_td_err_inf is not None:
            eoc_rec_td.add_data_point(result.h_max, result.rel_td_err_inf)

    if case.bc_type == "dirichlet":
        tgt_order = case.qbx_order
    elif case.bc_type == "neumann":
        tgt_order = case.qbx_order-1
    else:
        assert False

    print("TARGET ERROR:")
    print(eoc_rec_target)
    assert eoc_rec_target.order_estimate() > tgt_order - 1.3

    if case.check_tangential_deriv:
        print("TANGENTIAL DERIVATIVE ERROR:")
        print(eoc_rec_td)
        assert eoc_rec_td.order_estimate() > tgt_order - 2.3

# }}}


# {{{ integral identity tester

d1 = sym.Derivative()
d2 = sym.Derivative()


def get_starfish_mesh(refinement_increment, target_order):
    nelements = [30, 50, 70][refinement_increment]
    return make_curve_mesh(starfish,
                np.linspace(0, 1, nelements+1),
                target_order)


def get_wobbly_circle_mesh(refinement_increment, target_order):
    nelements = [3000, 5000, 7000][refinement_increment]
    return make_curve_mesh(WobblyCircle.random(30, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)


def get_sphere_mesh(refinement_increment, target_order):
    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, target_order)
    from meshmode.mesh.refinement import Refiner

    refiner = Refiner(mesh)
    for i in range(refinement_increment):
        flags = np.ones(mesh.nelements, dtype=bool)
        refiner.refine(flags)
        mesh = refiner.get_current_mesh()

    return mesh


@pytest.mark.parametrize(("mesh_name", "mesh_getter", "qbx_order"), [
    #("circle", partial(ellipse, 1)),
    #("3-to-1 ellipse", partial(ellipse, 3)),
    ("starfish", get_starfish_mesh, 5),
    ("sphere", get_sphere_mesh, 3),
    ])
@pytest.mark.parametrize(("zero_op_name", "k"), [
    ("green", 0),
    ("green", 1.2),
    ("green_grad", 0),
    ("green_grad", 1.2),
    ("zero_calderon", 0),
    ])
# sample invocation to copy and paste:
# 'test_identities(cl._csc, "green", "starfish", get_starfish_mesh, 4, 0)'
def test_identities(ctx_getter, zero_op_name, mesh_name, mesh_getter, qbx_order, k):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    if mesh_name == "sphere" and k != 0:
        pytest.skip("both direct eval and generating the FMM kernels are too slow")

    if mesh_name == "sphere" and zero_op_name == "green_grad":
        pytest.skip("does not achieve sufficient precision")

    target_order = 8

    order_table = {
            "green": qbx_order,
            "green_grad": qbx_order-1,
            "zero_calderon": qbx_order-1,
            }

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for refinement_increment in [0, 1, 2]:
        mesh = mesh_getter(refinement_increment, target_order)
        if mesh is None:
            break

        d = mesh.ambient_dim

        u_sym = sym.var("u")
        grad_u_sym = sym.make_sym_mv("grad_u",  d)
        dn_u_sym = sym.var("dn_u")

        from sumpy.kernel import LaplaceKernel, HelmholtzKernel
        lap_k_sym = LaplaceKernel(d)
        if k == 0:
            k_sym = lap_k_sym
            knl_kwargs = {}
        else:
            k_sym = HelmholtzKernel(d)
            knl_kwargs = {"k": sym.var("k")}

        zero_op_table = {
                "green":
                sym.S(k_sym, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs)
                - sym.D(k_sym, u_sym, qbx_forced_limit="avg", **knl_kwargs)
                - 0.5*u_sym,

                "green_grad":
                d1.resolve(d1.dnabla(d) * d1(sym.S(k_sym, dn_u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - d2.resolve(d2.dnabla(d) * d2(sym.D(k_sym, u_sym,
                    qbx_forced_limit="avg", **knl_kwargs)))
                - 0.5*grad_u_sym,

                # only for k==0:
                "zero_calderon":
                -sym.Dp(lap_k_sym, sym.S(lap_k_sym, u_sym))
                - 0.25*u_sym + sym.Sp(lap_k_sym, sym.Sp(lap_k_sym, u_sym))
                }

        zero_op = zero_op_table[zero_op_name]

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        from pytential.qbx import QBXLayerPotentialSource
        pre_density_discr = Discretization(
                cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(target_order))

        if d == 2:
            order_bump = 15
        elif d == 3:
            order_bump = 8

        refiner_extra_kwargs = {}

        if k != 0:
            refiner_extra_kwargs["kernel_length_scale"] = 5/k

        qbx, _ = QBXLayerPotentialSource(
                pre_density_discr, 4*target_order,
                qbx_order, fmm_order=qbx_order + order_bump
                ).with_refinement(**refiner_extra_kwargs)

        density_discr = qbx.density_discr

        # {{{ compute values of a solution to the PDE

        nodes_host = density_discr.nodes().get(queue)
        normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
        normal_host = [normal[j].get() for j in range(d)]

        if k != 0:
            if d == 2:
                angle = 0.3
                wave_vec = np.array([np.cos(angle), np.sin(angle)])
                u = np.exp(1j*k*np.tensordot(wave_vec, nodes_host, axes=1))
                grad_u = 1j*k*wave_vec[:, np.newaxis]*u
            else:
                center = np.array([3, 1, 2])
                diff = nodes_host - center[:, np.newaxis]
                r = la.norm(diff, axis=0)
                u = np.exp(1j*k*r) / r
                grad_u = diff * (1j*k*u/r - u/r**2)
        else:
            center = np.array([3, 1, 2])[:d]
            diff = nodes_host - center[:, np.newaxis]
            dist_squared = np.sum(diff**2, axis=0)
            dist = np.sqrt(dist_squared)
            if d == 2:
                u = np.log(dist)
                grad_u = diff/dist_squared
            elif d == 3:
                u = 1/dist
                grad_u = -diff/dist**3
            else:
                assert False

        dn_u = 0
        for i in range(d):
            dn_u = dn_u + normal_host[i]*grad_u[i]

        # }}}

        u_dev = cl.array.to_device(queue, u)
        dn_u_dev = cl.array.to_device(queue, dn_u)
        grad_u_dev = cl.array.to_device(queue, grad_u)

        key = (qbx_order, mesh_name, refinement_increment, zero_op_name)

        bound_op = bind(qbx, zero_op)
        error = bound_op(
                queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)
        if 0:
            pt.plot(error)
            pt.show()

        l2_error_norm = norm(density_discr, queue, error)
        print(key, l2_error_norm)

        eoc_rec.add_data_point(qbx.h_max, l2_error_norm)

    print(eoc_rec)
    tgt_order = order_table[zero_op_name]
    assert eoc_rec.order_estimate() > tgt_order - 1.3

# }}}


# {{{ test off-surface eval

@pytest.mark.parametrize("use_fmm", [True, False])
def test_off_surface_eval(ctx_getter, use_fmm, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 30
    target_order = 8
    qbx_order = 3
    if use_fmm is True:
        fmm_order = qbx_order
    else:
        fmm_order = False

    mesh = make_curve_mesh(partial(ellipse, 3),
            np.linspace(0, 1, nelements+1),
            target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr,
            4*target_order,
            qbx_order,
            fmm_order=fmm_order,
            ).with_refinement()

    density_discr = qbx.density_discr

    from sumpy.kernel import LaplaceKernel
    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(queue) + 1

    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
    from pytential.target import PointsTarget
    fld_in_vol = bind(
            (qbx, PointsTarget(fplot.points)),
            op)(queue, sigma=sigma)

    err = cl.clmath.fabs(fld_in_vol - (-1))

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    if do_plot:
        fplot.show_scalar_in_matplotlib(fld_in_vol.get())
        import matplotlib.pyplot as pt
        pt.colorbar()
        pt.show()

    # FIXME: Why does the FMM only meet this sloppy tolerance?
    assert linf_err < 1e-2

# }}}


# {{{ test off-surface eval vs direct

def test_off_surface_eval_vs_direct(ctx_getter,  do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 300
    target_order = 8
    qbx_order = 3

    mesh = make_curve_mesh(WobblyCircle.random(8, seed=30),
                np.linspace(0, 1, nelements+1),
                target_order)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
    direct_qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=False,
            target_stick_out_factor=0.05,
            ).with_refinement()
    fmm_qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, 4*target_order, qbx_order,
            fmm_order=qbx_order + 3,
            expansion_disks_in_tree_have_extent=True,
            target_stick_out_factor=0.05,
            ).with_refinement()

    fplot = FieldPlotter(np.zeros(2), extent=5, npoints=1000)
    from pytential.target import PointsTarget
    ptarget = PointsTarget(fplot.points)
    from sumpy.kernel import LaplaceKernel

    op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=None)

    from pytential.qbx import QBXTargetAssociationFailedException
    try:
        direct_density_discr = direct_qbx.density_discr
        direct_sigma = direct_density_discr.zeros(queue) + 1
        direct_fld_in_vol = bind((direct_qbx, ptarget), op)(
                queue, sigma=direct_sigma)

    except QBXTargetAssociationFailedException as e:
        fplot.show_scalar_in_matplotlib(e.failed_target_flags.get(queue))
        import matplotlib.pyplot as pt
        pt.show()
        raise

    fmm_density_discr = fmm_qbx.density_discr
    fmm_sigma = fmm_density_discr.zeros(queue) + 1
    fmm_fld_in_vol = bind((fmm_qbx, ptarget), op)(queue, sigma=fmm_sigma)

    err = cl.clmath.fabs(fmm_fld_in_vol - direct_fld_in_vol)

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    if do_plot:
        #fplot.show_scalar_in_mayavi(0.1*cl.clmath.log10(1e-15 + err).get(queue))
        fplot.show_scalar_in_mayavi(fmm_fld_in_vol.get(queue))
        import mayavi.mlab as mlab
        mlab.show()

    assert linf_err < 1e-3

# }}}


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
