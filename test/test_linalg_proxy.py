__copyright__ = "Copyright (C) 2018 Alexandru Fikl"

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

from functools import partial

import numpy as np
import numpy.linalg as la

import pyopencl as cl

from pytential import bind, sym
from pytential import GeometryCollection
from pytential.utils import flatten_to_numpy

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh.generation import ellipse, NArmedStarfish

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import extra_matrix_data as extra
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# {{{ plot_partition_indices

def plot_partition_indices(actx, discr, indices, **kwargs):
    try:
        import matplotlib.pyplot as pt
    except ImportError:
        return

    args = [
        str(kwargs.get("tree_kind", "linear")).replace("-", "_"),
        kwargs.get("discr_stage", "stage1"),
        discr.ambient_dim
        ]

    pt.figure(figsize=(10, 8), dpi=300)
    pt.plot(np.diff(indices.ranges))
    pt.savefig("test_partition_{1}_{3}d_ranges_{2}.png".format(*args))
    pt.clf()

    if discr.ambient_dim == 2:
        sources = flatten_to_numpy(actx, discr.nodes())

        pt.figure(figsize=(10, 8), dpi=300)
        if indices.indices.shape[0] != discr.ndofs:
            pt.plot(sources[0], sources[1], "ko", alpha=0.5)

        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
            pt.plot(sources[0][isrc], sources[1][isrc], "o")

        pt.xlim([-1.5, 1.5])
        pt.ylim([-1.5, 1.5])
        pt.savefig("test_partition_{1}_{3}d_{2}.png".format(*args))
        pt.clf()
    elif discr.ambient_dim == 3:
        from meshmode.discretization.visualization import make_visualizer
        marker = -42.0 * np.ones(discr.ndofs)

        for i in range(indices.nblocks):
            isrc = indices.block_indices(i)
            marker[isrc] = 10.0 * (i + 1.0)

        from meshmode.dof_array import unflatten
        marker = unflatten(actx, discr, actx.from_numpy(marker))

        vis = make_visualizer(actx, discr, 10)

        filename = "test_partition_{0}_{1}_{3}d_{2}.vtu".format(*args)
        vis.write_vtk_file(filename, [
            ("marker", marker)
            ])

# }}}


PROXY_TEST_CASES = [
        extra.CurveTestCase(
            name="ellipse",
            target_order=7,
            curve_fn=partial(ellipse, 3.0)),
        extra.CurveTestCase(
            name="starfish",
            target_order=4,
            curve_fn=NArmedStarfish(5, 0.25),
            resolutions=[32]),
        extra.TorusTestCase(
            target_order=2,
            resolutions=[0])
        ]


# {{{ test_partition_points

@pytest.mark.parametrize("tree_kind", ["adaptive", None])
@pytest.mark.parametrize("case", PROXY_TEST_CASES)
def test_partition_points(ctx_factory, tree_kind, case, visualize=False):
    """Tests that the points are correctly partitioned (by visualization)."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(tree_kind=tree_kind, index_sparsity_factor=1.0)
    logger.info("\n%s", case)

    # {{{

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    indices = case.get_block_indices(actx, density_discr)
    indices = indices.get(actx.queue).row

    expected_indices = np.arange(0, density_discr.ndofs)
    assert indices.ranges[-1] == density_discr.ndofs

    assert la.norm(np.sort(indices.indices) - expected_indices) < 1.0e-14

    if visualize:
        plot_partition_indices(actx, density_discr, indices, tree_kind=tree_kind)

    # }}}

# }}}


# {{{ test_proxy_generator

@pytest.mark.parametrize("case", PROXY_TEST_CASES)
@pytest.mark.parametrize("index_sparsity_factor", [1.0, 0.6])
@pytest.mark.parametrize("proxy_radius_factor", [0.0, 1.1])
def test_proxy_generator(ctx_factory, case,
        index_sparsity_factor, proxy_radius_factor, visualize=False):
    """Tests that the proxies generated are all at the correct radius from the
    points in the cluster, etc.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(
            index_sparsity_factor=index_sparsity_factor,
            proxy_radius_factor=proxy_radius_factor)
    logger.info("\n%s", case)

    # {{{ check proxies

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    srcindices = case.get_block_indices(actx, density_discr, matrix_indices=False)

    from pytential.linalg.proxy import ProxyGenerator
    proxies = ProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=case.proxy_approx_count)(
                    actx, places.auto_source, srcindices)

    proxies = proxies.get(queue)
    pxypoints = np.vstack(proxies.points)
    pxycenters = np.vstack(proxies.centers)

    srcindices = srcindices.get(queue)
    nodes = np.vstack(flatten_to_numpy(actx, density_discr.nodes()))

    for i in range(srcindices.nblocks):
        isrc = srcindices.block_indices(i)
        ipxy = proxies.indices.block_indices(i)

        r = la.norm(pxypoints[:, ipxy] - pxycenters[:, i].reshape(-1, 1), axis=0)
        p_error = la.norm(r - proxies.radii[i])

        r = la.norm(nodes[:, isrc] - pxycenters[:, i].reshape(-1, 1), axis=0)
        n_error = la.norm(r - proxies.radii[i], np.inf)

        assert p_error < 1.0e-14, f"block {i}"
        assert n_error - proxies.radii[i] < 1.0e-14, f"block {i}"

    # }}}

    # {{{ visualization

    if not visualize:
        return

    ambient_dim = places.ambient_dim
    if ambient_dim == 2:
        try:
            import matplotlib.pyplot as pt
        except ImportError:
            return

        ci = np.vstack(flatten_to_numpy(actx,
            bind(places, sym.expansion_centers(ambient_dim, -1))(actx)
            ))
        ce = np.vstack(flatten_to_numpy(actx,
            bind(places, sym.expansion_centers(ambient_dim, +1))(actx)
            ))
        r = flatten_to_numpy(actx,
                bind(places, sym.expansion_radii(ambient_dim))(actx)
                )

        fig = pt.figure(figsize=(10, 8))
        for i in range(srcindices.nblocks):
            isrc = srcindices.block_indices(i)
            ipxy = proxies.indices.block_indices(i)

            ax = pt.gca()
            for j in isrc:
                c = pt.Circle(ci[:, j], r[j], color="k", alpha=0.1)
                ax.add_artist(c)
                c = pt.Circle(ce[:, j], r[j], color="k", alpha=0.1)
                ax.add_artist(c)

            ax.plot(nodes[0], nodes[1], "ko", ms=2.0, alpha=0.5)
            ax.plot(nodes[0, srcindices.indices], nodes[1, srcindices.indices],
                    "o", ms=2.0)
            ax.plot(nodes[0, isrc], nodes[1, isrc], "o", ms=1.0)
            ax.plot(pxypoints[0, ipxy], pxypoints[1, ipxy], "o", ms=1.0)
            ax.plot(pxycenters[0, i], pxycenters[1, i], "ko", ms=2.0)
            ax.set_aspect("equal")
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])

            filename = f"test_proxy_generator_{ambient_dim}d_{i:04}"
            fig.savefig(filename, dpi=300)
            fig.clf()
        pt.close(fig)
    else:
        from meshmode.discretization.visualization import make_visualizer
        from meshmode.mesh.processing import ( # noqa
                affine_map, merge_disjoint_meshes)
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

        from meshmode.mesh.generation import generate_icosphere
        ref_mesh = generate_icosphere(1, proxies.nproxy)

        # NOTE: this does not plot the actual proxy points
        for i in range(srcindices.nblocks):
            mesh = affine_map(ref_mesh,
                A=(proxies.radii[i] * np.eye(ambient_dim)),
                b=pxycenters[:, i].reshape(-1))

            mesh = merge_disjoint_meshes([mesh, density_discr.mesh])
            discr = Discretization(actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(10))

            vis = make_visualizer(actx, discr, 10)

            filename = f"test_proxy_generator_{ambient_dim}d_{i:04}.vtu"
            vis.write_vtk_file(filename, [])

    # }}}

# }}}


# {{{ test_neighbor_points

@pytest.mark.parametrize("case", PROXY_TEST_CASES)
@pytest.mark.parametrize("index_sparsity_factor", [1.0, 0.6])
@pytest.mark.parametrize("proxy_radius_factor", [0.0, 1.1])
def test_neighbor_points(ctx_factory, case,
        index_sparsity_factor, proxy_radius_factor, visualize=False):
    """Test that neighboring points (inside the proxy balls, but outside the
    current block/cluster) are actually inside.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(
            index_sparsity_factor=index_sparsity_factor,
            proxy_radius_factor=proxy_radius_factor)
    logger.info("\n%s", case)

    # {{{ check neighboring points

    qbx = case.get_layer_potential(actx, case.resolutions[-1], case.target_order)
    places = GeometryCollection(qbx, auto_where=case.name)

    density_discr = places.get_discretization(case.name)
    srcindices = case.get_block_indices(actx, density_discr, matrix_indices=False)

    # generate proxy points
    from pytential.linalg.proxy import ProxyGenerator
    proxies = ProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=case.proxy_approx_count)(
                    actx, places.auto_source, srcindices)

    # get neighboring points
    from pytential.linalg.proxy import gather_block_neighbor_points
    nbrindices = gather_block_neighbor_points(actx,
            density_discr, srcindices, proxies)

    srcindices = srcindices.get(queue)
    nbrindices = nbrindices.get(queue)

    proxies = proxies.get(queue)
    pxycenters = np.vstack(proxies.centers)
    nodes = np.vstack(flatten_to_numpy(actx, density_discr.nodes()))

    for i in range(srcindices.nblocks):
        isrc = srcindices.block_indices(i)
        inbr = nbrindices.block_indices(i)

        assert not np.any(np.isin(inbr, isrc))

        r = la.norm(nodes[:, inbr] - pxycenters[:, i].reshape(-1, 1), axis=0)
        assert np.all(r - proxies.radii[i] < 0.0)

    # }}}

    # {{{ visualize

    if not visualize:
        return

    ambient_dim = places.ambient_dim
    if ambient_dim == 2:
        try:
            import matplotlib.pyplot as pt
        except ImportError:
            return

        pxycenters = np.vstack(proxies.centers)
        pxyradii = proxies.radii

        fig = pt.figure(figsize=(10, 10))
        for i in range(srcindices.nblocks):
            isrc = srcindices.block_indices(i)
            inbr = nbrindices.block_indices(i)

            ax = fig.gca()

            ax.plot(nodes[0], nodes[1], "ko", ms=2.0, alpha=0.5)
            ax.plot(nodes[0, srcindices.indices], nodes[1, srcindices.indices],
                    "o", ms=2.0)
            ax.plot(nodes[0, isrc], nodes[1, isrc], "o", ms=1.0)
            ax.plot(nodes[0, inbr], nodes[1, inbr], "o", ms=1.0)
            ax.plot(pxycenters[0, i], pxycenters[1, i], "ko", ms=2.0)
            ax.axis("equal")

            c = pt.Circle(pxycenters[:, i], pxyradii[i], color="k", alpha=0.1)
            ax.add_artist(c)

            pt.xlim([-1.5, 1.5])
            pt.ylim([-1.5, 1.5])

            filename = f"test_area_query_{ambient_dim}d_{i:04}"
            fig.savefig(filename, dpi=300)
            fig.clf()
        pt.close(fig)
    elif ambient_dim == 3:
        from meshmode.discretization.visualization import make_visualizer
        marker = np.empty(density_discr.ndofs)

        for i in range(srcindices.nblocks):
            isrc = srcindices.block_indices(i)
            inbr = nbrindices.block_indices(i)

            marker.fill(0.0)
            marker[srcindices.indices] = 0.0
            marker[isrc] = -42.0
            marker[inbr] = +42.0

            from meshmode.dof_array import unflatten
            marker_dev = unflatten(actx, density_discr, actx.from_numpy(marker))

            vis = make_visualizer(actx, density_discr, 10)
            filename = "test_area_query_{}d_{:04}.vtu".format(ambient_dim, i)
            vis.write_vtk_file(filename, [
                ("marker", marker_dev),
                ])

    # }}}

# }}}


# {{{ test_skeletonize_by_proxy

def _plot_skeleton_with_proxies(name, sources, pxy, isrc, iskl):
    import matplotlib.pyplot as pt

    fig, ax = pt.subplots(1, figsize=(10, 10), dpi=300)
    ax.plot(sources[0][isrc.indices], sources[1][isrc.indices],
            "ko", alpha=0.5)

    pxycenters = np.vstack(pxy.centers)
    pxyradii = pxy.radii
    for i in range(isrc.nblocks):
        iblk = iskl.block_indices(i)
        pt.plot(sources[0][iblk], sources[1][iblk], "o")

        c = pt.Circle(pxycenters[:, i], pxyradii[i], color="k", alpha=0.1)
        ax.add_artist(c)
        ax.text(*pxycenters[:, i], f"{i}",
                fontweight="bold", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    fig.savefig(f"test_skeletonize_by_proxy_{name}")
    pt.close(fig)


@pytest.mark.parametrize("case", PROXY_TEST_CASES)
def test_skeletonize_by_proxy(ctx_factory, case, visualize=False):
    """Test single-level level skeletonization accuracy."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = case.copy(approx_block_count=6, op_type="scalar", id_eps=1.0e-8)
    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[0], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    srcindices = case.get_block_indices(actx, density_discr)

    logger.info("nblocks %3d ndofs %7d", srcindices.nblocks, density_discr.ndofs)

    # }}}

    # {{{ wranglers

    from pytential.linalg.proxy import ProxyGenerator
    from pytential.linalg.skeletonization import make_block_evaluation_wrangler
    proxy_generator = ProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=case.proxy_approx_count)

    sym_u, sym_op = case.get_operator(places.ambient_dim)
    wrangler = make_block_evaluation_wrangler(places, sym_op, sym_u,
            domains=None,
            context=case.knl_concrete_kwargs,
            _weighted_farfield=case.weighted_farfield,
            _farfield_block_builder=case.farfield_block_builder,
            _nearfield_block_builder=case.nearfield_block_builder)

    # }}}

    # {{{ check proxy id decomposition

    # dense matrix
    from pytential.symbolic.execution import build_matrix
    mat = actx.to_numpy(
            build_matrix(actx, places, sym_op, sym_u,
                context=case.knl_concrete_kwargs)
            )

    # skeleton
    from pytential.linalg.skeletonization import \
            _skeletonize_block_by_proxy_with_mats
    L, R, sklindices, src, tgt = _skeletonize_block_by_proxy_with_mats(actx,
            0, 0, places, proxy_generator, wrangler, srcindices,
            id_eps=case.id_eps,
            max_particles_in_box=case.max_particles_in_box)

    srcindices = srcindices.get(queue)
    sklindices = sklindices.get(queue)
    for i in range(sklindices.nblocks):
        # targets (rows)
        bi = np.searchsorted(
            srcindices.row.block_indices(i),
            sklindices.row.block_indices(i),
            )

        A = tgt[i, i]
        S = A[bi, :]
        tgt_error = la.norm(A - L[i, i] @ S) / la.norm(A)

        # sources (columns)
        bj = np.searchsorted(
            srcindices.col.block_indices(i),
            sklindices.col.block_indices(i),
            )

        A = src[i, i]
        S = A[:, bj]
        src_error = la.norm(A - S @ R[i, i]) / la.norm(A)

        logger.info("[%04d] id_eps %.5e src %.5e tgt %.5e rank %d/%d",
                i, case.id_eps,
                src_error, tgt_error, R[i, i].shape[0], R[i, i].shape[1])

        assert src_error < 6 * case.id_eps
        assert tgt_error < 6 * case.id_eps

    # }}}

    # {{{ check skeletonize

    from pytential.linalg.skeletonization import SkeletonizedBlock
    skeleton = SkeletonizedBlock(L=L, R=R, sklindices=sklindices)

    blk_err_l, blk_err_r, err_f = extra.skeletonization_error(
            mat, skeleton, srcindices)
    err_l = la.norm(blk_err_l, np.inf)
    err_r = la.norm(blk_err_r, np.inf)

    # FIXME: why is the 3D error so large?
    rtol = 2 * 10**places.ambient_dim * case.id_eps

    logger.info("error: id_eps %.5e L %.5e R %.5e F %.5e (rtol %.5e)",
            case.id_eps, err_l, err_r, err_f, rtol)

    assert err_l < rtol
    assert err_r < rtol
    assert err_f < rtol

    # }}}

    # {{{ visualize

    if not visualize:
        return

    import matplotlib.pyplot as pt
    pt.imshow(np.log10(blk_err_l + 1.0e-16))
    pt.colorbar()
    pt.savefig("test_skeletonize_by_proxy_err_l")
    pt.clf()

    pt.imshow(np.log10(blk_err_r + 1.0e-16))
    pt.colorbar()
    pt.savefig("test_skeletonize_by_proxy_err_r")
    pt.clf()

    if places.ambient_dim == 2:
        pxy = proxy_generator(actx, wrangler.domains[0], srcindices.row).get(queue)
        sources = flatten_to_numpy(actx, density_discr.nodes())
        srcindices = srcindices.get(queue)
        sklindices = sklindices.get(queue)

        _plot_skeleton_with_proxies("sources", sources, pxy,
                srcindices.col, sklindices.col)
        _plot_skeleton_with_proxies("targets", sources, pxy,
                srcindices.row, sklindices.row)
    else:
        # TODO: would be nice to figure out a way to visualize some of these
        # skeletonization results in 3D. Probably need to teach the
        # visualizers to spit out point clouds
        pass

    # }}}

# }}}


# {{{ test_skeletonize_by_proxy_symmetry

def test_skeletonize_by_proxy_symmetry(ctx_factory, visualize=False):
    """Tests skeletonization in a symmetric setting."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    case = extra.CurveTestCase(
            name="ellipse",
            op_type="double",
            max_particles_in_box=96,
            proxy_radius_factor=1.2,
            proxy_approx_count=16,
            tree_kind=None,
            curve_fn=partial(ellipse, 2.0),
            target_order=16,
            resolutions=[96],
            weighted_farfield=None)

    logger.info("\n%s", case)

    # {{{ geometry

    dd = sym.DOFDescriptor(case.name, discr_stage=case.skel_discr_stage)
    qbx = case.get_layer_potential(actx, case.resolutions[0], case.target_order)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    srcindices = case.get_block_indices(actx, density_discr)

    logger.info("nblocks %3d ndofs %7d", srcindices.nblocks, density_discr.ndofs)

    # }}}

    # {{{ wranglers

    from pytential.linalg.proxy import ProxyGenerator
    from pytential.linalg.skeletonization import make_block_evaluation_wrangler
    proxy_generator = ProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=case.proxy_approx_count)

    sym_u, sym_op = case.get_operator(places.ambient_dim, qbx_forced_limit="avg")
    wrangler = make_block_evaluation_wrangler(places, sym_op, sym_u,
            domains=None,
            context=case.knl_concrete_kwargs,
            _weighted_farfield=case.weighted_farfield,
            _farfield_block_builder=case.farfield_block_builder,
            _nearfield_block_builder=case.nearfield_block_builder)

    # }}}

    # {{{ skeletonize

    from pytential.linalg.skeletonization import \
            _skeletonize_block_by_proxy_with_mats
    L, R, sklindices, src, tgt = _skeletonize_block_by_proxy_with_mats(actx,
            0, 0, places, proxy_generator, wrangler, srcindices,
            id_eps=case.id_eps,
            max_particles_in_box=case.max_particles_in_box)

    srcindices = srcindices.get(queue)
    sklindices = sklindices.get(queue)

    pxy_dev = proxy_generator(actx, case.name, srcindices.col)

    pxy = pxy_dev.get(queue)
    pxypoints = np.vstack(pxy.points)
    pxycenters = np.vstack(pxy.centers)

    # }}}

    # {{{ checks

    sources = np.vstack(flatten_to_numpy(actx, density_discr.nodes()))
    if visualize:
        _plot_skeleton_with_proxies("symmetry", sources, pxy,
                srcindices.col, sklindices.col)

    # NOTE: picking different blocks will need some changes below too because
    # they will have different symmetries: (3, 12) flips up-down, but
    # (0, 7) flips left-right

    # pick symmetric blocks
    # i, j = 0, 7
    # i, j = 8, 15
    i, j = 3, 12

    # {{{ check nodes are actually symmetric (sanity check)

    isrc = srcindices.col.block_indices(i)
    jsrc = srcindices.col.block_indices(j)[::-1]
    assert isrc.shape == jsrc.shape

    error_x = la.norm(sources[0, isrc] - sources[0, jsrc])/la.norm(sources[0, isrc])
    error_y = la.norm(sources[1, isrc] + sources[1, jsrc])/la.norm(sources[1, isrc])

    logger.info("error: x %.6e y %.5e", error_x, error_y)
    assert error_x < 5.0e-15 and error_y < 5.0e-15

    # }}}

    # {{{ check proxies are also symmetric

    isrc = pxy.indices.block_indices(i)
    jsrc = pxy.indices.block_indices(j)

    ipxypoints = (pxypoints[:, isrc] - pxycenters[:, i:i+1]) / pxy.radii[i]
    jpxypoints = (pxypoints[:, jsrc] - pxycenters[:, j:j+1]) / pxy.radii[j]

    error_rad = abs(pxy.radii[i] - pxy.radii[j]) / abs(pxy.radii[i])
    error_pxy = la.norm(ipxypoints - jpxypoints)

    logger.info("error: radii %.6e pxy %.6e", error_rad, error_pxy)
    assert error_rad < 2.0e-12
    assert error_pxy < 5.0e-15

    # }}}

    # {{{ check proxy matrix symmetry

    ipxysrc = src[i, i][:pxy.nproxy, :]
    jpxysrc = src[j, j][:pxy.nproxy, :]

    itop = np.s_[:jpxysrc.shape[0] // 2]
    ibot = np.s_[jpxysrc.shape[0] // 2:]
    jpxysrc = np.vstack([
        jpxysrc[itop, :],
        jpxysrc[ibot, :],
        ])[::-1, ::-1]

    if visualize:
        import matplotlib.pyplot as pt
        fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(12, 4))
        p = ax1.imshow(ipxysrc)
        fig.colorbar(p, ax=ax1, shrink=0.75, orientation="horizontal")
        p = ax2.imshow(np.log10(np.abs(jpxysrc - ipxysrc) + 1.0e-16))
        # p = ax2.imshow(jpxysrc)
        fig.colorbar(p, ax=ax2, shrink=0.75, orientation="horizontal")
        fig.savefig(f"test_skeletonize_proxy_blocks_{i:02d}_and_{j:02d}")
        pt.close(fig)

    error_pxy_mat = la.norm(ipxysrc - jpxysrc)
    logger.info("error: pxysrc %.6e", error_pxy_mat)
    assert error_pxy_mat < 1.0e-14

    # }}}

    # {{{ check neighbor matrix symmetry

    inbrsrc = src[i, i][pxy.nproxy:, :]
    jnbrsrc = src[j, j][pxy.nproxy:, :]

    itop = np.s_[:jnbrsrc.shape[0] // 2]
    ibot = np.s_[jnbrsrc.shape[0] // 2:]
    jnbrsrc = np.vstack([
        jnbrsrc[itop, :][::-1, ::-1],
        jnbrsrc[ibot, :][::-1, ::-1]
        ])

    if visualize:
        fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(12, 4))
        p = ax1.imshow(inbrsrc)
        fig.colorbar(p, ax=ax1, shrink=0.75, orientation="horizontal")
        p = ax2.imshow(np.log10(np.abs(jnbrsrc - inbrsrc) + 1.0e-16))
        # p = ax2.imshow(jnbrsrc)
        fig.colorbar(p, ax=ax2, shrink=0.75, orientation="horizontal")
        fig.savefig(f"test_skeletonize_neighbor_blocks_{i:02d}_and_{j:02d}")
        pt.close(fig)

    error_nbr_mat = la.norm(inbrsrc - jnbrsrc)
    logger.info("error: nbrsrc %.6e", error_nbr_mat)
    assert error_nbr_mat < 5.0e-13

    # }}}

    # }}}

    # {{{ plot proxy blocks

    if visualize:
        from matplotlib.patches import Rectangle
        nrows = pxypoints.shape[1]
        ncols = density_discr.ndofs

        fig = pt.figure(figsize=(12, 5))
        ax = fig.gca()
        ax.imshow(np.eye(nrows, ncols), cmap="gray")
        for i in range(srcindices.nblocks):
            x0, x1 = srcindices.col.ranges[i:i+2]
            y0, y1 = pxy.indices.ranges[i:i+2]

            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=0.5)
            ax.add_artist(rect)

        fig.savefig("test_skeletonize_proxy_identity")
        pt.close(fig)

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
