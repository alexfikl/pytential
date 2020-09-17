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


import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array
from pytools import memoize_method

from sumpy.tools import BlockIndexRanges
from boxtree.tools import DeviceDataRecord

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """
Proxy Point Generation
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ProxyGenerator
.. autoclass:: BlockProxyPoints

.. autofunction:: partition_by_nodes
.. autofunction:: gather_block_neighbor_points
"""


# {{{ point index partitioning

def make_block_index(actx, indices, ranges=None):
    """Wrap a ``(indices, ranges)`` tuple into a ``BlockIndexRanges``."""
    if ranges is None:
        ranges = np.cumsum([0] + [r.size for r in indices])
        indices = np.hstack(indices)

    return BlockIndexRanges(actx.context,
            actx.freeze(actx.from_numpy(indices)),
            actx.freeze(actx.from_numpy(ranges)))


def partition_by_nodes(actx, discr,
        tree_kind="adaptive-level-restricted", max_particles_in_box=None):
    """Generate equally sized ranges of nodes. The partition is created at the
    lowest level of granularity, i.e. nodes. This results in balanced ranges
    of points, but will split elements across different ranges.

    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg tree_kind: if not *None*, it is passed to :class:`boxtree.TreeBuilder`.
    :arg max_particles_in_box: passed to :class:`boxtree.TreeBuilder`.

    :return: a :class:`sumpy.tools.BlockIndexRanges`.
    """

    if max_particles_in_box is None:
        # FIXME: this is just an arbitrary value
        max_particles_in_box = 32

    if tree_kind is not None:
        from boxtree import box_flags_enum
        from boxtree import TreeBuilder

        builder = TreeBuilder(actx.context)

        from meshmode.dof_array import flatten, thaw
        tree, _ = builder(actx.queue,
                flatten(thaw(actx, discr.nodes())),
                max_particles_in_box=max_particles_in_box,
                kind=tree_kind)

        tree = tree.get(actx.queue)
        leaf_boxes, = (tree.box_flags
                       & box_flags_enum.HAS_CHILDREN == 0).nonzero()

        indices = np.empty(len(leaf_boxes), dtype=np.object)
        ranges = None

        for i, ibox in enumerate(leaf_boxes):
            box_start = tree.box_source_starts[ibox]
            box_end = box_start + tree.box_source_counts_cumul[ibox]
            indices[i] = tree.user_source_ids[box_start:box_end]
    else:
        indices = np.arange(0, discr.ndofs, dtype=np.int)
        ranges = np.arange(0, discr.ndofs + 1, max_particles_in_box, dtype=np.int)

    return make_block_index(actx, indices, ranges=ranges)

# }}}


# {{{ proxy point generator

class BlockProxyPoints(DeviceDataRecord):
    """
    .. attribute:: indices

        A :class:`~sumpy.tools.BlockIndexRanges` describing which proxies
        belong to which block.

    .. attribute:: points

        A concatenated list of all the proxy points. Can be sliced into
        using :attr:`indices` (shape ``(dim, nproxies * nblocks)``).

    .. attribute:: centers

        A list of all the proxy ball centers (shape ``(dim, nblocks)``).

    .. attribute:: radii

        A list of all the proxy ball radii (shape ``(nblocks,)``).
    """

    @property
    def nproxy(self):
        return self.points[0].shape[0] // self.indices.nblocks


def _generate_unit_sphere(ambient_dim, approx_npoints):
    """Generate uniform points on a unit sphere.

    :arg ambient_dim: dimension of the ambient space.
    :arg approx_npoints: approximate number of points to generate. If the
        ambient space is 3D, this will not generate the exact number of points.
    :return: array of shape ``(ambient_dim, npoints)``, where ``npoints``
        will not generally be the same as ``approx_npoints``.
    """

    if ambient_dim == 2:
        t = np.linspace(0.0, 2.0 * np.pi, approx_npoints)
        points = np.vstack([np.cos(t), np.sin(t)])
    elif ambient_dim == 3:
        # https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        # code by Matt Wala from
        # https://github.com/mattwala/gigaqbx-accuracy-experiments/blob/d56ed063ffd7843186f4fe05d2a5b5bfe6ef420c/translation_accuracy.py#L23
        a = 4.0 * np.pi / approx_npoints
        m_theta = int(np.round(np.pi / np.sqrt(a)))
        d_theta = np.pi / m_theta
        d_phi = a / d_theta

        points = []
        for m in range(m_theta):
            theta = np.pi * (m + 0.5) / m_theta
            m_phi = int(np.round(2.0 * np.pi * np.sin(theta) / d_phi))

            for n in range(m_phi):
                phi = 2.0 * np.pi * n / m_phi
                points.append(np.array([np.sin(theta) * np.cos(phi),
                                        np.sin(theta) * np.sin(phi),
                                        np.cos(theta)]))

        for i in range(ambient_dim):
            for sign in [-1, 1]:
                pole = np.zeros(ambient_dim)
                pole[i] = sign
                points.append(pole)

        points = np.array(points).T
    else:
        raise ValueError("ambient_dim > 3 not supported.")

    return points


class ProxyGenerator:
    r"""
    .. attribute:: places

        A :class:`~pytential.GeometryCollection`
        containing the geometry on which the proxy balls are generated.

    .. attribute:: nproxy

        Number of proxy points in a single proxy ball.

    .. attribute:: radius_factor

        A factor used to compute the proxy ball radius. The radius
        is computed in the :math:`\ell^2` norm, resulting in a circle or
        sphere of proxy points. For QBX, we have two radii of interest
        for a set of points: the radius :math:`r_{block}` of the
        smallest ball containing all the points and the radius
        :math:`r_{qbx}` of the smallest ball containing all the QBX
        expansion balls in the block. If the factor :math:`\theta \in
        [0, 1]`, then the radius of the proxy ball is

        .. math::

            r = (1 - \theta) r_{block} + \theta r_{qbx}.

        If the factor :math:`\theta > 1`, the the radius is simply

        .. math::

            r = \theta r_{qbx}.

    .. attribute:: ref_points

        Reference points on a unit ball. Can be used to construct the points
        of a proxy ball :math:`i` by translating them to ``center[i]`` and
        scaling by ``radii[i]``, as obtained by :meth:`__call__`.

    .. automethod:: __call__
    """

    def __init__(self, places, approx_nproxy=None, radius_factor=None):
        from pytential import GeometryCollection
        if not isinstance(places, GeometryCollection):
            places = GeometryCollection(places)

        self.places = places
        self.ambient_dim = places.ambient_dim
        self.radius_factor = 1.1 if radius_factor is None else radius_factor

        approx_nproxy = 32 if approx_nproxy is None else approx_nproxy
        self.ref_points = \
                _generate_unit_sphere(self.ambient_dim, approx_nproxy)

    @property
    def nproxy(self):
        return self.ref_points.shape[1]

    # {{{ proxy ball kernels

    @memoize_method
    def get_kernel(self):
        if self.radius_factor < 1.0:
            radius_expr = "(1.0 - {f}) * rblk + {f} * rqbx"
        else:
            radius_expr = "{f} * rqbx"
        radius_expr = radius_expr.format(f=self.radius_factor)

        knl = lp.make_kernel([
            "{[irange]: 0 <= irange < nranges}",
            "{[i]: 0 <= i < npoints}", "{[j]: 0 <= j < npoints}",
            "{[idim]: 0 <= idim < ndim}"
            ],
            """
            for irange
                <> ioffset = srcranges[irange]
                <> npoints = srcranges[irange + 1] - srcranges[irange]

                for idim
                    <> bbox_max = \
                        simul_reduce(max, i, sources[idim, srcindices[i + ioffset]])
                    <> bbox_min = \
                        simul_reduce(min, i, sources[idim, srcindices[i + ioffset]])

                    proxy_center[idim, irange] = (bbox_max + bbox_min) / 2.0
                end

                <> rblk = simul_reduce(max, j, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         sources[idim, srcindices[j + ioffset]]) ** 2))) \
                        {dup=idim}
            """
            + ("""
                <> rqbx_int = simul_reduce(max, j, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_int[idim, srcindices[j + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[j + ioffset]]) \
                         {dup=idim}
                <> rqbx_ext = simul_reduce(max, j, sqrt(simul_reduce(sum, idim, \
                        (proxy_center[idim, irange] -
                         center_ext[idim, srcindices[j + ioffset]]) ** 2)) + \
                         expansion_radii[srcindices[j + ioffset]]) \
                         {dup=idim}
                <> rqbx = rqbx_int if rqbx_ext < rqbx_int else rqbx_ext
            """ if self.radius_factor > 1.0e-14 else "<> rqbx = 0.0")
            + """
                proxy_radius[irange] = {radius_expr}
            end
            """.format(radius_expr=radius_expr),
            ([] if self.radius_factor < 1.0e-14 else [
                lp.GlobalArg("center_int", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C"),
                lp.GlobalArg("center_ext", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C")
            ]) + [
                lp.GlobalArg("sources", None,
                    shape=(self.ambient_dim, "nsources"), dim_tags="sep,C"),
                lp.GlobalArg("proxy_center", None,
                    shape=(self.ambient_dim, "nranges")),
                lp.GlobalArg("proxy_radius", None,
                    shape="nranges"),
                lp.ValueArg("nsources", np.int),
                "..."
            ],
            name="find_proxy_radii_knl",
            assumptions="ndim>=1 and nranges>=1",
            fixed_parameters=dict(ndim=self.ambient_dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.tag_inames(knl, "idim*:unr")
        return knl

    @memoize_method
    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "irange", 128, outer_tag="g.0")

        return knl

    def _get_proxy_centers_and_radii(self, actx, source_dd, indices, **kwargs):
        from pytential import bind, sym
        source_dd = sym.as_dofdesc(source_dd)
        discr = self.places.get_discretization(
                source_dd.geometry, source_dd.discr_stage)

        from meshmode.dof_array import flatten, thaw
        context = {
                "sources": flatten(thaw(actx, discr.nodes())),
                "srcindices": indices.indices,
                "srcranges": indices.ranges
                }

        if self.radius_factor > 1.0e-14:
            radii = bind(self.places, sym.expansion_radii(
                self.ambient_dim, dofdesc=source_dd))(actx)
            center_int = bind(self.places, sym.expansion_centers(
                self.ambient_dim, -1, dofdesc=source_dd))(actx)
            center_ext = bind(self.places, sym.expansion_centers(
                self.ambient_dim, +1, dofdesc=source_dd))(actx)

            context["center_int"] = flatten(center_int)
            context["center_ext"] = flatten(center_ext)
            context["expansion_radii"] = flatten(radii)

        knl = self.get_kernel()
        _, (centers, radii,) = knl(actx.queue, **context, **kwargs)

        return centers, radii

    # }}}

    def __call__(self, actx, source_dd, indices, **kwargs):
        """Generate proxy points for each given range of source points in
        the discretization in *source_dd*.

        :arg actx: a :class:`~meshmode.array_context.ArrayContext`.
        :arg source_dd: a :class:`~pytential.symbolic.primitives.DOFDescriptor`
            for the discretization on which the proxy points are to be
            generated.
        :arg indices: a :class:`sumpy.tools.BlockIndexRanges`.

        :return: a tuple of ``(proxies, pxyranges, pxycenters, pxyranges)``,
            where each element is a :class:`pyopencl.array.Array`. The
            sizes of the arrays are as follows: ``pxycenters`` is of size
            ``(2, nranges)``, ``pxyradii`` is of size ``(nranges,)``,
            ``pxyranges`` is of size ``(nranges + 1,)`` and ``proxies`` is
            of size ``(2, nranges * nproxy)``. The proxy points in a range
            :math:`i` can be obtained by a slice
            ``proxies[pxyranges[i]:pxyranges[i + 1]]`` and are all at a
            distance ``pxyradii[i]`` from the range center ``pxycenters[i]``.
        """

        # {{{ compute proxy centers and radii

        centers_dev, radii_dev = self._get_proxy_centers_and_radii(actx,
                source_dd, indices, **kwargs)

        from pytential.utils import flatten_to_numpy
        centers = np.vstack(flatten_to_numpy(actx, centers_dev))
        radii = flatten_to_numpy(actx, radii_dev)

        # }}}

        # {{{ build proxy points for each block

        def _affine_map(v, A, b):
            return np.dot(A, v) + b

        nproxy = self.nproxy * indices.nblocks
        proxies = np.empty((self.ambient_dim, nproxy), dtype=centers.dtype)
        pxy_nr_base = 0

        for i in range(indices.nblocks):
            bball = _affine_map(self.ref_points,
                    A=(radii[i] * np.eye(self.ambient_dim)),
                    b=centers[:, i].reshape(-1, 1))

            proxies[:, pxy_nr_base:pxy_nr_base + self.nproxy] = bball
            pxy_nr_base += self.nproxy

        proxies = make_obj_array([
            actx.freeze(actx.from_numpy(p)) for p in proxies
            ])
        centers = make_obj_array([
            actx.freeze(actx.from_numpy(c)) for c in centers
            ])
        pxyindices = np.arange(0, nproxy, dtype=indices.indices.dtype)
        pxyranges = np.arange(0, nproxy + 1, self.nproxy)

        assert proxies[0].size == pxyranges[-1]
        return BlockProxyPoints(
                indices=make_block_index(actx, pxyindices, pxyranges),
                points=proxies,
                centers=centers,
                radii=actx.freeze(radii_dev),
                )

# }}}


# {{{ gather_block_neighbor_points

def gather_block_neighbor_points(actx, discr, indices, pxy,
        max_particles_in_box=None):
    """Generate a set of neighboring points for each range of points in
    *discr*. Neighboring points of a range :math:`i` are defined
    as all the points inside the proxy ball :math:`i` that do not also
    belong to the range itself.

    :arg discr: a :class:`meshmode.discretization.Discretization`.
    :arg indices: a :class:`sumpy.tools.BlockIndexRanges`.
    :arg pxy: a :class:`BlockProxyPoints`.

    :return: a :class:`sumpy.tools.BlockIndexRanges`.
    """

    if max_particles_in_box is None:
        # FIXME: this is a fairly arbitrary value
        max_particles_in_box = 32

    indices = indices.get(actx.queue)

    from pytential.utils import flatten_to_numpy
    sources = flatten_to_numpy(actx, discr.nodes())
    sources = make_obj_array([
        actx.from_numpy(sources[idim][indices.indices])
        for idim in range(discr.ambient_dim)])

    # construct tree
    from boxtree import TreeBuilder
    builder = TreeBuilder(actx.context)
    tree, _ = builder(actx.queue, sources,
            max_particles_in_box=max_particles_in_box)

    from boxtree.area_query import AreaQueryBuilder
    builder = AreaQueryBuilder(actx.context)
    query, _ = builder(actx.queue, tree, pxy.centers, pxy.radii)

    # find nodes inside each proxy ball
    tree = tree.get(actx.queue)
    query = query.get(actx.queue)

    pxycenters = np.vstack([actx.to_numpy(c) for c in pxy.centers])
    pxyradii = actx.to_numpy(pxy.radii)

    nbrindices = np.empty(indices.nblocks, dtype=np.object)
    for iproxy in range(indices.nblocks):
        # get list of boxes intersecting the current ball
        istart = query.leaves_near_ball_starts[iproxy]
        iend = query.leaves_near_ball_starts[iproxy + 1]
        iboxes = query.leaves_near_ball_lists[istart:iend]

        # get nodes inside the boxes
        istart = tree.box_source_starts[iboxes]
        iend = istart + tree.box_source_counts_cumul[iboxes]
        isources = np.hstack([np.arange(s, e)
                              for s, e in zip(istart, iend)])
        nodes = np.vstack([s[isources] for s in tree.sources])
        isources = tree.user_source_ids[isources]

        # get nodes inside the ball but outside the current range
        center = pxycenters[:, iproxy].reshape(-1, 1)
        radius = pxyradii[iproxy]
        mask = ((la.norm(nodes - center, axis=0) < radius)
                & ((isources < indices.ranges[iproxy])
                    | (indices.ranges[iproxy + 1] <= isources)))

        nbrindices[iproxy] = indices.indices[isources[mask]]

    return make_block_index(actx, nbrindices)

# }}}

# vim: foldmethod=marker
