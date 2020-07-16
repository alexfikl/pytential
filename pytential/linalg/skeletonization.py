from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2018-2020 Alexandru Fikl"

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

from contextlib import contextmanager

import numpy as np
import numpy.linalg as la

import pyopencl as cl

from pytools.obj_array import make_obj_array
from pytools import Record

from sumpy.tools import MatrixBlockIndexRanges, BlockIndexRanges


# {{{ helpers

def interp_decomp(A, rank, eps):
    """Wrapper for :func:`~scipy.linalg.interpolative.interp_decomp` that
    always has the same output signature.

    :return: a tuple ``(k, idx, interp)`` containing the numerical rank,
        the column indices and the resulting interpolation matrix.
    """

    import scipy.linalg.interpolative as sli    # pylint:disable=no-name-in-module
    if rank is None:
        k, idx, proj = sli.interp_decomp(A, eps)
    else:
        idx, proj = sli.interp_decomp(A, rank)
        k = rank

    # NOTE: fix should be in scipy 1.2.0
    # https://github.com/scipy/scipy/pull/9125
    if k == A.shape[1]:
        proj = np.empty((k, 0), dtype=proj.dtype)

    interp = sli.reconstruct_interp_matrix(idx, proj)
    return k, idx, interp


def make_block_diag(blk, blkindices):
    """Construct a block diagonal matrix from a linear representation of
    the matrix blocks.
    """

    nblocks = blkindices.nblocks
    diag = np.full((nblocks, nblocks), 0, dtype=np.object)

    for i in range(nblocks):
        diag[i, i] = blkindices.block_take(blk, i)

    return diag


class SkeletonizedBlock(Record):
    """
    .. attribute:: L
    .. attribute:: R
    .. attribute:: sklindices
    """

# }}}


# {{{ evaluation

class QBXForcedLimitReplacer(IdentityMapper):
    def __init__(self, qbx_forced_limit=None):
        super(QBXForcedLimitReplacer, self).__init__()
        self.qbx_forced_limit = qbx_forced_limit

    def map_int_g(self, expr):
        return expr.copy(qbx_forced_limit=self.qbx_forced_limit)


class DOFDescReplacer(ToTargetTagger):
    def _default_dofdesc(self, dofdesc):
        return self.default_where


def prepare_expr(places, expr, auto_where=None, qbx_forced_limit=None):
    from pytential.symbolic.execution import _prepare_auto_where
    auto_where = _prepare_auto_where(auto_where, places=places)

    from pytential.symbolic.execution import _prepare_expr
    expr = _prepare_expr(places, expr, auto_where=auto_where)

    expr = QBXForcedLimitReplacer(qbx_forced_limit)(expr)
    expr = DOFDescReplacer(auto_where[0], auto_where[1])(expr)

    return expr


class BlockEvaluationWrangler:
    """
    .. automethod:: evaluate_source_farfield
    .. automethod:: evaluate_target_farfield
    .. automethod:: evaluate_nearfield
    """

    def __init__(self, exprs, input_exprs, domains,
            context=None,
            weighted_farfield=None,
            farfield_block_builder=None,
            nearfield_block_builder=None):
        """
        :arg exprs: an :class:`~numpy.ndarray` of expressions (layer potentials)
            that correspond to the output blocks of the matrix.
        :arg input_exprs: a :class:`list` of densities that correspond to the
            input blocks of the matrix.
        :arg domains: a :class:`list` of the same length as *input_exprs*
            defining the domain of each input.

        :arg context: a :class:`dict` with additional parameters required to
            evaluated the expressions.

        :arg weighted_farfield: a :class:`tuple` containing two :class:`bool`\ s
            which turn on the weighing proxy interactions by the corresponding
            quadrature weights.
        :arg farfield_block_builder:
            a :class:`pytential.symbolic.matrix.MatrixBlockBuilderBase` that
            is used to evaluate farfield proxy interactions.
        :arg nearfield_block_builder:
            a :class:`pytential.symbolic.matrix.MatrixBlockBuilderBase` that
            is used to evaluate nearfield neighbour interactions.
        """
        if context is None:
            context = {}

        self.exprs = exprs
        self.input_exprs = input_exprs
        self.domains = domains
        self.context = context

        if weighted_farfield is None:
            weighted_source = True
            weighted_target = False
        elif isinstance(weighted_farfield, bool):
            weighted_source = weighted_target = weighted_farfield
        elif isinstance(weighted_farfield, (list, tuple)):
            weighted_source, weighted_target = weighted_farfield
        else:
            raise ValueError("unknown value for weighting: `{}`".format(
                weighted_farfield))

        self.weighted_source = weighted_source
        self.weighted_target = weighted_target

        self.nearfield_block_builder = nearfield_block_builder
        if self.nearfield_block_builder is None:
            from pytential.symbolic.matrix import NearFieldBlockBuilder
            self.nearfield_block_builder = NearFieldBlockBuilder

        self.farfield_block_builder = farfield_block_builder
        if self.farfield_block_builder is None:
            from pytential.symbolic.matrix import FarFieldBlockBuilder
            self.farfield_block_builder = FarFieldBlockBuilder

    def _evaluate(self, actx, places, builder_cls,
            expr, idomain, index_set, auto_where, **kwargs):
        domain = self.domains[idomain]
        dep_source = places.get_geometry(domain.geometry)
        dep_discr = places.get_discretization(domain.geometry, domain.discr_stage)

        builder = builder_cls(actx,
                dep_expr=self.input_exprs[idomain],
                other_dep_exprs=(
                    self.input_exprs[:idomain]
                    + self.input_exprs[idomain+1:]),
                dep_source=dep_source,
                dep_discr=dep_discr,
                places=places,
                index_set=index_set,
                context=self.context,
                **kwargs)

        return builder(expr)

    def evaluate_source_farfield(self,
            actx, places, ibrow, ibcol, index_set, auto_where=None):
        expr = prepare_expr(places, expr, auto_where=auto_where)
        return self._evaluate(actx, places,
                self.farfield_block_builder,
                expr, ibcol, index_set, auto_where,
                weighted=self.weighted_source,
                exclude_self=False)

    def evaluate_target_farfield(self,
            actx, places, ibrow, ibcol, index_set, auto_where=None):
        expr = prepare_expr(places, expr, auto_where=auto_where)
        return self._evaluate(actx, places,
                self.farfield_block_builder,
                expr, ibcol, index_set, auto_where,
                weighted=self.weighted_target,
                exclude_self=False)

    def evaluate_nearfield(self,
            actx, places, ibrow, ibcol, index_set, auto_where=None):
        return self._evaluate(actx, places,
                self.nearfield_block_builder,
                self.exprs[ibrow], ibcol, index_set, auto_where)


def make_block_evaluation_wrangler(places, exprs, input_exprs,
        domains=None, context=None,
        _weighted_farfield=None,
        _farfield_block_builder=None,
        _nearfield_block_builder=None):

    if not is_obj_array(exprs):
        exprs = make_obj_array([exprs])

    try:
        input_exprs = list(input_exprs)
    except TypeError:
        input_exprs = [input_exprs]

    from pytential.symbolic.execution import _prepare_auto_where
    auto_where = _prepare_auto_where(auto_where, places)
    from pytential.symbolic.execution import _prepare_domains
    domains = _prepare_domains(len(input_exprs), places, domains, auto_where[0])

    if context is None:
        context = {}

    return BlockEvaluationWrangler(
            exprs=exprs,
            input_exprs=input_exprs,
            domains=domains,
            context=context,
            weighted_farfield=_weighted_farfield,
            farfield_block_builder=_farfield_block_builder,
            nearfield_block_builder=_nearfield_block_builder)

# }}}


# {{{ skeletonize_by_proxy

@contextmanager
def add_to_geometry_collection(places, proxy):
    # NOTE: this is a bit of a hack to keep all the caches in `places` and
    # just add the proxy points to it, since otherwise all the DISCRETIZATION
    # scope stuff would be recomputed all the time

    try:
        previous_cse_cache = places._get_cache("cse")
        places.places["proxy"] = proxy
        yield places
    finally:
        del places.places["proxy"]
        # NOTE: this is meant to make sure that proxy-related things don't
        # get cached over multiple runs and lead to some explosion of some sort
        places.cache["cse"] = previous_cse_cache


def make_block_proxy_skeleton(actx, places, proxy, wrangler, indices,
        ibrow, ibcol, source_or_target,
        max_particles_in_box=None):
    """Builds a block matrix that can be used to skeletonize the rows
    (targets) of the symbolic matrix block described by ``(ibrow, ibcol)``.
    """

    if source_or_target == "source":
        from pytential.target import PointsTarget as ProxyPoints

        swap_arg_order = lambda x, y: (x, y)
        evaluate_farfield = wrangler.evaluate_source_farfield
    elif source_or_target == "target":
        from pytential.source import PointPotentialSource as ProxyPoints

        swap_arg_order = lambda x, y: (y, x)
        evaluate_farfield = wrangler.evaluate_target_farfield
    else:
        raise ValueError(f"unknown value: '{source_or_target}'")

    # {{{ generate proxies

    domain = wrangler.domains[ibcol]
    dep_source = places.get_geometry(domain.geometry)
    dep_discr = places.get_discretization(domain.geometry, domain.discr_stage)

    pxy = proxy(actx, domain, indices)

    # }}}

    # {{{ evaluate (farfield) proxy interactions

    with add_to_geometry_collection(places, ProxyPoints(pxy.points)):
        pxyindices = MatrixBlockIndexRanges(actx.context,
                *swap_arg_order(pxy.indices, indices))
        pxymat = evaluate_farfield(actx, pxyplaces, ibrow, ibcol, pxyindices,
                auto_where=swap_arg_order(domain, "proxy"))

    if indices.nblocks == 1:
        # TODO: evaluate nearfield at the root level?
        return make_block_diag(pxymat, pxyindices.get(actx.queue))

    # }}}

    # {{{ evaluate (nearfield) neighbor interactions

    nbrindices = gather_block_neighbor_points(
            actx, dep_discr, indices, pxy,
            max_particles_in_box=max_particles_in_box)

    nbrindices = MatrixBlockIndexRanges(actx.context,
            *swap_arg_order(nbrindices, indices))
    nbrmat = wrangler.evaluate_nearfield(actx, places, ibrow, ibcol, nbrindices)

    # }}}

    # {{{ concatenate everything to get the blocks ready for ID-ing

    pxyindices = pxyindices.get(queue)
    nbrindices = nbrindices.get(queue)

    pxyblk = np.full((indices.nblocks, indices.nblocks), 0, dtype=np.object)
    for i in range(indices.nblocks):
        pxyblk[i, i] = np.hstack(swap_arg_order(
            pxyindices.block_take(pxymat, i)
            nbrindices.block_take(nbrmat, i),
            ))
    # }}}

    return pxyblk


def skeletonize_block_by_proxy(actx,
        ibcol, ibrow, places, proxy, wrangler, blkindices,
        id_eps=None, id_rank=None,
        tree_max_particles_in_box=None):
    r"""
    :arg places: a :class:`~meshmode.array_context.ArrayContext`.
    :arg proxy: a :class:`~pytential.linalg.proxy.ProxyGenerator`.
    :arg wrangler: a :class:`BlockEvaluationWrangler`.
    :arg blkindices: a :class:`~sumpy.tools.MatrixBlockIndexRanges`.

    :returns: a tuple ``(L, R, sklindices)`` encoding the block-by-block
        decompression of the matrix represented *wrangler*. :math:`L` and
        :math:`R` are :math:`n \times n` diagonal block matrix, where
        :math:`n` is :attr:`~sumpy.tools.MatrixBlockIndexRanges.nblocks``. The
        ``sklindices`` array contains the remaining (skeleton) nodes from
        ``blkindices`` after compression.
    """

    L = np.full((blkindices.nblocks, blkindices.nblocks), 0, dtype=np.object)
    R = np.full((blkindices.nblocks, blkindices.nblocks), 0, dtype=np.object)

    if blkindices.nblocks == 1:
        L[0, 0] = np.eye(blkindices.row.indices.size)
        R[0, 0] = np.eye(blkindices.col.indices.size)

        return L, R, blkindices

    # construct proxy matrices to skeletonize
    src_mat = build_source_skeleton_matrix(queue,
            places, proxy, wrangler, blkindices.col, 0, 0,
            max_particles_in_box=tree_max_particles_in_box)
    tgt_mat = build_target_skeleton_matrix(queue,
            places, proxy, wrangler, blkindices.row, 0, 0,
            max_particles_in_box=tree_max_particles_in_box)

    src_skl_indices = np.empty(blkindices.nblocks, dtype=np.object)
    tgt_skl_indices = np.empty(blkindices.nblocks, dtype=np.object)
    skl_ranges = np.zeros(blkindices.nblocks + 1, dtype=np.int)

    src_indices = blkindices.col.get(queue)
    tgt_indices = blkindices.row.get(queue)

    for i in range(blkindices.nblocks):
        k = id_rank

        assert not np.any(np.isnan(src_mat[i, i])), "block {}".format(i)
        assert not np.any(np.isinf(src_mat[i, i])), "block {}".format(i)
        assert not np.any(np.isnan(tgt_mat[i, i])), "block {}".format(i)
        assert not np.any(np.isinf(tgt_mat[i, i])), "block {}".format(i)

        # skeletonize target points
        k, idx, interp = _interp_decomp(tgt_mat[i, i].T, k, id_eps)
        assert k > 0

        L[i, i] = interp.T
        tgt_skl_indices[i] = tgt_indices.block_indices(i)[idx[:k]]

        # skeletonize source points
        k, idx, interp = _interp_decomp(src_mat[i, i], k, id_eps)
        assert k > 0

        R[i, i] = interp
        src_skl_indices[i] = src_indices.block_indices(i)[idx[:k]]

        skl_ranges[i + 1] = skl_ranges[i] + k
        assert R[i, i].shape == (k, src_mat[i, i].shape[1])
        assert L[i, i].shape == (tgt_mat[i, i].shape[0], k)

    src_skl_indices = _to_block_index(queue, src_skl_indices, skl_ranges)
    tgt_skl_indices = _to_block_index(queue, tgt_skl_indices, skl_ranges)
    skl_indices = MatrixBlockIndexRanges(queue.context,
                                         tgt_skl_indices,
                                         src_skl_indices)

    return L, R, skl_indices


def skeletonize_by_proxy(actx, places, proxy, wrangler, blkindices,
        id_eps=None, id_rank=None,
        max_particles_in_box=None):

    nrows = len(wrangler.exprs)
    ncols = len(wrangler.input_exprs)
    skel = np.empty((nrows, ncols), dtype=np.object)

    for ibrow in range(nrows):
        for ibcol in range(ncols)
            skel[ibrow, ibcol] = skeletonize_block_by_proxy(actx,
                    ibrow, ibcol, proxy, wrangler, blkindices,
                    id_eps=id_eps, id_rank=id_rank,
                    max_particles_in_box=max_particles_in_box)

    return skel

# }}}
