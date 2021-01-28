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

from pytools.obj_array import make_obj_array
from pytools import Record

from sumpy.tools import MatrixBlockIndexRanges
from pytential.symbolic.mappers import IdentityMapper, LocationTagger


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


# }}}


# {{{ evaluation

class QBXForcedLimitReplacer(IdentityMapper):
    def __init__(self, qbx_forced_limit=None):
        super(QBXForcedLimitReplacer, self).__init__()
        self.qbx_forced_limit = qbx_forced_limit

    def map_int_g(self, expr):
        return expr.copy(qbx_forced_limit=self.qbx_forced_limit)


class LocationReplacer(LocationTagger):
    def _default_dofdesc(self, dofdesc):
        return self.default_where

    def map_int_g(self, expr):
        return type(expr)(
                expr.kernel,
                self.operand_rec(expr.density),
                expr.qbx_forced_limit,
                self.default_source, self.default_where,
                kernel_arguments=dict(
                    (name, self.operand_rec(arg_expr))
                    for name, arg_expr in expr.kernel_arguments.items()
                    ))


class DOFDescriptorReplacer(LocationReplacer):
    def __init__(self, default_source, default_target):
        super(DOFDescriptorReplacer, self).__init__(
                default_target, default_source=default_source)
        self.operand_rec = LocationReplacer(
                default_source, default_source=default_source)


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
        r"""
        :arg exprs: an :class:`~numpy.ndarray` of expressions (layer potentials)
            that correspond to the output blocks of the matrix.
        :arg input_exprs: a :class:`list` of densities that correspond to the
            input blocks of the matrix.
        :arg domains: a :class:`list` of the same length as *input_exprs*
            defining the domain of each input.

        :arg context: a :class:`dict` with additional parameters required to
            evaluated the expressions.

        :arg weighted_farfield: a ``Tuple[bool, bool]``, where the first
            entry refers to the sources and the second to the targets. If
            the entry is *True*, farfield evaluation is multiplied by quadrature
            weights.
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

        self.weighted_source = weighted_farfield[0]
        self.weighted_target = weighted_farfield[1]

        self.nearfield_block_builder = nearfield_block_builder
        self.farfield_block_builder = farfield_block_builder

    def _prepare_farfield_expr(self, places, expr, auto_where=None):
        from pytential.symbolic.execution import _prepare_auto_where
        auto_where = _prepare_auto_where(auto_where, places=places)

        from pytential.symbolic.mappers import OperatorCollector
        expr = QBXForcedLimitReplacer(qbx_forced_limit=None)(expr)
        expr = DOFDescriptorReplacer(auto_where[0], auto_where[1])(expr)
        expr, = OperatorCollector()(expr)

        return expr

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
        expr = self._prepare_farfield_expr(
                places, self.exprs[ibrow], auto_where=auto_where)

        return self._evaluate(actx, places,
                self.farfield_block_builder,
                expr, ibcol, index_set, auto_where,
                weighted=self.weighted_source,
                exclude_self=False)

    def evaluate_target_farfield(self,
            actx, places, ibrow, ibcol, index_set, auto_where=None):
        expr = self._prepare_farfield_expr(
                places, self.exprs[ibrow], auto_where=auto_where)

        return self._evaluate(actx, places,
                self.farfield_block_builder,
                expr, ibcol, index_set, auto_where,
                weighted=self.weighted_target,
                exclude_self=False)

    def evaluate_nearfield(self,
            actx, places, ibrow, ibcol, index_set, auto_where=None):
        from pytential.symbolic.execution import _prepare_expr
        expr = _prepare_expr(places, self.exprs[ibrow], auto_where=auto_where)

        return self._evaluate(actx, places,
                self.nearfield_block_builder,
                expr, ibcol, index_set, auto_where)


def make_block_evaluation_wrangler(places, exprs, input_exprs,
        domains=None, context=None, auto_where=None,
        _weighted_farfield=None,
        _farfield_block_builder=None,
        _nearfield_block_builder=None):

    from pytential.symbolic.execution import _prepare_auto_where
    auto_where = _prepare_auto_where(auto_where, places)

    if not (isinstance(exprs, np.ndarray) and exprs.dtype.char == "O"):
        exprs = make_obj_array([exprs])

    try:
        input_exprs = list(input_exprs)
    except TypeError:
        input_exprs = [input_exprs]

    from pytential.symbolic.execution import _prepare_domains
    domains = _prepare_domains(len(input_exprs), places, domains, auto_where[0])

    if context is None:
        context = {}

    if _weighted_farfield is None:
        weighted_source = True
        weighted_target = False
    elif isinstance(_weighted_farfield, bool):
        weighted_source = weighted_target = _weighted_farfield
    elif isinstance(_weighted_farfield, (list, tuple)):
        weighted_source, weighted_target = _weighted_farfield
    else:
        raise ValueError("unknown value for weighting: `{}`".format(
            _weighted_farfield))

    if _nearfield_block_builder is None:
        from pytential.symbolic.matrix import NearFieldBlockBuilder
        _nearfield_block_builder = NearFieldBlockBuilder

    if _farfield_block_builder is None:
        from pytential.symbolic.matrix import FarFieldBlockBuilder
        _farfield_block_builder = FarFieldBlockBuilder

    return BlockEvaluationWrangler(
            exprs=exprs,
            input_exprs=input_exprs,
            domains=domains,
            context=context,
            weighted_farfield=(weighted_source, weighted_target),
            farfield_block_builder=_farfield_block_builder,
            nearfield_block_builder=_nearfield_block_builder)

# }}}


# {{{ skeletonize_block_by_proxy

@contextmanager
def add_to_geometry_collection(places, proxy):
    # NOTE: this is a giant hack to keep all the caches in `places` and
    # just add the proxy points to it, since otherwise all the DISCRETIZATION
    # scope stuff would be recomputed all the time

    try:
        pxyplaces = places.merge({"proxy": proxy})
        yield pxyplaces
    finally:
        pass

    # try:
    #     # NOTE: this needs to match `EvaluationMapper.map_common_subexpression`
    #     previous_cse_cache = places._get_cache("cse")
    #     places.places["proxy"] = proxy
    #     yield places
    # finally:
    #     del places.places["proxy"]
    #     # NOTE: this is meant to make sure that proxy-related things don't
    #     # get cached over multiple runs and lead to some explosion of some sort
    #     places.caches["cse"] = previous_cse_cache


def make_block_proxy_skeleton(actx, ibrow, ibcol,
        places, proxy_generator, wrangler, indices,
        source_or_target=None,
        max_particles_in_box=None):
    """Builds a block matrix that can be used to skeletonize the
    rows (targets) or columns (sources) of the symbolic matrix block
    described by ``(ibrow, ibcol)``.
    """

    if indices.nblocks == 1:
        raise ValueError("cannot make a proxy skeleton for a single block")

    if source_or_target == "source":
        from pytential.target import PointsTarget as ProxyPoints

        swap_arg_order = lambda x, y: (x, y)        # noqa: E731
        evaluate_farfield = wrangler.evaluate_source_farfield
        block_stack = np.vstack
    elif source_or_target == "target":
        from pytential.source import PointPotentialSource as ProxyPoints

        swap_arg_order = lambda x, y: (y, x)        # noqa: E731
        evaluate_farfield = wrangler.evaluate_target_farfield
        block_stack = np.hstack
    else:
        raise ValueError(f"unknown value: '{source_or_target}'")

    # {{{ generate proxies

    domain = wrangler.domains[ibcol]
    dep_discr = places.get_discretization(domain.geometry, domain.discr_stage)
    pxy = proxy_generator(actx, domain, indices)

    # }}}

    # {{{ evaluate (farfield) proxy interactions

    with add_to_geometry_collection(places, ProxyPoints(pxy.points)) as pxyplaces:
        pxyindices = MatrixBlockIndexRanges(actx.context,
                *swap_arg_order(pxy.indices, indices))
        pxymat = evaluate_farfield(actx, pxyplaces, ibrow, ibcol, pxyindices,
                auto_where=swap_arg_order(domain, "proxy"))

    # }}}

    # {{{ evaluate (nearfield) neighbor interactions

    from pytential.linalg.proxy import gather_block_neighbor_points
    nbrindices = gather_block_neighbor_points(
            actx, dep_discr, indices, pxy,
            max_particles_in_box=max_particles_in_box)

    nbrindices = MatrixBlockIndexRanges(actx.context,
            *swap_arg_order(nbrindices, indices))
    nbrmat = wrangler.evaluate_nearfield(actx, places, ibrow, ibcol, nbrindices,
            auto_where=domain)

    # }}}

    # {{{ concatenate everything to get the blocks ready for ID-ing

    pxyindices = pxyindices.get(actx.queue)
    nbrindices = nbrindices.get(actx.queue)

    pxyblk = np.full((indices.nblocks, indices.nblocks), 0, dtype=np.object)
    for i in range(indices.nblocks):
        pxyblk[i, i] = block_stack(swap_arg_order(
            pxyindices.block_take(pxymat, i),
            nbrindices.block_take(nbrmat, i),
            ))
    # }}}

    return pxyblk


def _skeletonize_block_by_proxy_with_mats(actx,
        ibcol, ibrow, places, proxy_generator, wrangler, blkindices,
        id_eps=None, id_rank=None,
        max_particles_in_box=None):
    L = np.full((blkindices.nblocks, blkindices.nblocks), 0, dtype=np.object)
    R = np.full((blkindices.nblocks, blkindices.nblocks), 0, dtype=np.object)

    if blkindices.nblocks == 1:
        L[0, 0] = np.eye(blkindices.row.indices.size)
        R[0, 0] = np.eye(blkindices.col.indices.size)

        return L, R, blkindices

    # construct proxy matrices to skeletonize
    src_mat = make_block_proxy_skeleton(actx, ibrow, ibcol,
            places, proxy_generator, wrangler, blkindices.col,
            source_or_target="source",
            max_particles_in_box=max_particles_in_box)
    tgt_mat = make_block_proxy_skeleton(actx, ibrow, ibcol,
            places, proxy_generator, wrangler, blkindices.row,
            source_or_target="target",
            max_particles_in_box=max_particles_in_box)

    src_skl_indices = np.empty(blkindices.nblocks, dtype=np.object)
    tgt_skl_indices = np.empty(blkindices.nblocks, dtype=np.object)
    skl_ranges = np.zeros(blkindices.nblocks + 1, dtype=np.int)

    src_indices = blkindices.col.get(actx.queue)
    tgt_indices = blkindices.row.get(actx.queue)

    for i in range(blkindices.nblocks):
        k = id_rank

        # skeletonize target points
        assert not np.any(np.isinf(tgt_mat[i, i])), \
                np.where(np.isinf(tgt_mat[i, i]))
        assert not np.any(np.isnan(tgt_mat[i, i])), \
                np.where(np.isnan(tgt_mat[i, i]))

        k, idx, interp = interp_decomp(tgt_mat[i, i].T, k, id_eps)
        assert k > 0

        L[i, i] = interp.T
        tgt_skl_indices[i] = tgt_indices.block_indices(i)[idx[:k]]

        # skeletonize source points
        assert not np.any(np.isinf(src_mat[i, i])), \
                np.where(np.isinf(src_mat[i, i]))
        assert not np.any(np.isnan(src_mat[i, i])), \
                np.where(np.isnan(src_mat[i, i]))

        k, idx, interp = interp_decomp(src_mat[i, i], k, id_eps)
        assert k > 0

        R[i, i] = interp
        src_skl_indices[i] = src_indices.block_indices(i)[idx[:k]]

        skl_ranges[i + 1] = skl_ranges[i] + k
        assert R[i, i].shape == (k, src_mat[i, i].shape[1])
        assert L[i, i].shape == (tgt_mat[i, i].shape[0], k)

    from pytential.linalg.proxy import make_block_index
    src_skl_indices = make_block_index(actx, np.hstack(src_skl_indices), skl_ranges)
    tgt_skl_indices = make_block_index(actx, np.hstack(tgt_skl_indices), skl_ranges)
    skl_indices = MatrixBlockIndexRanges(actx.context,
            tgt_skl_indices, src_skl_indices)

    return L, R, skl_indices, src_mat, tgt_mat


def skeletonize_block_by_proxy(actx,
        ibcol, ibrow, places, proxy_generator, wrangler, blkindices,
        id_eps=None, id_rank=None,
        max_particles_in_box=None):
    L, R, sklindices, _, _ = _skeletonize_block_by_proxy_with_mats(actx,
            ibcol, ibrow, places, proxy_generator, wrangler, blkindices,
            id_eps=id_eps,
            id_rank=id_rank,
            max_particles_in_box=max_particles_in_box)

    return L, R, sklindices

# }}}


# {{{ skeletonize_by_proxy

class SkeletonizedBlock(Record):
    """
    .. attribute:: L
    .. attribute:: R
    .. attribute:: sklindices
    """


def skeletonize_by_proxy(actx, places, proxy_generator, wrangler, blkindices,
        id_eps=None, id_rank=None, max_particles_in_box=None):
    r"""
    :arg places: a :class:`~meshmode.array_context.ArrayContext`.
    :arg proxy_generator: a :class:`~pytential.linalg.proxy.ProxyGenerator`.
    :arg wrangler: a :class:`BlockEvaluationWrangler`.
    :arg blkindices: a :class:`~sumpy.tools.MatrixBlockIndexRanges`.

    :returns: a tuple ``(L, R, sklindices)`` encoding the block-by-block
        decompression of the matrix represented *wrangler*. :math:`L` and
        :math:`R` are :math:`n \times n` diagonal block matrix, where
        :math:`n` is :attr:`~sumpy.tools.MatrixBlockIndexRanges.nblocks``. The
        ``sklindices`` array contains the remaining (skeleton) nodes from
        ``blkindices`` after compression.
    """

    nrows = len(wrangler.exprs)
    ncols = len(wrangler.input_exprs)
    skel = np.empty((nrows, ncols), dtype=np.object)

    for ibrow in range(nrows):
        for ibcol in range(ncols):
            L, R, sklindices = skeletonize_block_by_proxy(actx,
                    ibrow, ibcol, places, proxy_generator, wrangler, blkindices,
                    id_eps=id_eps, id_rank=id_rank,
                    max_particles_in_box=max_particles_in_box)

            skel[ibrow, ibcol] = SkeletonizedBlock(L=L, R=R, sklindices=sklindices)

    return skel

# }}}
