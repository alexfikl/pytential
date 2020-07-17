import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array
from sumpy.tools import BlockIndexRanges, MatrixBlockIndexRanges

from pytential import sym

import extra_int_eq_data as extra


# {{{ helpers

def max_block_error(mat, blk, index_set, p=None):
    error = -np.inf
    for i in range(index_set.nblocks):
        mat_i = index_set.take(mat, i)
        blk_i = index_set.block_take(blk, i)

        error = max(error, la.norm(mat_i - blk_i, ord=p) / la.norm(mat_i, ord=p))

    return error


def skeletonization_error(mat, skel, srcindices, p=None):
    r"""Matrix errors are computed as follows:

        .. math::

            \begin{aligned}
            \epsilon_{l, ij} = \|A_{ij} - L_{ii} S_{ij}\|_p, \\
            \epsilon_{r, ij} = \|A_{ij} - S_{ij} R_{jj}\|_p, \\
            \epsilon_f = \|A - L S R\|_p.
            \end{aligned}

    :arg mat: dense matrix.
    :arg skel: a :class:`~pytential.linalg.skeletonization.SkeletonizedBlock`.
    :arg p: norm type (follows :func:`~numpy.linalg.norm` rules for matrices).

    :returns: a tuple ``(err_l, err_r, err_f)`` of the left, right and full
        matrix errors.
    """
    from itertools import product

    L = skel.L
    R = skel.R
    sklindices = skel.sklindices

    def mnorm(x, y):
        return la.norm(x - y, ord=p) / la.norm(x, ord=p)

    # build block matrices
    nblocks = srcindices.nblocks
    S = np.full((nblocks, nblocks), 0, dtype=np.object)
    A = np.full((nblocks, nblocks), 0, dtype=np.object)

    def block_indices(blk, i):
        return blk.indices[blk.ranges[i]:blk.ranges[i + 1]]

    # compute max block error
    err_l = np.empty((nblocks, nblocks))
    err_l = np.empty((nblocks, nblocks))
    for i, j in product(range(nblocks), repeat=2):
        if i == j:
            continue

        # full matrix indices
        f_tgt = block_indices(srcindices.row, i)
        f_src = block_indices(srcindices.col, j)
        # skeleton matrix indices
        s_tgt = block_indices(sklindices.row, i)
        s_src = block_indices(sklindices.col, j)

        S[i, j] = mat[np.ix_(s_tgt, s_src)]
        A[i, j] = mat[np.ix_(f_tgt, f_src)]

        blk = mat[np.ix_(s_tgt, f_src)]
        err_l[i, j] = mnorm(A[i, j], L[i, i].dot(blk))

        blk = mat[np.ix_(f_tgt, s_src)]
        err_r[i, j] = mnorm(A[i, j], blk.dot(R[j, j]))

    # compute full matrix error
    from pytential.symbolic.execution import _bmat
    A = _bmat(A, dtype=mat.dtype)
    L = _bmat(L, dtype=mat.dtype)
    S = _bmat(S, dtype=mat.dtype)
    R = _bmat(R, dtype=mat.dtype)

    assert L.shape == (A.shape[0], S.shape[0])
    assert R.shape == (S.shape[1], A.shape[1])
    err_f = mnorm(A, L.dot(S.dot(R)))

    return err_l, err_r, err_f

# }}}


# {{{ MatrixTestCase

class MatrixTestCaseMixin:
    # operators
    op_type = "scalar"
    # disable fmm for matrix tests
    fmm_backend = None

    # partitioning
    approx_block_count = 10
    max_particles_in_box = None
    tree_kind = "adaptive-level-restricted"
    index_sparsity_factor = 1.0

    # proxy
    proxy_radius_factor = None
    proxy_approx_count = None

    # skeletonization
    id_eps = 1.0e-8
    skel_discr_stage = sym.QBX_SOURCE_STAGE2

    weighted_farfield = None
    farfield_block_builder = None
    nearfield_block_builder = None

    def get_block_indices(self, actx, discr, matrix_indices=True):
        max_particles_in_box = self.max_particles_in_box
        if max_particles_in_box is None:
            max_particles_in_box = discr.ndofs // self.approx_block_count

        from pytential.linalg.proxy import partition_by_nodes
        indices = partition_by_nodes(actx, discr,
                tree_kind=self.tree_kind,
                max_particles_in_box=max_particles_in_box)

        if abs(self.index_sparsity_factor - 1.0) < 1.0e-14:
            if not matrix_indices:
                return indices
            return MatrixBlockIndexRanges(actx.context, indices, indices)

        # randomly pick a subset of points
        indices = indices.get(actx.queue)

        subset = np.empty(indices.nblocks, dtype=np.object)
        for i in range(indices.nblocks):
            iidx = indices.block_indices(i)
            isize = int(self.index_sparsity_factor * len(iidx))
            isize = max(1, min(isize, len(iidx)))

            subset[i] = np.sort(np.random.choice(iidx, size=isize, replace=False))

        ranges = actx.from_numpy(np.cumsum([0] + [r.shape[0] for r in subset]))
        indices = actx.from_numpy(np.hstack(subset))

        indices = BlockIndexRanges(actx.context,
                actx.freeze(indices), actx.freeze(ranges))

        if not matrix_indices:
            return indices
        return MatrixBlockIndexRanges(actx.context, indices, indices)

    def get_operator(self, ambient_dim, qbx_forced_limit="avg"):
        knl = self.knl_class(ambient_dim)
        kwargs = self.knl_sym_kwargs.copy()
        kwargs["qbx_forced_limit"] = qbx_forced_limit

        if self.op_type == "scalar":
            sym_u = sym.var("u")
            sym_op = sym.S(knl, sym_u, **kwargs)
        elif self.op_type == "scalar_mixed":
            sym_u = sym.var("u")
            sym_op = sym.S(knl, 0.3 * sym_u, **kwargs) \
                    + sym.D(knl, 0.5 * sym_u, **kwargs)
        elif self.op_type == "vector":
            sym_u = sym.make_sym_vector("u", ambient_dim)

            sym_op = make_obj_array([
                sym.Sp(knl, sym_u[0], **kwargs)
                + sym.D(knl, sym_u[1], **kwargs),
                sym.S(knl, 0.4 * sym_u[0], **kwargs)
                + 0.3 * sym.D(knl, sym_u[0], **kwargs)
                ])
        else:
            raise ValueError(f"unknown operator type: '{self.op_type}'")

        return sym_u, sym_op


class CurveTestCase(MatrixTestCaseMixin, extra.CurveTestCase):
    pass


class TorusTestCase(MatrixTestCaseMixin, extra.TorusTestCase):
    pass

# }}}
