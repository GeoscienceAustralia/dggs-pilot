import numpy as np
import pyproj
import cv2
from .tools import apply_affine


class DGGS(object):
    n2i = dict(S=0,
               O=4, P=5, Q=6, R=7,
               N=1)

    i2n = np.r_[['S', 'N', '_', '_',
                 'O', 'P', 'Q', 'R']]

    proj_opts = dict(proj='rhealpix',
                     no_defs=True,
                     wkext=True,
                     datum='WGS84',
                     lat_0=0,
                     south_square=0,
                     north_square=0)

    @staticmethod
    def _compute_norm_factor(rh):
        xrange, _ = rh([-180, 180], [0, 0])
        return 4.0/(xrange[1] - xrange[0])

    @staticmethod
    def address_to_tuple(s):
        assert isinstance(s, str)
        return (s[0],) + tuple(int(x) for x in s[1:])

    @staticmethod
    def addr2ixys(addr):
        ch, *cells = DGGS.address_to_tuple(addr)
        idx = DGGS.n2i[ch]

        x = 0
        y = 0

        for v in cells:
            assert v >= 0 and v <= 8
            x = x*3 + (v % 3)
            y = y*3 + (v // 3)

        return idx, x, y, len(cells)

    def __init__(self):
        self._rh = pyproj.Proj(a=1, **DGGS.proj_opts)
        self._s = DGGS._compute_norm_factor(self._rh)
        self._rhm = pyproj.Proj(**DGGS.proj_opts)
        self._sm = DGGS._compute_norm_factor(self._rhm)

    def mk_norm(self, idx, scale_level, norm_factor=None):
        if norm_factor is None:
            norm_factor = self._s

        OFFSET = ((0, 0), (0, 2), None, None,
                  (0, 1), (1, 1), (2, 1), (3, 1))

        scale = 3**scale_level
        s_x = 1.0/scale
        s_y = -1.0/scale

        off_x, off_y = OFFSET[idx]
        off_x -= 2
        off_y -= 1.5
        off_y += 1

        denorm = 1.0/norm_factor
        s_x *= denorm
        s_y *= denorm
        off_x *= denorm
        off_y *= denorm

        # y transform:
        # 1. Change direction: scale - y
        # 2. Unscale: 1/scale
        # 3. Offset based on top-level cell position

        def norm(x, y):
            return x*s_x + off_x, y*s_y + off_y

        return norm

    def to_ixy(self, lx, ly, scale_level):
        ss = 3**scale_level

        x, y = self._rh(lx, ly)

        x *= self._s
        y *= self._s
        # x in [-2,2], y in [-1.5, 1.5]

        x += 2
        y += 1.5
        # x in [0,4], y in [0, 3]

        idx = (y.astype('uint32') << 2) + x.astype('uint32')

        x = (ss*(x - (idx % 4))).astype('uint32')
        y = (ss*(1 + (idx // 4)) - ss*y).astype('uint32')

        idx = (idx & 0b111) | (idx >> 3)  # remap 8 -> 1

        return idx, x, y

    def to_address(self, lx, ly, scale_level):
        idx, x, y = self.to_ixy(lx, ly, scale_level)

        pad_bits = (15 - scale_level)*4

        # fill header first
        v = (idx | 0b1000).astype('uint64') << 60
        # fill padding bits
        v |= (0xFFFFFFFFFFFFFFFF >> (64-pad_bits))

        cell = np.empty_like(v)

        for i in range(scale_level):
            cell[:] = ((x % 3) + 3*(y % 3))
            cell <<= pad_bits
            v |= cell

            pad_bits += 4
            x = x//3
            y = y//3

        return v

    def pixel_coord_transform(self, addr, w=0, h=0):
        """
           Return method that can map pixel coord x,y to lon,lat
        """
        top_cell, x0, y0, scale_level = self.addr2ixys(addr)
        pix2rh = self.mk_norm(top_cell, scale_level)

        side = 3**scale_level

        maxW = side - x0
        maxH = side - y0

        # translate to pixel centre
        x0 += 0.5
        y0 += 0.5

        rh = self._rh

        def pix2lonlat(x, y, radians=False):
            x, y = pix2rh(x + x0, y + y0)
            lx, ly = rh(x, y, inverse=True, radians=radians)
            return lx, ly

        # TODO: make caching version

        return pix2lonlat, (maxW, maxH)

    def mk_warper(self, addr, w=0, h=0):
        tr, (maxW, maxH) = self.pixel_coord_transform(addr, w, h)

        w = maxW if w == 0 else w
        h = maxH if h == 0 else h

        u, v = np.meshgrid(range(w), range(h))
        u, v = [a.astype('float32') for a in tr(u, v)]

        u[u <= -180] = -180  # work-around for polar region numeric artifacts

        def warp(src, affine, src_crs=None):
            if src_crs is None:
                s_u, s_v = u, v
            else:
                # TODO: u,v + src_crs -> s_u, s_v
                raise NotImplementedError('TODO: CRS')

            src_x, src_y = apply_affine(~affine, s_u, s_v)

            return cv2.remap(src, src_x, src_y, cv2.INTER_CUBIC)

        return warp

    def mk_display_helper(self, south_square=0, north_square=0, meters=True):
        norm_factor = self._sm if meters else self._s

        def bounds(addr, shape, b=1e-6):
            top_cell, x0, y0, scale_level = self.addr2ixys(addr)
            pix2rh = self.mk_norm(top_cell, scale_level, norm_factor)

            h, w = shape[:2]
            p1 = pix2rh(x0+b, y0+b)
            p2 = pix2rh(x0+w-b, y0+h-b)

            return p1, p2

        def points2extent(p1, p2):
            xx = sorted([p1[0], p2[0]])
            yy = sorted([p1[1], p2[1]])
            return [xx[0], xx[1], yy[0], yy[1]]

        def simple(addr, im):
            p1, p2 = bounds(addr, im.shape)
            return im, points2extent(p1, p2)

        if south_square == 0 and north_square == 0:
            return simple

        RR = {
            0: np.eye(2),
            90: np.asarray([[0, -1],
                            [1, +0]], dtype='float32'),

            270: np.asarray([[+0, 1],
                             [-1, 0]], dtype='float32'),

            180: np.asarray([[-1,  0],
                             [+0, -1]], dtype='float32'),
        }

        def mk_correction(pole, square):
            if square == 0:
                return None

            side = 1.0/norm_factor

            angles = dict(N=[0, 90, 180, 270],
                          S=[0, 270, 180, 90])

            t0 = dict(N=(-1.5,  1),
                      S=(-1.5, -1))[pole]

            t0 = np.vstack(t0)
            t1 = np.vstack([square, 0])

            a = angles[pole][square]
            R = RR[a]

            t = t1 + t0 - np.dot(R, t0)

            return a, R, t*side

        corrections = dict(
            N=mk_correction('N', north_square),
            S=mk_correction('S', south_square))

        def rot(im, a):
            if a == 0:
                return im
            return cv2.rotate(im, {90: cv2.ROTATE_90_CLOCKWISE,
                                   180: cv2.ROTATE_180,
                                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a])

        def with_rot(addr, im):
            cc = corrections.get(addr[0])

            if cc is None:
                return simple(addr, im)

            a, R, t = cc
            p1, p2 = [np.dot(R, np.vstack(p)) + t for p in bounds(addr, im.shape)]

            return rot(im, a), points2extent(p1.ravel(), p2.ravel())

        return with_rot
