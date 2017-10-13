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
        proj_string = ('+proj=rhealpix +wktext +no_defs +ellps=WGS84 +datum=WGS84'
                       ' +a=1 +lat_0=0 +south_square=0 +north_square=0')
        self._rh = pyproj.Proj(proj_string)
        self._s = DGGS._compute_norm_factor(self._rh)

    def mk_norm(self, idx, scale_level):
        OFFSET = ((0, 0), (0, 2), None, None,
                  (0, 1), (1, 1), (2, 1), (3, 1))

        scale = 3**scale_level
        s_x = 1.0/scale
        s_y = -1.0/scale

        off_x, off_y = OFFSET[idx]
        off_x -= 2
        off_y -= 1.5
        off_y += 1

        denorm = 1.0/self._s
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
