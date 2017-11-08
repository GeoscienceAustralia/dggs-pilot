import math
import shapely.geometry
import shapely.ops
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
                     ellps='WGS84',
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
        self.equatorial_thresh = self._rhm(0, 0.5/self._sm - 1e-6, inverse=True)[1]

    @property
    def top_level_extents(self):
        """ Returns dictionary of lon-lat extents for 6 top-level cells

        (left, right, bottom, top)
        """
        eqt = self.equatorial_thresh

        return dict(N=(-180, 180, +eqt,  +90),
                    S=(-180, 180,  -90, -eqt),
                    O=(-180, -90, -eqt, +eqt),
                    P=( -90,   0, -eqt, +eqt),  # noqa: E201
                    Q=(   0, +90, -eqt, +eqt),  # noqa: E201
                    R=(  90, 180, -eqt, +eqt))  # noqa: E201

    def mk_norm(self, idx, scale_level, norm_factor=None):
        if isinstance(idx, str):
            idx = self.n2i[idx]

        if norm_factor is None:
            norm_factor = self._sm

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
            """ "pixel coords" => meters in rHealPix
            """
            return x*s_x + off_x, y*s_y + off_y

        def norm_inverse(x, y):
            """ meters in rHealPix => "pixel coords"
            """
            return (x - off_x)/s_x, (y - off_y)/s_y

        norm.inverse = norm_inverse
        return norm

    @staticmethod
    def _rh_to_ixy(x, y, scale_level, rh_scale):

        ss = 3**scale_level

        x *= rh_scale
        y *= rh_scale
        # x in [-2,2], y in [-1.5, 1.5]

        x += 2
        y += 1.5
        # x in [0,4], y in [0, 3]

        if isinstance(x, (int, float)):
            idx = (int(y) << 2) + int(x)
            x = int(ss*(x - (idx % 4)))
            y = int(ss*(1 + (idx // 4)) - ss*y)
        else:
            idx = (y.astype('uint32') << 2) + x.astype('uint32')
            x = (ss*(x - (idx % 4))).astype('uint32')
            y = (ss*(1 + (idx // 4)) - ss*y).astype('uint32')

        idx = (idx & 0b111) | (idx >> 3)  # remap 8 -> 1

        return idx, x, y

    def to_ixy(self, lx, ly, scale_level):
        x, y = self._rhm(lx, ly)
        return self._rh_to_ixy(x, y, scale_level, self._sm)

    def to_address(self, lx, ly, scale_level, native=False):
        if native:
            idx, x, y = self._rh_to_ixy(lx, ly, scale_level, self._sm)
        else:
            idx, x, y = self.to_ixy(lx, ly, scale_level)

        pad_bits = (15 - scale_level)*4

        # fill header first
        if isinstance(idx, int):
            array_mode = False
            v = (idx | 0b1000) << 60
            cell = 0
        else:
            array_mode = True
            v = (idx | 0b1000).astype('uint64') << 60
            cell = np.empty_like(v)

        # fill padding bits
        v |= (0xFFFFFFFFFFFFFFFF >> (64-pad_bits))

        for i in range(scale_level):
            if array_mode:
                cell[:] = ((x % 3) + 3*(y % 3))
            else:
                cell = ((x % 3) + 3*(y % 3))

            cell <<= pad_bits
            v |= cell

            pad_bits += 4
            x = x//3
            y = y//3

        if not array_mode:
            return DGGS.i2n[idx] + '{:16x}'.format(v)[1:scale_level+1]

        return v

    def compute_overlap(self, scale_level, border_x, border_y, crs=None, tol=1e-4):
        """Returns a list of triplets (address, width, height) that fully enclose a
        shape specified by a boundary (border_x, border_y, crs)

        If crs is not supplied then border_x, border_y is assumed to be in
        lonlat on WGS84.

        """
        if crs is None:
            lx, ly = border_x, border_y
        else:
            # TODO: deal with possible geoid differences, this should really be
            #       using transform function going into lonlat on WGS84
            prj = pyproj.Proj(crs)
            lx, ly = prj(border_x, border_y, inverse=True)

        src_poly = shapely.geometry.Polygon(np.vstack([lx, ly]).T)

        to_pix = self._sm*(3**scale_level)

        out = []

        for c, box in self.top_level_extents.items():
            xmin, xmax, ymin, ymax = box
            # TODO: don't like this use of tolerances. Ideally we should have a
            # method that projects into the reference frame of the chosen top
            # level cell we can then clamp in there.
            cell = shapely.geometry.box(xmin+tol, ymin+tol, xmax-tol, ymax-tol)
            overlap = cell.intersection(src_poly)

            if not overlap.is_empty:
                ov_ = shapely.ops.transform(lambda x, y: self._rhm(x, y), overlap)

                rh_box = ov_.bounds
                x1, y1, x2, y2 = rh_box
                addr = self.to_address(x1, y2, scale_level, native=True)  # Use top-left corner for address
                dx = math.ceil((x2-x1) * to_pix)
                dy = math.ceil((y2-y1) * to_pix)
                out.append((addr, dx, dy))

        return out

    def pixel_coord_transform(self, addr, w=0, h=0, dst_proj=None, no_offset=False, native=False):
        """
           Return method that can map pixel coord x,y to lon,lat
        """
        assert isinstance(addr, (str, tuple))

        if isinstance(dst_proj, (str, dict)):
            dst_proj = pyproj.Proj(dst_proj)

        if isinstance(addr, str):
            top_cell, x0, y0, scale_level = self.addr2ixys(addr)
        elif isinstance(addr, tuple):
            top_cell, x0, y0, scale_level = addr

        rh = self._rhm
        pix2rh = self.mk_norm(top_cell, scale_level, self._sm)

        side = 3**scale_level

        maxW = side - x0
        maxH = side - y0

        if no_offset is False:
            # translate to pixel centre
            x0 += 0.5
            y0 += 0.5

        # TODO: make caching version
        def pix2lonlat(x, y, radians=False):
            x, y = pix2rh(x + x0, y + y0)
            lx, ly = rh(x, y, inverse=True, radians=radians)
            return lx, ly

        def pix2dst(x, y):
            x, y = pix2rh(x + x0, y + y0)
            lx, ly = pyproj.transform(rh, dst_proj, x, y)
            return lx, ly

        def pix2native(x, y):
            return pix2rh(x + x0, y + y0)

        def pix2native_inv(mx, my):
            x, y = pix2rh.inverse(mx, my)
            x -= x0
            y -= y0
            return x, y

        pix2native.inverse = pix2native_inv

        if native:
            transform = pix2native
        elif dst_proj:
            transform = pix2dst
        else:
            transform = pix2lonlat

        return transform, (maxW, maxH)

    def mk_warper(self, addr, w=0, h=0, src_proj=None, src_crs=None):
        if (src_proj is None) and (src_crs is not None):
            src_proj = pyproj.Proj(src_crs)

        tr, (maxW, maxH) = self.pixel_coord_transform(addr, w, h, dst_proj=src_proj)

        w = maxW if w == 0 else w
        h = maxH if h == 0 else h

        u, v = np.meshgrid(range(w), range(h))
        u, v = [a.astype('float32') for a in tr(u, v)]

        if src_proj is None:
            u[u <= -180] = -180  # work-around for polar region numeric artifacts

        human2cv = dict(linear=cv2.INTER_LINEAR,
                        nearest=cv2.INTER_NEAREST,
                        cubic=cv2.INTER_CUBIC,
                        area=cv2.INTER_AREA,
                        lanczos4=cv2.INTER_LANCZOS4)

        def warp(src, affine, nodata=0, inter=None):
            inter = inter if inter else 'linear'
            assert inter in human2cv
            if nodata is None:
                nodata = 0

            src_x, src_y = apply_affine(~affine, u, v)
            return cv2.remap(src, src_x, src_y, human2cv[inter], borderValue=nodata)

        return warp

    def mk_display_helper(self, south_square=0, north_square=0, meters=True):
        norm_factor = self._sm if meters else self._s

        def shape(a):
            if isinstance(a, (tuple, list)):
                return a
            return a.shape

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
            p1, p2 = bounds(addr, shape(im))
            return im, points2extent(p1, p2)

        if south_square == 0 and north_square == 0:
            return simple

        # Rotation is clockwise
        RR = {
            0: np.eye(2),
            270: np.asarray([[0, -1],
                             [1, +0]], dtype='float32'),

            90: np.asarray([[+0, 1],
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

            if isinstance(im, (tuple, list)):
                if a == 180:
                    return im
                return im[:2][::-1] + im[2:]  # Swap first 2 coords

            return cv2.rotate(im, {90: cv2.ROTATE_90_CLOCKWISE,
                                   180: cv2.ROTATE_180,
                                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a])

        def with_rot(addr, im):
            cc = corrections.get(addr[0])

            if cc is None:
                return simple(addr, im)

            a, R, t = cc
            p1, p2 = [np.dot(R, np.vstack(p)) + t for p in bounds(addr, shape(im))]

            return rot(im, a), points2extent(p1.ravel(), p2.ravel())

        return with_rot
