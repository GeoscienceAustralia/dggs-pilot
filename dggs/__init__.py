import array
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

    i2n_dict = {v: k for k, v in n2i.items()}

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
    def _round_down(v, scale_level):
        s = 3**scale_level
        return v - (v % s)

    @staticmethod
    def _round_up(v, scale_level):
        s = 3**scale_level
        if v % s == 0:
            return v
        return v + (s - (v % s))

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

    @staticmethod
    def ixys2addr(idx, x, y, scale_level):
        code = DGGS.i2n_dict.get(idx)

        if code is None:
            raise ValueError('Incorrect idx')

        s = ''
        for i in range(scale_level):
            cell = ((x % 3)) + 3*(y % 3)
            s = str(cell) + s

            x = x//3
            y = y//3

        if x > 0 or y > 0:
            raise ValueError('Wrong x,y -- too large')

        return code + s

    class Address(object):
        def __init__(self, val):
            if isinstance(val, str):
                addr = val
                idx, x, y, scale = DGGS.addr2ixys(val)
            elif isinstance(val, tuple):
                addr = DGGS.ixys2addr(*val)
                idx, x, y, scale = val

            self.addr = addr
            self._ixys = (idx, x, y, scale)

        @property
        def scale(self):
            return self._ixys[3]

        @property
        def x(self):
            return self._ixys[1]

        @property
        def y(self):
            return self._ixys[2]

        @property
        def xy(self):
            return self._ixys[1:3]

        @property
        def code(self):
            return self.addr[0]

        def __iter__(self):
            return iter(self._ixys)

        def __hash__(self):
            return hash(self.addr)

        def __eq__(self, o):
            return o.addr == self.addr

        def __add__(self, offset):
            dx, dy = offset
            idx, x, y, scale = self._ixys
            x += dx
            y += dy

            max_valid = (3**scale) - 1
            if x < 0 or y < 0 or x > max_valid or y > max_valid:
                raise OverflowError("Not a valid address after offset")

            return DGGS.Address((idx, x, y, scale))

        def __sub__(self, a):
            """Given two addresses at the same level and in the same top-level cell,
               computes distance in pixel along x and y dimensions.

               a - b => (dx, dy) | None if undefined
            """

            if isinstance(a, tuple):
                return self.__add__((-a[0], -a[1]))

            if isinstance(a, str):
                return self.__sub__(DGGS.Address(a))

            if isinstance(a, DGGS.Address):
                x1, y1 = self._ixys[1:3]
                x2, y2 = a._ixys[1:3]
                return (x1 - x2, y1 - y2)

            raise ValueError('Accept (x,y) or other address only')

        def round_by(self, levels_up):
            if levels_up <= 0:
                return self

            x, y = (DGGS._round_down(v, levels_up) for v in self.xy)
            idx = self._ixys[0]
            return DGGS.Address((idx, x, y, self.scale))

        def round_to(self, scale_level):
            return self.round_by(self.scale - scale_level)

        def scale_up(self, num_levels):
            if num_levels == 0:
                return self

            if num_levels > self.scale:
                return DGGS.Address(self.addr[0])
            return DGGS.Address(self.addr[:-num_levels])

        def scale_down(self, num_levels, mode='tl'):
            c = dict(tl='0',
                     tr='2',
                     c='4',
                     bl='6',
                     br='8')[mode]

            return DGGS.Address(self.addr + (c*num_levels))

        def __str__(self):
            return self.addr

        def __repr__(self):
            return self.addr

    class ROI(object):
        def __init__(self, addr, w=1, h=1):
            if isinstance(addr, str):
                addr = DGGS.Address(addr)
            self.addr = addr
            self.w = w
            self.h = h

        def __iter__(self):
            yield self.addr
            yield self.w
            yield self.h

        def __eq__(self, o):
            return tuple(self) == tuple(o)

        def __str__(self):
            return f'{self.addr} {self.w}x{self.h}'

        def __repr__(self):
            return self.__str__()

        @property
        def shape(self):
            return (self.h, self.w)

        @property
        def scale(self):
            return self.addr.scale

        def align_by(self, levels_up):
            """Expand region so that address and size align to given scale
            Returns expanded region
            """
            if levels_up <= 0:
                return self

            addr, w, h = self
            addr_new = addr.round_by(levels_up)
            dw, dh = addr - addr_new

            w = DGGS._round_up(w + dw, levels_up)
            h = DGGS._round_up(h + dh, levels_up)

            return DGGS.ROI(addr_new, w, h)

        def align_to(self, scale_level):
            return self.align_by(self.addr.scale - scale_level)

        def scale_up(self, num_levels):
            addr, w, h = self.align_by(num_levels)
            s = 3**num_levels
            w, h = [max(1, v//s) for v in [w, h]]
            return DGGS.ROI(addr.scale_up(num_levels), w, h)

        def scale_down(self, num_levels):
            addr, w, h = self
            s = 3**num_levels
            return DGGS.ROI(addr.scale_down(num_levels), w*s, h*s)

    @staticmethod
    def roi_slice(outter, inner):
        assert inner.scale <= outter.scale
        if inner.scale < outter.scale:
            inner = inner.scale_down(outter.scale - inner.scale)

        oh, ow = outter.shape
        h, w = inner.shape
        dx, dy = inner.addr - outter.addr

        if dx < 0 or dy < 0 or (dy+h) > oh or (dx+w) > ow:
            raise ValueError('Inner roi is not inside outter one')

        return (slice(dy, dy+h), slice(dx, dx+w))

    class Image:
        def __init__(self, im, addr):
            if isinstance(addr, DGGS.ROI):
                assert addr.shape == im.shape
                addr = addr.addr

            self._data = im
            h, w = im.shape[:2]
            self._roi = DGGS.ROI(addr, w, h)

        @property
        def value(self):
            return self._data

        @property
        def roi(self):
            return self._roi

        @property
        def addr(self):
            return self._roi.addr

        @property
        def shape(self):
            return self._data.shape

        def __repr__(self):
            return 'Image @ ' + repr(self._roi)

        def _to_slice(self, obj):
            if isinstance(obj, DGGS.ROI):
                return DGGS.roi_slice(self._roi, obj)
            if isinstance(obj, (DGGS.Address, str)):
                return DGGS.roi_slice(self._roi, DGGS.ROI(obj, 1, 1))
            return obj

        def __getitem__(self, roi):
            return self._data[self._to_slice(roi)]

        def __setitem__(self, roi, val):
            self._data.__setitem__(self._to_slice(roi), val)

    @staticmethod
    def roi_from_points(aa, scale=None):
        if scale is None:
            scale = max(a.scale for a in aa)

        def update_bounds(bounds, addr):
            x, y = addr.xy
            if bounds is None:
                return (x, x, y, y)

            xmin, xmax, ymin, ymax = bounds
            return (min(x, xmin), max(x, xmax), min(y, ymin), max(y, ymax))

        def normed_points(aa, scale):
            for a in aa:
                if a.scale < scale:
                    yield a.scale_down(scale - a.scale, mode='tl')
                    yield a.scale_down(scale - a.scale, mode='br')
                else:
                    assert a.scale == scale
                    yield a

        _i = None
        bounds = None
        for a in normed_points(aa, scale):
            if _i is None:
                _i, *_ = a._ixys

            if _i != a._ixys[0]:
                raise ValueError('Currently assumes that all points are in the same top level cell')

            bounds = update_bounds(bounds, a)

        xmin, xmax, ymin, ymax = bounds

        addr = DGGS.Address((_i, xmin, ymin, scale))
        w = xmax - xmin + 1
        h = ymax - ymin + 1

        return DGGS.ROI(addr, w, h)

    @staticmethod
    def crop(src, src_roi, crop_roi):
        if src_roi == crop_roi:
            return src, src_roi

        sh, sw = src_roi.shape
        h, w = crop_roi.shape
        dx, dy = crop_roi.addr - src_roi.addr

        if dx < 0 or dy < 0 or (dx+w) > sw or (dy+h) > sh:
            raise ValueError('Crop region is not inside of the source region')

        return src[dy:dy+h, dx:dx+w], crop_roi

    @staticmethod
    def pad(src, src_roi, dst_roi, nodata=None):
        assert src.shape == src_roi.shape

        if src_roi == dst_roi:
            return src

        # TODO: support n-dimensional data, n>2
        out = np.empty(dst_roi.shape, dtype=src.dtype)
        h, w = src.shape
        (dx, dy) = src_roi.addr - dst_roi.addr

        if nodata is not None:
            # TODO: efficiency -- only fill boundary instead
            out[:] = nodata

        out[dy:dy+h, dx:dx+w] = src
        return out

    @staticmethod
    def expand(im, roi):
        from .tools import expand_3x3
        return expand_3x3(im), roi.scale_down(1)

    @staticmethod
    def expand_to_roi(src, src_roi, dst_roi):
        assert dst_roi.scale >= src_roi.scale

        if dst_roi.scale == src_roi.scale:
            return DGGS.crop(src, src_roi, dst_roi)

        n = dst_roi.scale - src_roi.scale
        for i in range(n):
            src, src_roi = DGGS.expand(src, src_roi)
            src, src_roi = DGGS.crop(src, src_roi, dst_roi.scale_up(n-i-1))

        return src, src_roi

    @staticmethod
    def scale_op_sum(im, roi, nodata=None, dtype=None, tight=False):
        from .tools import nodata_to_num, sum3x3
        roi_ = roi.align_by(1 if tight else 2)
        im = DGGS.pad(nodata_to_num(im, nodata), roi, roi_, nodata=0)
        return sum3x3(im, dtype=dtype), roi_.scale_up(1)

    @staticmethod
    def scale_op_and(im, roi, tight=False):
        from .tools import logical_and_3x3
        roi_ = roi.align_by(1 if tight else 2)
        im = DGGS.pad(im, roi, roi_, nodata=False)
        return logical_and_3x3(im), roi_.scale_up(1)

    @staticmethod
    def scale_op_or(im, roi, tight=False):
        from .tools import logical_or_3x3
        roi_ = roi.align_by(1 if tight else 2)
        im = DGGS.pad(im, roi, roi_, nodata=False)
        return logical_or_3x3(im), roi_.scale_up(1)

    def __init__(self):
        self._rhm = pyproj.Proj(**DGGS.proj_opts)
        self._sm = DGGS._compute_norm_factor(self._rhm)
        self._prj_lonlat = pyproj.Proj(self._rhm.to_latlong().srs.decode())
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

        def norm_array(a):
            if isinstance(a, array.array):
                return np.asarray(a)
            return a

        x = norm_array(x)
        y = norm_array(y)

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

    def to_ixy(self, scale_level, x=None, y=None, crs=None):
        to_native = self.to_native(crs)

        def convert(x, y):
            x, y = to_native(x, y)
            return self._rh_to_ixy(x, y, scale_level, self._sm)

        if x is None:
            return convert

        return convert(x, y)

    def to_address(self, scale_level, lx, ly, native=False, crs=None):
        if native:
            idx, x, y = self._rh_to_ixy(lx, ly, scale_level, self._sm)
        else:
            idx, x, y = self.to_ixy(scale_level, lx, ly, crs=crs)

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

    def _as_proj(self, crs):
        if isinstance(crs, pyproj.Proj):
            return crs

        if isinstance(crs, (str, dict)):
            if crs == 'native':
                return self._rhm

            return pyproj.Proj(crs)

        if crs is None:
            return None

        if hasattr(crs, '__getitem__'):
            return pyproj.Proj(crs)

        raise ValueError('Not sure how to convert to pyproj.Proj')

    def to_lonlat(self, crs, x, y):
        """ Convert to lonlat on the right kind of datum for rHealPix.
        """
        src_prj = self._as_proj(crs)

        def convert(x, y, z=None):
            return pyproj.transform(src_prj, self._prj_lonlat, x, y)

        if x is None:
            return convert

        return convert(x, y)

    def to_native(self, crs=None, x=None, y=None):
        src_prj = self._prj_lonlat if crs is None else self._as_proj(crs)

        def convert(x, y, z=None):
            return pyproj.transform(src_prj, self._rhm, x, y, z)

        if x is None:
            return convert

        return convert(x, y)

    def compute_overlap(self, scale_level, border_x, border_y, crs=None, tol=1e-4):
        """Returns a list of ROI objects (address,w,h) that fully enclose a
        shape specified by a boundary (border_x, border_y, crs)

        If crs is not supplied then border_x, border_y is assumed to be in
        lonlat on WGS84.

        """
        if crs is None:
            lx, ly = border_x, border_y
        else:
            lx, ly = self.to_lonlat(crs, border_x, border_y)

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
                addr = self.to_address(scale_level, x1, y2, native=True)  # Use top-left corner for address
                dx = math.ceil((x2-x1) * to_pix)
                dy = math.ceil((y2-y1) * to_pix)
                out.append(DGGS.ROI(addr, dx, dy))

        return out

    def roi_from_geo(self, geo, scale, align_by=None):
        """
        geo.{crs, affine, shape} -> [ROI]
        """
        from .tools import geo_boundary
        x, y = geo_boundary(geo.affine, geo.shape)
        rr = self.compute_overlap(scale, x, y, geo.crs)

        if align_by is not None:
            return [roi.align_by(align_by) for roi in rr]

        return rr

    def pixel_coord_transform(self, addr, w=0, h=0, dst_crs=None, no_offset=False, native=False):
        """
           Return method that can map pixel coord x,y to coordinate system defined by dst_crs
        """
        dst_proj = self._as_proj(dst_crs)

        assert isinstance(addr, (str, tuple, DGGS.Address, DGGS.ROI))

        if isinstance(addr, DGGS.ROI):
            addr, w, h = addr

        if isinstance(addr, str):
            top_cell, x0, y0, scale_level = self.addr2ixys(addr)
        elif isinstance(addr, (tuple, DGGS.Address)):
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

    def mk_warper(self, roi, src_crs=None):
        src_proj = self._as_proj(src_crs)

        tr, (maxW, maxH) = self.pixel_coord_transform(roi, dst_crs=src_proj)

        h, w = roi.shape

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

            # Depending on pixel origin notations we might have to
            # subtract 0.5 from src_x, src_y, or better adjust affine matrix
            # for equivalent computation, I believe rasterio (source of Affine
            # matrix) places 0,0 at the top left corner of the top left pixel,
            # while OpenCV (remap) might be using pixel center of the top-left
            # pixel as an origin. However I couldn't find any reference in the
            # openCV docs that states categorically what the coordinates of
            # pixel corners are.
            A = ~affine

            # TODO: make this configurable
            if True:  # Change 0,0 from top-left corner of the pixel to pixel center
                A = A.translation(-0.5, -0.5)*A

            src_x, src_y = apply_affine(A, u, v)
            return cv2.remap(src, src_x, src_y, human2cv[inter], borderValue=nodata)

        return warp

    def xy_from_roi(self, roi):
        """
        Return x coordinates of pixel columns and y coordinates of pixel rows

        Coordinates are in rHealPix for centers of pixels
        """
        addr, w, h = roi
        tr, *_ = self.pixel_coord_transform(addr, native=True)

        x, _ = tr(np.arange(w), np.zeros(w))
        _, y = tr(np.zeros(h), np.arange(h))

        return x, y

    def mk_display_helper(self, south_square=0, north_square=0):
        norm_factor = self._sm

        def shape(a):
            if isinstance(a, (tuple, list)):
                return a
            return a.shape

        def bounds(addr, shape, b=1e-6):
            if isinstance(addr, str):
                addr = DGGS.Address(addr)

            top_cell, x0, y0, scale_level = addr
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
            if isinstance(addr, str):
                addr = DGGS.Address(addr)

            cc = corrections.get(addr.code)

            if cc is None:
                return simple(addr, im)

            a, R, t = cc
            p1, p2 = [np.dot(R, np.vstack(p)) + t for p in bounds(addr, shape(im))]

            return rot(im, a), points2extent(p1.ravel(), p2.ravel())

        return with_rot


def mask_from_addresses(aa, roi=None):
    roi = DGGS.roi_from_points(aa) if roi is None else roi
    mm = DGGS.Image(np.zeros(roi.shape, dtype='bool'), roi)

    for a in aa:
        mm[a] = True

    return mm


def mask_to_addresses(im, dg=DGGS()):
    def offsets_to_txt(addr, xx, yy):
        return sorted(str(addr + (x, y)) for x, y in zip(xx, yy))

    def mask_lvl_diff(m, roi, m2, m2_roi):
        assert roi.scale == m2_roi.scale + 1
        m2, m2_roi = dg.expand(m2, m2_roi)
        m2, _ = dg.crop(m2, m2_roi, roi)
        return m*(~m2)

    m = im.value
    roi = im.roi

    out = []

    while m.any():
        m2, m2_roi = dg.scale_op_and(m, roi, tight=True)
        m_diff = mask_lvl_diff(m, roi, m2, m2_roi)
        y, x = np.where(m_diff)

        out = offsets_to_txt(roi.addr, x, y) + out

        m, roi = m2, m2_roi

    return out


def shape_to_mask(poly, crs, scale_level, dg=DGGS(), align=None):
    from shapely import ops
    from rasterio.features import rasterize

    def mk_to_pix_transform(roi, src_crs):
        to_native = dg.to_native(src_crs)
        tr, *_ = dg.pixel_coord_transform(roi, native=True, no_offset=True)

        def transform(x, y):
            x, y = to_native(x, y)
            return tr.inverse(x, y)

        return transform

    overlaps = dg.compute_overlap(scale_level, *poly.boundary.xy, crs=crs)

    if len(overlaps) > 1:
        raise NotImplemented("Error: shape crosses top level cell, this is currently not supported")
    elif len(overlaps) == 0:
        raise ValueError("Something went wrong: couldn't convert polygon to lonlat properly")

    roi, *_ = overlaps

    if align is not None:
        roi = roi.align_by(align)

    poly_pix = ops.transform(mk_to_pix_transform(roi, crs), poly)
    im = rasterize([poly_pix], out_shape=roi.shape).astype(np.bool)

    return dg.Image(im, roi.addr)
