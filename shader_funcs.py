import taichi as ti
import math

# %% type shortcuts

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
vec5 = ti.types.vector(5, ti.f32)
vec6 = ti.types.vector(6, ti.f32)
vec7 = ti.types.vector(7, ti.f32)
vec8 = ti.types.vector(8, ti.f32)
vec9 = ti.types.vector(9, ti.f32)
vec10 = ti.types.vector(10, ti.f32)
vec11 = ti.types.vector(11, ti.f32)
vec12 = ti.types.vector(12, ti.f32)
vec13 = ti.types.vector(13, ti.f32)
vec14 = ti.types.vector(14, ti.f32)
vec15 = ti.types.vector(15, ti.f32)
vec16 = ti.types.vector(16, ti.f32)

mat2 = ti.types.matrix(2, 2, ti.f32)
mat3 = ti.types.matrix(3, 3, ti.f32)
mat4 = ti.types.matrix(4, 4, ti.f32)


tmpl = ti.template()
# %% constants

twopi = 2 * math.pi
pi180 = math.pi / 180.

# %% shader language functions


@ti.func
def length(p):
    """
    :param p: vector
    :return ti.sqrt(p.dot(p)): length of the vector p
    """
    return ti.sqrt(p.dot(p))


@ti.func
def normalize(p):
    """
    :param p: vector
    :return normalised vector p
    """
    n = p.norm()
    return p / (n if n != 0. else 1.)


@ti.func
def mix(x, y, a):
    """
    :param x: vector
    :param y: vector
    :param a: coefficient of mixing
    :return x * (1. - a) + y * a: resulting mixing vector
    """
    return x * (1. - a) + y * a


@ti.func
def dot(p, q):
    """
    :param p: vector
    :param q: vector
    :return p.dot(q): scalar multiplication of p and q
    """
    return p.dot(q)


@ti.func
def dot2(p):
    """
    :param p: vector
    :return p.dot(p): scalar multiplication of p and p
    """
    return p.dot(p)


@ti.func
def cross(x, y):
    """
    :param x: vector
    :param y: vector
    :return: a vector perpendicular to both x and y
    """
    return vec3(x[1] * y[2] - y[1] * x[2],
                x[2] * y[0] - y[2] * x[0],
                x[0] * y[1] - y[0] * x[1])


@ti.func
def reflect(rd, n):  # rd: vec3, n: vec3
    # https: // www.khronos.org / registry / OpenGL - Refpages / gl4 / html / reflect.xhtml
    """reflect incident vector rd using surface normal n"""
    return rd - 2.0 * dot(n, rd) * n


@ti.func
def deg2rad(a):
    """
    :param a: angle in degrees
    :return: angle `a` in radians
    """
    return a * pi180


@ti.func
def rot(a):   #rotation matrix
    """
    :param a: angle of the rotation in rad
    :return mat2([[c, -s], [s, c]]): matrix of the rotation in a 2-dim space
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return mat2([[c, -s], [s, c]])


@ti.func
def rot_y(a):
    """
    :param a: angle of the rotation relatively to axis y in rad
    :return: matrix of the rotation
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return mat3([[c, 0, -s],
                 [0, 1,  0],
                 [s, 0,  c]])


@ti.func
def rot_x(a):
    """
    :param a: angle of the rotation relatively to axis x in rad
    :return: matrix of the rotation
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return mat3([[1, 0,  0],
                 [0, c, -s],
                 [0, s,  c]])


@ti.func
def rot_z(a):
    """
    :param a: angle of the rotation relatively to axis z in rad
    :return: matrix of the rotation
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return mat3([[c, -s, 0],
                 [s,  c, 0],
                 [0,  0, 1]])

@ti.func
def sign(x: ti.f32):
    """
    :param x: ti.f32: a number
    :return: sign of x
    """
    return 1. if x > 0. else -1. if x < 0. else 0.


@ti.func
def signv(x: tmpl):
    """
    :param x: set of numbers
    :return: r: set of signs of x
    """
    r = ti.Vector(x.shape[0], x.dtype)
    for i in ti.static(range(x.shape[0])):
        r[i] = sign(x[i])
    return r


@ti.func
def clamp(x, low, high):
    """
    :param x: a number
    :param low: a low border of diapason
    :param high: a high border of diapason
    :return: x if low < x < high or one of the borders
    """
    return ti.max(ti.min(x, high), low)


@ti.func
def fract(x):
    """ returns fractional part of the inserted number
    :param
     x: inserted number
    :returns
     x - ti.floor(x): fractional part of the inserted number
    """
    return x - ti.floor(x)


@ti.func
def step(edge, x):
    """
    :param edge: a number
    :param x: a number
    :return: 0. if x < edge else 1.
    """
    return 0. if x < edge else 1.


@ti.func
def smoothstep(edge0, edge1, x):
    """
    :param edge0: a number
    :param edge1: a number
    :param x: a number
    :return: 0. if x < edge else 1.
    """
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def smoothmin(a, b, k):
    h = max(k - abs(a - b), 0.) / k
    return min(a, b) - h * h * k * (1./4.)


@ti.func
def smoothmax(a, b, k):
    return smoothmin(a, b, -k)


@ti.func
def smoothmin3(a, b, k):
    h = max(k - abs(a - b), 0.) / k
    return min(a, b) - h * h * h * k * (1./6.)


@ti.func
def skewsin(x, t):
    return ti.atan2(t * ti.sin(x), (1. - t * ti.cos(x))) / t

#%% PRNG

@ti.func
def random2():
    return vec2(ti.random(ti.f32), ti.random(ti.f32))


@ti.func
def hash1(n):
    return fract(ti.sin(n) * 43758.5453)


@ti.func
def hash21(p):
    q = fract(p * vec2(123.34, 345.56))
    q += dot(q, q + 34.23)
    return fract(q.x * q.y)

@ti.func
def hash31(p):
    q = fract(p * vec3(123.34, 345.56, 567.78))
    q += dot(q, q + 34.23)
    return fract(q.x * q.y * q.z)


@ti.func
def hash22(p):
    x = hash21(p)
    y = hash21(p + x)
    return vec2(x, y)


@ti.func
def hash33(p):
    x = hash31(p)
    y = hash31(p + x)
    z = hash31(p + x + y)
    return vec3(x, y, z)


# https://www.shadertoy.com/view/ll2GD3
@ti.func
def pal(t, a, b, c, d):
    return a + b * ti.cos(twopi * (c * t + d))

#%% SDF 2D


@ti.func
def sd_circle(p, r):  # == sd_sphere
    return p.norm() - r


@ti.func
def sd_segment(p, a, b):  # same for 3D
    pa = p - a
    ba = b - a
    h = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0)
    return (pa - ba * h).norm()


@ti.func
def sd_box(p, b):  # same for 3D
    d = abs(p) - b
    return max(d, 0.).norm() + min(d.max(), 0.0)

@ti.func
def sd_octahedron(p, s):
    p = abs(p)
    m = p.x + p.y + p.z - s
    q = vec3(0.)
    if 3.0 * p.x < m:
        q = p.xyz
    else:
        if 3.0 * p.y < m:
            q = p.yzx
        else:
            if 3.0 * p.z < m:
                q = p.zxy
            else:
                return m * 0.57735027

    k = clamp(0.5 * (q.z - q.y + s), 0.0, s)
    return length(vec3(q.x, q.y - s + k, q.z - k))

@ti.func
def sd_roundbox(p, b, r):
    rr = vec2(r[0], r[1]) if p[0] > 0. else vec2(r[2], r[3])
    rr[0] = rr[0] if p.y > 0. else rr[1]
    q = abs(p) - b + rr[0]
    return min(max(q[0], q[1]), 0.) + max(q, 0.0).norm() - rr[0]


@ti.func
def sd_trapezoid(p, r1, r2, he):
    k1 = vec2(r2, he)
    k2 = vec2(r2 - r1, 2. * he)
    pp = vec2(abs(p[0]), p[1])
    ca = vec2(pp[0] - min(pp[0], r1 if pp[1] < 0. else r2), abs(pp[1]) - he)
    cb = pp - k1 + k2 * clamp(dot(k1 - pp, k2) / dot2(k2), 0., 1.)
    s = -1. if cb[0] < 0. and ca[1] < 0. else 1.
    return s * ti.sqrt(min(dot2(ca), dot2(cb)))

#%% SDF 3D
# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm


@ti.func
def length2(p):
    return length(p)


@ti.func
def length6(p):
    q = p * p * p
    q *= q
    return (q.x + q.y + q.z)**(1./6.)


@ti.func
def length8(p):
    q = p * p
    q *= q
    q *= q
    return (q.x + q.y + q.z)**(1./8.)


@ti.func
def ndot(a, b):
    return a.x*b.x - a.y*b.y


@ti.func
def sd_sphere(p, r):
    # same as sd_circle
    return length(p) - r


@ti.func
def sd_torus(p, r):  # p: vec3, t: vec2
    q = vec2(length(vec2(p.x, p.z)) - r.x, p.y)
    return length(q) - r.y


@ti.func
def sd_cylinder(p, c):  # p: vec3, c: vec3
    pxz = vec2(p.x, p.z)
    cxy = vec2(c.x, c.y)
    return length(pxz - cxy) - c.z


@ti.func
def sd_cappedcylinder(p, h, r):  # p: vec3, h: ti.f32, r: ti.f32
    pxz = vec2(p.x, p.z)
    d = abs(vec2(length(pxz), p.y)) - vec2(h, r)
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@ti.func
def sd_cone(p, c, h):  # p: vec3, c: vec2, h: ti.f32
    """
    c is the sin / cos of the angle, h is height
    Alternatively pass q instead of(c, h), which is the
    point at the base in 2D
    """
    q = h * vec2(c.x / c.y, -1.0)
    w = vec2(length(p.xz), p.y)
    a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)
    b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0)
    k = sign(q.y)
    d = min(dot(a, a), dot(b, b))
    s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y))
    return ti.sqrt(d) * sign(s)


@ti.func
def sd_ellipsoid(p, r):  # p: vec3, r: vec3
  pr = p / r
  k0 = length(pr)
  k1 = length(pr/r)
  return k0 * (k0 - 1.0) / k1


@ti.func
def op_rep(p, c):  # p: vec3, c: vec3
    c_half = 0.5 * c
    return (p + c_half) % c - c_half


@ti.func
def op_replim(p, c, l):  # p: vec3, c: ti.f32, l: vec3
    return p - c * clamp(ti.round(p / c), -l, l)


@ti.func
def op_cheapbend(p, k):  # p: vec3, k: ti.f32
    alpha = k * p.x
    m = rot(alpha)
    q = m @ p.xy
    return vec3(q.x, q.y, p.z)

#%%

@ti.func
def lookat_raydir(uv, p, l, z):  # uv: vec2, p: vec3, l: vec3, z: ti.f32
    """
    :param uv: pixel coordinates
    :param p: camera position?
    :param l: camera look_at?
    :param z: scale?
    :return:
    """
    f = normalize(l - p)
    r = normalize(cross(vec3(0., 1., 0.), f))
    u = cross(f, r)
    c = f * z
    i = c + uv.x * r + uv.y * u
    return normalize(i)


@ti.func
def argmin(v):
    m = v[0]
    j = 0
    for i in ti.static(range(1, len(v))):
        if v[i] < m:
            j = i
            m = v[i]
    return m, j