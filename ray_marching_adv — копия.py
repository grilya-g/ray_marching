import time
import numpy as np
import taichi as ti
from shader_funcs import *

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)

# %% Resolution and pixel buffer

INF = 10e7
asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
AA = 1
# style = 'toon'
style = 'standard'
NUM_BOUNCES = 2

# %% Fields (should always be before first call of any taichi scope kernels)

global_time = ti.field(dtype=ti.f32, shape=())
mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
mouse_btn = ti.field(dtype=ti.i32, shape=(2,))
mouse_btn_prev = ti.field(dtype=ti.i32, shape=(2,))
flags = ti.field(dtype=ti.i32, shape=(8,))
materials = ti.Vector.field(3, dtype=ti.f32, shape=(8,))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

# %%

materials.from_numpy(np.array([[1.0, 0.7, 0.0],
                               [0.0, 0.8, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.7, 1.0],
                               [0.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [0.5, 0.8, 0.9]  # background
                               ], dtype=np.float32))


# %%

@ti.func
def offset_uv(uv, offset):
    """
    :param uv: vec(3), coordinate of the pixel
    :param offset: vec(3), vertex of the shift
    :return: vec3, new coordinates
    """
    return ((uv * res[1] + 0.5 * vec2(res) + offset) - 0.5 * vec2(res)) / res[1]


@ti.func
def renderAAx4(ro, uv, rot_matr):
    """
    :param ro: vec3, vector of the origin of the ray
    :param uv: vec3, coordinate of the pixel
    :param rot_matr: matr3, matrix of  the rotation of the ray of view
    :return: vec(3), colour
    """
    e = vec4(0.125, -0.125, 0.375, -0.375)

    u_xz = offset_uv(uv, vec2(e.x, e.z))
    u_yw = offset_uv(uv, vec2(e.y, e.w))
    u_wx = offset_uv(uv, vec2(e.w, e.x))
    u_zy = offset_uv(uv, vec2(e.z, e.y))

    rd_xz = rot_matr @ normalize(vec3(u_xz.x, u_xz.y, 1.))
    rd_yw = rot_matr @ normalize(vec3(u_yw.x, u_yw.y, 1.))
    rd_wx = rot_matr @ normalize(vec3(u_wx.x, u_wx.y, 1.))
    rd_zy = rot_matr @ normalize(vec3(u_zy.x, u_zy.y, 1.))

    col_xz = render(ro, rd_xz)
    col_yw = render(ro, rd_yw)
    col_wx = render(ro, rd_wx)
    col_zy = render(ro, rd_zy)

    colAA = col_xz + col_zy + col_yw + col_wx

    return colAA / 4.0  # функция работает, но медленнее, чем рассмотренная на консультации





@ti.func
def sd_octahedron(p, s):  # 1   vec3 p, float s
    """
    sdf for octahedron
    :param p: vec3, coordinates of the center of octahedron
    :param s: float, scale
    :return: ti.f32,   sdf for octahedron
    """
    p = abs(p)
    m = p.x + p.y + p.z - s
    q = vec3(0.)
    if 3.0 * p.x < m:
        q = vec3(p.x, p.y, p.z)
    else:
        if 3.0 * p.y < m:
            q = vec3(p.y, p.z, p.x)
        else:
            if 3.0 * p.z < m:
                q = vec3(p.z, p.x, p.y)

    k = clamp(0.5 * (q.z - q.y + s), 0.0, s)
    return m * 0.57735027 if 3.0 * p.x >= m and 3.0 * p.y >= m and 3.0 * p.z >= m \
        else length(vec3(q.x, q.y - s + k, q.z - k))


@ti.func
def sd_hexagonalprism(p, h):  # 2   vec3 p, vec2 h
    """
    sdf for hexagonal prism
    :param p: vec3, plecement of the prism
    :param h: vec2, scale of the front part and width
    :return: sdf for hexagonal prism
    """
    k = vec3(-0.8660254, 0.5, 0.57735)
    p = abs(p)
    correction = 2.0 * min(dot(vec2(k.x, k.y), vec2(p.x, p.y)), 0.0) * vec2(k.x, k.y)
    p.x -= correction.x
    p.y -= correction.y
    d = vec2(
        length(vec2(p.x, p.y) - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
        p.z - h.y)
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@ti.func
def sd_solidangle(p, c, ra):  # 3  vec3 p, vec2 c, float ra
    """
    sdf for solid angle
    :param p: vec3, plecement of the prism
    :param c: vec2, sin / cos of the angle
    :param ra: ti.f32, radius
    :return: ti.f32, sdf for solid angle
    """
    # c is the sin/cos of the angle
    q = vec2(length(vec2(p.x, p.z)), p.y)
    l = length(q) - ra
    m = length(q - c * clamp(dot(q, c), 0.0, ra))
    return max(l, m * sign(c.y * q.x - c.x * q.y))


@ti.func
def sd_capsule(p, a, b, r):  # 4  #vec3 p, vec3 a, vec3 b, ti.f32 r
    """
    sdf for capsule
    :param p: vec3, coord of the center of the capsule
    :param a: vec3, coord of the one of the apexes
    :param b:vec3, coord of the another apex
    :param r: radius of the rounding
    :return: ti.f32, sdf for capsule
    """
    pa = p - a
    ba = b - a
    h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)
    return length(pa - ba * h) - r


@ti.func
def sd_boxframe(p, b, e):  # 5 # vec3 p, vec3 b, float e
    """
    sdf for box frame
    :param p: vec3, coord of the center of the box
    :param b: vec3, coord of the one of the apexes
    :param e: ti.f32, width of the frame
    :return: ti.f32, sdf for box frame
    """
    p = abs(p) - b
    q = abs(p + e) - e
    return min(min(length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
                   length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
               length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0))


# @ti.func
# def sd_deathstar(p2, ra, rb, d):  # 4 # vec3 p2, ti.f32 ra, ti.f32 rb, ti.f32 d
#     a = (ra * ra - rb * rb + d * d) / (2.0 * d)
#     b = ti.sqrt(max(ra * ra - a * a, 0.0))
#     p = vec2(p2.x, length(vec2(p2.y, p2.z)))
#
#     return length(p - vec2(a, b)) if (p.x * b - p.y * a > d * max(b - p.y, 0.0)) \
#         else max((length(p) - ra), -(length(p - vec2(d, 0)) - rb))
#
#
# @ti.func
# def sd_pyramid(p, h):  # 4 # p: vec3, h: ti.f32
#     m2 = h * h + 0.25
#     p.x = abs(p.x)
#     p.z = abs(p.z)
#     if p.z > p.x:
#         temp = p.z
#         p.z = p.x - 0.5
#         p.x = temp - 0.5
#
#     q = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y)
#
#     s = max(-q.x, 0.0)
#     t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0)
#
#     a = m2 * (q.x + s) * (q.x + s) + q.y * q.y
#     b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t)
#     d2 = 0.0
#     if min(q.y, -q.x * m2 - q.y * 0.5) > 0.0:
#         d2 = min(a, b)
#
#     return ti.sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y))


@ti.func
def texture(uv, s):
    """
    Draws texture
    :param uv: vec3, coord of pixel
    :param s: ti.f32, scale
    :return: vec3, color
    """
    d = 0.
    # if flags[0]:
    d = sd_box(uv, vec2(3.))
    # else:
    #     d = sd_circle(uv, 2.)
    col = vec3(0.)
    if d < 0:
        col = vec3(1.0, 0.7529411764705882, 0.796078431372549)
        col *= 0.8 + 0.2 * skewsin(s * d, 0.8)
        col *= 1.0 - ti.exp(-7.0 * abs(d))
    return col


@ti.func
def boxmap_texture(p, n, k, s):

    xcol = texture(vec2(p.y, p.z), s)
    ycol = texture(vec2(p.z, p.x), s)
    zcol = texture(vec2(p.x, p.y), s)
    w = abs(n) ** k
    return (xcol * w.x + ycol * w.y + zcol * w.z) / w.sum()


@ti.func
def rotate_cube(p):
    """
    Changes coords of the rotated cube
    :param p: vec3, coord of pixel
    :return: vec3, rotated coords
    """
    t = global_time[None]
    m = rot_x(0.1 * t)
    return m @ p


@ti.func
def translate_cube(p):
    """
    Transform coords of translated cube
    :param p: vec3, coords of pixel
    :return: ti.f32, translated coords
    """
    t = global_time[None]
    return p - vec3(ti.sin(0.2 * t), 0., 0.)


@ti.func
def dist_vector(p):
    """
    Contains all the primitives
    :param p: vec3, coords of pixel
    :return: vec6, distances to the objects
    """
    # d1 = sd_box(translate_cube(rotate_cube(p)), vec3(0.5))
    d0 = p.y + 1.5
    d1 = sd_hexagonalprism(translate_cube(rotate_cube(p - vec3(1., -0.5, 2.5))), vec2(0.5, 0.67))
    d2 = sd_boxframe(p - vec3(1.5, 0.5, 0.), vec3(0.6, 0.6, 0.6), 0.05)
    d3 = sd_capsule(p - vec3(-1.5, 0.5, -0.7), vec3(0.6, 0.6, 0.6), vec3(0.2, 0.2, 0.2), 0.5)
    d4 = sd_octahedron(p - vec3(0.5, 1.5, 0.7), 0.67)
    # d5 = sd_trapezoid(p - vec3(5., 1.5, -5.) / 3, 5., 5., 1.)
    #            0   1
    # return vec5(d0, d1, d2, d3, d4)
    return vec5(d0, d1, d2, d3, d4)


@ti.func
def dist(p):
    """
    Returns min distance to objects
    :param p: vec3, pixel coords
    :return: ti.f32, min distance
    """
    return dist_vector(p).min()


@ti.func
def dist_mat(p):
    """
    Returns index of the nearest object
    :param p: vec3, pixel coords
    :return: int, index of the nearest object
    """
    return argmin(dist_vector(p))


MAX_STEPS = 100
MAX_DIST = 200.
EPS = 1e-3


@ti.func
def raymarch(ro, rd):
    """
    Finds what object will intersect ray of looking
    :param ro: vec3, ray origin
    :param rd: vec3, ray direction
    :return: (ti.f32, int, vec3), min dist, index of the material, pixel coords
    """
    p = vec3(0.)
    d = 0.
    mat_i = 0
    for i in range(MAX_STEPS):
        p = ro + d * rd
        ds, mat_i = dist_mat(p)
        d += ds
        if ds < EPS:
            break
        if d > MAX_DIST:
            mat_i = -1
            break
    return d, mat_i, p


@ti.func
def raymarch_outline(ro, rd, ow):
    """
   function raymarch for toon image
   ow — outline wight
    """
    p = ro
    d = 0.
    ds, mat_i = dist_mat(p)
    min_dist = ds
    mat_i = 0
    for i in range(MAX_STEPS):
        min_dist = min(min_dist, ds)
        d += ds
        p = ro + d * rd
        ds, mat_i = dist_mat(p)
        if ds < EPS:
            break
        if d > MAX_DIST:
            mat_i = -1
            break
        if ds > min_dist and min_dist < ow:
            mat_i = -2
            break
    return d, mat_i, p


e_x = vec3(EPS, 0., 0.)
e_y = vec3(0., EPS, 0.)
e_z = vec3(0., 0., EPS)


@ti.func
def normal(p):
    """
    Finds the normal to vector p
    :param p: vec3, vector
    :return: vec3, normal
    """
    n = dist(p) - vec3(dist(p - e_x), dist(p - e_y), dist(p - e_z))
    return normalize(n)


@ti.func
def normal(p, rd):
    """
    Finds the normal to vector p in the direction rd
    :param p: vec3, vector
    :param rd: vec3, vector
    :return: vec3, normal
    """
    n = dist(p) - vec3(dist(p - e_x), dist(p - e_y), dist(p - e_z))
    return normalize(n - max(0., dot(n, rd)) * rd)


@ti.func
def phong(n, rd, ld, sh):
    """
    Function for calculation the light
    :param n: vec3, normal
    :param rd: vec3, ray direction
    :param ld: vec3, light direction
    :param sh: ti.f32, shadow
    :return: (ti.f32, ti.f32, ti.f32) —  coeffs of the differential, specular and fresnel contains
    """
    diff = max(dot(n, ld), 0.)
    r = reflect(-ld, n)
    v = -rd
    spec = max(dot(r, v), 0.) ** sh
    # fresnel = pow(clamp(1.0 + dot(rd, n), 0., 1.), 3.0)
    return diff, spec  # , fresnel


@ti.func
def shadow(ro, rd):
    """
    Finds shadow
    :param ro: vec3, ray origin
    :param rd: vec3, ray direction
    :return: ti.f32, coeff of the shadow
    """
    sh = 1.
    d = 0.
    for i in range(MAX_STEPS):
        ds = dist(ro + d * rd)
        d += ds
        if ds < EPS:
            sh = 0.
            break
        if d > MAX_DIST:
            break
    return sh


@ti.func
def softshadow(ro, rd, mint=0.3, maxt=170.0, k=32):
    """
    Finds soft shadow
    :param ro: vec3, ray origin
    :param rd: vec3, ray direction
    :param mint: ti.f32, minimal step over the border
    :param maxt: ti.f32, max step over the border
    :param k: coefficient of the softness
    :return: ti.f32, coeff of the soft shadow
    """
    res_shad = 1.0
    ph = 1e20
    t = mint
    for i in range(MAX_STEPS):
        while t < maxt:
            h = dist(ro + rd * t)
            if h < 0.001:
                res_shad = 0.
                break
            y = h * h / (2.0 * ph)
            d = ti.sqrt(h * h - y * y)
            res_shad = min(res_shad, k * d / max(0.0, t - y))
            ph = h
            t += h

    return res_shad


@ti.func
def ambientocclusion(p, normal):  # vec3 p, vec3 normal
    """
    Finds ambient occlusion
    :param p: vec3, pixel coords
    :param normal: vec3, normal
    :return: ti.f32, ambient occlusion
    """
    occ = 0.0
    weight = 1.0
    for i in range(8):
        len = 0.01 + 0.02 * i * i
        dst = dist(p + normal * len)
        occ += (len - dst) * weight
        weight *= 0.85
    return 1.0 - clamp(0.6 * occ, 0.0, 1.0)


@ti.func
def render(ro, rd):  # ro: vec3, rd: vec3, col: vec3
    """
    Finds color of the pixel
    :param ro: vec3, ray origin
    :param rd: vec3, ray direction
    :param col: vec3, prev step color
    :return: vec3, color
    """
    d, mat_i, p, shad = 0., 0., vec3(0.), 0.

    if style == 'toon':
        d, mat_i, p = raymarch_outline(ro, rd, 0.05)
    else:
        d, mat_i, p = raymarch(ro, rd)

    n = normal(p, rd)
    lp = vec3(5., 5., -5.)
    ld = normalize(lp - p)
    mat_n = materials.shape[0]
    background = materials[mat_n - 1] - abs(rd.y)
    mat_i = (mat_i + mat_n) % mat_n
    mate = materials[mat_i]

    occ = 1.0
    col = vec3(0.)
    if mat_i >= mat_n - 2:
        col = mate  # background, outline
    else:
        # diff, spec , fres = phong(n, rd, ld, 16)
        diff, spec = phong(n, rd, ld, 16)

        if style == 'toon':
            shad = shadow(p + n * EPS, ld)
            diff = ti.ceil(diff * 3.) / 3.

        else:
            shad = softshadow(p + n * EPS, ld)
            occ = ambientocclusion(p, n)

        # if mat_i == 1:
        #     mate = texture(vec2(p.x, p.z) / 20, 32.)
        if mat_i == 1:
            mate = boxmap_texture(translate_cube(rotate_cube(p - vec3(1., -0.5, 2.5))), rotate_cube(n), 60, 32.)


        k_a = 0.3
        k_d = 1.0
        k_s = 1.5  # 1.5
        k_f = 1.0
        amb = mate
        dif = diff * mate
        spe = spec * vec3(1.)
        col = k_a * amb * occ + (k_d * dif + k_s * spe * occ) * shad

    # fog
    col = mix(col, background, smoothstep(20., 50., d))
    return col


# @ti.func
# def render(ro, rd):  # ro: vec3, rd: vec3, t: ti.f32
#     d, mat_i, p = 0., 0, vec3(0.)
#     if style == 'toon':
#         d, mat_i, p = raymarch_outline(ro, rd, 0.05)
#     else:
#         d, mat_i, p = raymarch(ro, rd)
#     n = normal(p, rd)
#     col = vec3(0.)
#     lp = vec3(5., 5., -5.)
#     ld = normalize(lp - p)
#     mat_n = materials.shape[0]
#     background = materials[mat_n - 1] - abs(rd.y)
#     mat_i = (mat_i + mat_n) % mat_n
#     mate = materials[mat_i]
#     shad = 0.
#     shad = 0.
#     shad = 0.
#     shad = 0.
#
#
#     if mat_i >= mat_n - 2:
#         col = mate  # background, outline
#     else:
#         diff, spec = phong(n, rd, ld, 16)
#
#         # if mat_i == 1:
#         #     mate = texture(vec2(p.x, p.z)/20, 32.)
#         if mat_i == 0 or mat_i == 2:
#             #mate = boxmap_texture(translate_cube(rotate_cube(p)), rotate_cube(n), 60, 32.)
#             diff = ti.ceil(diff*2.)/2.
#
#         if style == 'toon':
#             shad = shadow(p + n * EPS, ld)
#         else:
#             shad = softshadow(p + n * EPS, ld)
#
#         k_a = 0.3
#         k_d = 1.0
#         k_s = 0. #1.5
#         amb = mate
#         dif = diff * mate
#         spe = spec * vec3(1.)
#         col = k_a * amb + (k_d * dif + k_s * spe) * shad
#
#     # fog
#     col = mix(col, background, smoothstep(20., 50., d))
#     return col


@ti.kernel
def main_image():
    """
    Main cicle
    :return: -
    """
    t = global_time[None]
    background = materials[-1]
    mp = ti.static(mouse_pos)
    mb = ti.static(mouse_btn)
    mbp = ti.static(mouse_btn_prev)
    muv = 2 * np.pi * (mp[None] - 0.5)


    mat_cam = rot_y(frame_counter*3)
    m = rot_y(muv.x)
    ro = mat_cam @ (m @ vec3(0., 1., -6. + muv.y))
    la = vec3(0.)
    up = vec3(0., 1., 0.)
    c, r, u = lookat(ro, la, up, 1.)

    if mb[0] == 0 and mbp[0] == 1:
        flags[0] = 1 - flags[0]

    for fragCoord in ti.grouped(pixels):
        col = vec3(0.)
        for i in range(AA):
            for j in range(AA):
                uv = (fragCoord + vec2(i, j) / AA - 0.5 * vec2(res)) / res[1]
                rd =  normalize(c + uv.x * r + uv.y * u)
                col += render(ro, rd)
        col /= AA ** 2
        col /= 5 / 4
        # gamma correction, clamp, write to pixel
        pixels[fragCoord] = clamp(col ** (1 / 2.2), 0., 1.)
    for i in range(2):
        mbp[i] = mb[i]


# %% GUI and main loop
frame_counter = 0
gui = ti.GUI("Taichi ray marching shader", res=res, fast_gui=True)
start = time.time()
result_dir = 'C:\\Users\\acer\\PycharmProjects\HW_Ray_Marching\\'
video_manager = ti.VideoManager(width=w, height=h, output_dir=result_dir, framerate=60, automatic_build=False)
mouse_btn_prev[0] = 1 if gui.is_pressed(ti.ui.LMB) else 0
mouse_btn_prev[1] = 1 if gui.is_pressed(ti.ui.RMB) else 0

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    # mouse position
    mpos = gui.get_cursor_pos()  # [0..1], [0..1]
    mouse_pos[None] = [np.float32(mpos[0]), np.float32(mpos[1])]
    # mouse buttons
    mouse_btn[0] = 1 if gui.is_pressed(ti.ui.LMB) else 0
    mouse_btn[1] = 1 if gui.is_pressed(ti.ui.RMB) else 0
    # time
    global_time[None] = time.time() - start


    main_image()

    # gui.set_image(pixels)
    frame_counter += 1
    video_manager.write_frame(pixels)

    gui.show()
    if frame_counter >= 1800:
        video_manager.make_video(mp4=True)
        # gui.close()
        exit(0)

