import taichi as ti
import time
import numpy as np
from shader_funcs import *

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)

# %% Resolution and pixel buffer

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h

# %% Fields (should always be before first call of any taichi scope kernels)

global_time = ti.field(dtype=ti.f32, shape=())
mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
flags = ti.field(dtype=ti.i32, shape=(8,))
mouse_btn = ti.field(dtype=ti.i32, shape=(2,))
materials = ti.Vector.field(3, dtype=ti.f32, shape=(4,))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

# %%

materials.from_numpy(np.array([
    [76, 125, 18],  # green
    [200, 50, 10],  # red
    [94, 63, 43],  # brown
    [23, 33, 76],
    [223, 63, 100],
    [30, 3, 42],
    # [ 30, 3, 42],
    [1, 0, 0],  # red
    [128, 205, 230]  # background
],
    dtype=np.float32) / 255.)
# %%

MAX_DIST = 100.
MAX_STEPS = 200
EPS = 1e-3
INF = 1e10

e_x = vec3(EPS, 0., 0.)
e_y = vec3(0., EPS, 0.)
e_z = vec3(0., 0., EPS)


# %%

@ti.func
def sd_snowmen(p1, R1, Rcoef):
    R2 = Rcoef * R1
    R3 = Rcoef * R2
    p2 = p1 - vec3(0., R1 + R2 / 2, 0.)
    p3 = p2 - vec3(0., R2 + R3 / 2, 0.)
    p4 = p3 - normalize(vec3(R3 * ti.cos(deg2rad(20)), R3 * ti.sin(deg2rad(20)), R3 / 2)) * R3
    p5 = p3 - normalize(vec3(R3 * ti.cos(deg2rad(20)), R3 * ti.sin(deg2rad(20)), -R3 / 2)) * R3

    sphere1 = vec2(sd_sphere(p1, R1), 0)
    sphere2 = vec2(sd_sphere(p2, R2), 0)
    sphere3 = vec2(sd_sphere(p3, R3), 0)
    sphere4 = vec2(sd_sphere(p4, R3 / 12), 0)
    sphere5 = vec2(sd_sphere(p5, R3 / 12), 0)

    res12 = f_op_union_i(sphere1, sphere2)
    res123 = f_op_union_i(res12, sphere3)
    res1234 = f_op_union_i(res123, sphere4)
    res12345 = f_op_union_i(res1234, sphere5)
    return res12345[0]


@ti.func
def sd_hexagonalprism(p, h):
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
def sd_solidangle(p, c, ra):
    # c is the sin/cos of the angle
    q = vec2(length(vec2(p.x, p.z)), p.y)
    l = length(q) - ra
    m = length(q - c * clamp(dot(q, c), 0.0, ra))
    return max(l, m * sign(c.y * q.x - c.x * q.y))


# @ti.func
# def texture(uv, scale):
#     d = 0.
#     if flags[0]:
#         d = sd_box(uv, vec2(1.))
#     else:
#         d = sd_circle(uv, 0.5)
#     col = vec3(0.)
#     if d < 0:
#         col = vec3(0.5, 0.85, 1.0)
#         col *= 0.8 + 0.2 * skewsin(scale * d, 0.8)
#         col *= 1.0 - ti.exp(-7.0 * abs(d))
#     return col
#
#
# @ti.func
# def boxmap_texture(p, n, k, s):
#     xcol = texture(vec2(p.y, p.z), s)
#     ycol = texture(vec2(p.z, p.x), s)
#     zcol = texture(vec2(p.x, p.y), s)
#     w = abs(n) ** k
#     return (xcol * w.x + ycol * w.y + zcol * w.z) / w.sum()
#
#
# @ti.func
# def rotate_cube(p):
#     t = global_time[None]
#     m = rot_x(0.1 * t)
#     return m @ p
#
#
# @ti.func
# def translate_cube(p):
#     t = global_time[None]
#     return p - vec3(ti.sin(0.2 * t), 0., 0.)


@ti.func
def set_camera(ro, ta, cr):
    cw = normalize(ta - ro)
    cp = vec3(ti.sin(cr), ti.cos(cr), 0.0)
    cu = normalize(cross(cw, cp))
    cv = (cross(cu, cw))
    return vec3(cu, cv, cw)


@ti.func
def dist_vector(p):
    p1 = p - vec3(0., 1., 0.)
    p2 = p
    p3 = p - vec3(3., 0., 0.)
    p4 = p - vec3(-1, 0, -1)
    p5 = p - vec3(5., 0., 5.)
    p6 = p - vec3(0., 1., 0.)
    p7 = p - vec3(5., 2., 5.)

    # d1 = sd_hexagonalprism(p1, vec2(1., 2.))
    d2 = p.y
    # d3 = sd_torus(p1, vec2(1., 0.5))
    ds = sd_snowmen(p4, 2.0, 2 / 3)
    dc = sd_cappedcylinder(p3, 0.3, 2)
    # drb = sd_roundbox(p5 + 6, 0.5, vec4(0.75, 0.75, 0.75, 0.75) / 2)
    #
    # p8 = p - vec3(-2, 0, 1)
    # d = sd_torus(p8, vec2(1., -1.))

    # # sphere = vec2(d1, 1.0)
    # plane = vec2(d2, 2.0)
    # torus = vec2(d3, 3.0)
    # cappedcylinder = vec2(dc, 4.0)
    # roundbox = vec2(drb, 5.0)
    # snowmen = vec2(ds, 4)

    # cubik = vec2(sd_box(p5, 1.), 6.0)
    # roof = vec2(sd_cone(p7, vec2(1.0, 2.0), 2.0), 7.0)
    # dh = f_op_union_i(cubik, roof)[0]
    # d4 = f_op_union_i(sphere, torus)[0]

    # return d
    # return vec3(d1, ds, d)
    return vec2(d2, dc)
    # return vec6(d2, ds, d4, d3, drb, dc)


@ti.func
def dist(p):
    return dist_vector(p).min()


@ti.func
def dist_mat(p):
    return argmin(dist_vector(p))


@ti.func
def normal_p(p):
    n = dist(p) - vec3(dist(p - e_x), dist(p - e_y), dist(p - e_z))
    return normalize(n)


@ti.func
def normal(p, rd):
    n = dist(p) - vec3(dist(p - e_x), dist(p - e_y), dist(p - e_z))
    return normalize(n - max(0., dot(n, rd)) * rd)


@ti.func
def phong(n, rd, ld, sh):
    diff = max(dot(n, ld), 0.)
    r = reflect(-ld, n)
    v = -rd
    spec = max(dot(r, v), 0.) ** sh
    return diff, spec


@ti.func
def f_op_union_i(v1, v2):
    return v1 if v1.x < v2.x else v2


# @ti.func
# def f_op_union_i_pro(vs):
#     l = 0
#     for v in vs:
#         l += 1
#
#     vs[l - 2] = f_op_union_i(vs[l - 1], vs[l - 2])
#     vs.pop() #delete last
#     return f_op_union_i_pro(vs)[0] if l > 1 else vs[0][0]


@ti.func
def map(p):
    cone_dist = sd_cone(p, 0.45, 1.2)
    cone_i = 2.0
    cone = vec2(cone_dist, cone_i)

    sphere_dist = sd_sphere(p, 1.0)
    sphere_i = 1.0
    sphere = vec2(sphere_dist, sphere_i)

    res = f_op_union_i(sphere, cone)

    return res


@ti.func
def raymarch(ro, rd):
    p = vec3(0.)
    d = 0.
    mat_i = -1
    for i in range(MAX_STEPS):
        p = ro + d * rd
        ds, mat_i = dist_mat(p)
        d += ds
        if ds < EPS or d > MAX_DIST:
            break
    return d, p, mat_i


@ti.func
def toonmarch(ro, rd, border_w=0.01):
    prev_dist = INF
    p = vec3(0.)
    d = 0.
    mat_i = -1
    for i in range(MAX_STEPS):
        p = ro + d * rd
        ds, mat_i = dist_mat(p)
        d += ds
        if prev_dist <= ds < border_w:
            mat_i = -2
            break
        if ds < EPS or d > MAX_DIST:
            break
        prev_dist = ds
    return d, p, mat_i


@ti.func
def raymarch_outline(ro, rd, ow=0.01):
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
    return d, p, mat_i


@ti.func
def blinn_phong(n, rd, ld, sh):
    vd = -rd
    # r = reflect(ld, n)
    hd = normalize(ld + vd)
    lamb = max(dot(ld, n), 0.)
    spec = max(dot(hd, n), 0.) ** sh
    return lamb, spec


@ti.func
def shadow(ro, rd):
    d = EPS * 20
    res = 1.
    for i in range(MAX_STEPS):
        ds = dist(ro + d * rd)
        d += ds
        if d > MAX_DIST:
            break
        if ds < EPS:
            res = 0.
            break
    return res


@ti.func
def softshadow(ro, rd, mint=0.03, maxt=3.0, k=32):
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
def offset_uv(uv, offset):
    return ((uv * res[1] + 0.5 * vec2(res) + offset) - 0.5 * vec2(res)) / res[1]


@ti.func
def renderAAx4(ro, uv, rot_matr):
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

    return colAA / 4.0


# %%

@ti.func
def render(ro, rd):  # ro: vec3, rd: vec3, t: ti.f32
    # d, p, mat_i = raymarch(ro, rd)
    d, p, mat_i = toonmarch(ro, rd)
    d, p, mat_i = raymarch_outline(ro, rd)
    n = normal(p, rd)
    mate = materials[mat_i]
    lp = vec3(20., 20., -20.)
    ld = normalize(lp - p)
    background = materials[-1]
    mat_i = (mat_i + materials.shape[0]) % materials.shape[0]

    col = vec3(0.)

    if mat_i >= mat_n - 2:
        col = mate  # background, outline
    else:
        diff, spec = phong(n, rd, ld, 16)

        if mat_i == 1:
            mate = texture(vec2(p.x, p.z)/20, 32.)
        elif mat_i == 0 or mat_i == 2:
            #mate = boxmap_texture(translate_cube(rotate_cube(p)), rotate_cube(n), 60, 32.)
            diff = ti.ceil(diff*2.)/2.

        shad = shadow(p + n * EPS, ld)
        k_a = 0.3
        k_d = 1.0
        k_s = 0. #1.5
        amb = mate
        dif = diff * mate
        spe = spec * vec3(1.)
        col = k_a * amb + (k_d * dif + k_s * spe) * shad

    if d < MAX_DIST:
        n = normal_p(p)
        lamb, spec = blinn_phong(n, rd, ld, 32)
        shd = shadow(p, ld)
        # shd = softshadow(p, ld)
        # mate = vec3(1., 0., 0.)
        amb = 0.3 * mate
        dif = mate * lamb
        dif = ti.ceil(dif * 2.) / 2.
        spe = background * spec

        col = amb + (dif + spe) * shd
    else:
        col = background - abs(0.9 * rd.y)

    # fog
    col = mix(col, background, smoothstep(20., 50., d))

    return col


# @ti.func
# def render_cons(ro, rd):  # ro: vec3, rd: vec3, t: ti.f32
#     # d, mat_i, p = raymarch(ro, rd)
#     d, mat_i, p = raymarch_outline(ro, rd, 0.05)
#     n = normal(p, rd)
#     col = vec3(0.)
#     lp = vec3(5., 5., -5.)
#     ld = normalize(lp - p)
#     mat_n = materials.shape[0]
#     background = materials[mat_n - 1] - abs(rd.y)
#     mat_i = (mat_i + mat_n) % mat_n
#     mate = materials[mat_i]
#
#     if mat_i >= mat_n - 2:
#         col = mate  # background, outline
#     else:
#         diff, spec = phong(n, rd, ld, 16)
#
#         if mat_i == 1:
#             mate = texture(vec2(p.x, p.z) / 20, 32.)
#         elif mat_i == 0 or mat_i == 2:
#             # mate = boxmap_texture(translate_cube(rotate_cube(p)), rotate_cube(n), 60, 32.)
#             diff = ti.ceil(diff * 2.) / 2.
#
#         shad = shadow(p + n * EPS, ld)
#         k_a = 0.3
#         k_d = 1.0
#         k_s = 0.  # 1.5
#         amb = mate
#         dif = diff * mate
#         spe = spec * vec3(1.)
#         col = k_a * amb + (k_d * dif + k_s * spe) * shad
#
#     # fog
#     col = mix(col, background, smoothstep(20., 50., d))
#     return col


#
# @ti.func
# def f_movement(p):
#     m = rot_y(ti.sin(2.0 * global_time[None]))
#     p.yz = m @ p.yz
#     return (ti.sin(p.x + 4.0 * global_time[None]) *
#             ti.sin(p.y + ti.sin(2.0 * global_time[None])) *
#             ti.sin(p.z + 6.0 * global_time[None]))


@ti.kernel
def main_image():
    k = 1
    t = global_time[None]
    # background = materials[-1]
    mp = ti.static(mouse_pos)
    mb = ti.static(mouse_btn)
    muv = 2 * np.pi * (mp[None] - 0.5)
    col = vec3(0.)
    for fragCoord in ti.grouped(pixels):
        uv = (fragCoord - 0.5 * vec2(res)) / res[1]
        m = rot_y(0.)
        if mb[0] == 1:
            # m = rot_y(muv.x * 2)
            m = rot_y(muv.x * 5)

        ro = m @ vec3(0., 3., 0.)
        if k == 1:
            rd = m @ normalize(vec3(uv.x, uv.y, 1.))
            col = render(ro, rd)
        if k == 2:
            col = renderAAx4(ro, uv, m)

        # gamma correction, clamp, write to pixel
        pixels[fragCoord] = clamp(col ** (1 / 2.2), 0., 1.)


# %% GUI and main loop

gui = ti.GUI("Taichi ray marching shader", res=res, fast_gui=True)
start = time.time()

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
    gui.set_image(pixels)
    gui.show()

gui.close()
