# %% Imports

import taichi as ti
from shader_funcs import *
import time

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

# %% Resolution and pixel buffer

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


# arr = np.zeros((w, h, 3))
# arr[i, j] = col
# %% Kernel

# https://www.shadertoy.com/new
@ti.kernel
def render(t: ti.f32, frame: ti.int32):
    col00 = vec3(0.50, 0.50, 0.50)
    col01 = vec3(1.00, 1.00, 1.00)
    col02 = vec3(0.00, 0.33, 0.67)
    col03 = vec3(0.00, 0.10, 0.20)

    for fragCoord in ti.grouped(pixels):
        uv0 = fragCoord / res[1]
        tr = 0.5 * vec2(res) / res[1]
        uv = rot(-t * 0.1) @ (uv0 - tr) + tr - ti.sin(hash22(vec2(1.)) * 0.1 * t)
        uv *= 10

        id = ti.floor(uv)
        h = hash22(id)
        fuv = fract(uv) - 0.5

        d = 100000.
        for i in range(-1, 2):
            for j in range(-1, 2):
                offset = vec2(i, j)
                id_ij = id + offset
                h_ij = hash22(id_ij)
                fuv_ij = fuv - offset - 0.4 * ti.sin(h_ij * t)
                d_ij = sd_circle(fuv_ij, 0.2 + 0.3 * h_ij.x)
                d = smoothmin(d, d_ij, 0.4)

        d -= 0.15 * skewsin(10 * (uv0.x * uv0.y) + 2 * t, 0.7)

        col = pal(d * ti.cos(uv.x + uv.y), col00, col00, col01, col02)
        if d < 0:
            # col = vec3(0., 1., 0.)
            col *= 0.8 + 0.2 * skewsin(64. * d, 0.8)
        else:
            # col = vec3(0., 0., 1.)
            col *= 0.8 + 0.2 * ti.cos(16. * d)

        col *= 1. - ti.exp(-10. * abs(d))
        col = mix(col, vec3(1.), smoothstep(0.01, 0.0, abs(d)))

        # grid
        # col = mix(col, vec3(1.), smoothstep(0.48, 0.5, abs(fuv).max()))

        pixels[fragCoord] = col


# %% GUI and main loop

gui = ti.GUI("Taichi simple shader", res=res, fast_gui=True)
frame = 0
start = time.time()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break

    t = time.time() - start
    render(t, frame)
    gui.set_image(pixels)
    gui.show()
    frame += 1

gui.close()
