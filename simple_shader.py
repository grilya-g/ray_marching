
#%% Imports

import taichi as ti
from shader_funcs import *
import time

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)

#%% Resolution and pixel buffer

asp = 16/9
h = 600
w = int(asp * h)
res = w, h

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
#%% Kernel

# https://www.shadertoy.com/new
@ti.kernel
def render(t: ti.f32, frame: ti.int32):
    for fragCoord in ti.grouped(pixels):
        uv = fragCoord / vec2(res)
        col = 0.5 + 0.5 * ti.cos(t + vec3(uv.x, uv.y, uv.x) + vec3(0, 2, 4))
        pixels[fragCoord] = col

#%% GUI and main loop

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
