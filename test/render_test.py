from evie import *
import tkinter as tk
from PIL import ImageTk, Image
import time


# TODO: Multithreading! go see:
# TODO: https://stackoverflow.com/questions/62576326/python3-process-and-display-webcam-stream-at-the-webcams-fps
root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.overrideredirect(1)
root.geometry("%dx%d+0+0" % (w, h))
root.focus_set()
root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))

canvas = tk.Canvas(root, width=w, height=h, bd=0, highlightthickness=0)
canvas.pack()
canvas.configure(background='black')
imagesprite = canvas.create_image(w/2, h/2, image=ImageTk.PhotoImage(image=Image.new('RGB', (w, h), 'black')))
fps_counter = canvas.create_text(100, 100, fill='red', anchor=tk.NW)


f = 50e-3  # 50mm focal length
s = 36e-3  # 36mm sensor width
res = (960, 960)

SCENE = Scene()
CAM_L = VirtualCamera(res, f, s)
CAM_R = VirtualCamera(res, f, s)
CAM_L.pos = np.array([30e-3, 0, 0])
CAM_R.pos = np.array([-30e-3, 0, 0])

RET = Reticle()
RET.depth = 0.5

AXES = Axes()
AXES.pos = np.array([0, 0, 5])
AXES.rot = np.array([0, 0, 0])

SQ = WireSquare()
SQ.pos = np.array([0, 0, 5])
SQ.rot = np.array([0, 0, 0])
SQ.scale *= 1

img = Image.open('../res/test_card.jpg').convert('RGBA')
CARD = ImagePlane(img)
CARD.pos = (0, 0, 5)
CARD.rot = (0, 0, 0)
CARD.scale *= 1

SCENE.add(AXES)
SCENE.add(RET)
SCENE.add(SQ)
SCENE.add(CARD)

a = np.linspace(-np.pi, np.pi, 100)


def anim(step, last):
    CARD.rot = (0, a[step], 0)
    step += 1
    if step == 100:
        step = 0

    render_l = SCENE.render(CAM_L)
    render_r = SCENE.render(CAM_R)
    render = Image.blend(render_l, render_r, 0.5)
    imgtk = ImageTk.PhotoImage(image=render)

    now = time.time()

    canvas.itemconfig(imagesprite, image=imgtk)
    canvas.itemconfigure(fps_counter, text=f'FPS: {int(1/(now - last))}')
    canvas.imgref = imgtk
    canvas.after(1, anim, step, now)


start = time.time()
anim(0, 0)
root.mainloop()
