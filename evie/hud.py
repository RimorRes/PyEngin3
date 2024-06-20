import tkinter as tk
from PIL import ImageTk, Image


# TODO: fullscreen, blend video stream and AR objects, add overlays
root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.overrideredirect(1)
root.geometry("%dx%d+0+0" % (w, h))
root.focus_set()
root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))

canvas = tk.Canvas(root, width=w, height=h, bd=0, highlightthickness=0)
canvas.pack()
canvas.configure(background='black')

img = Image.open('../res/test_card.jpg')
image = ImageTk.PhotoImage(img)
imagesprite = canvas.create_image(w/2, h/2, image=image)

root.mainloop()
