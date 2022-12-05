import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msgbox
from PIL import Image
from PIL import ImageTk

def create_new_file():
    print("create new file!!")

# 이미지 불러오기
def image_load():   
    file = filedialog.askopenfilename(title="이미지 파일을 선택하세요", 
        filetypes=(("모든 파일", "*.*"), ("PNG 파일", "*.png"), ("JPG 파일", "*.jpg")),
        initialdir="C:/")

    img_array = np.fromfile(file, np.uint8)
    src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return src


# 이미지 사이즈 비율 맞게 조정
def image_size_control(src):
    height, width, _ = src.shape

    if (width > height):
        adj_width = 1000 / width
        dst = cv2.resize(src, dsize=(0, 0), fx=adj_width, fy=adj_width, interpolation=cv2.INTER_LINEAR)
    else:
        adj_height = 790 / height
        dst = cv2.resize(src, dsize=(0, 0), fx=adj_height, fy=adj_height, interpolation=cv2.INTER_LINEAR)

    return dst


# 이미지 띄우기
def show_image(src):
    global label_image, cv_image

    src = image_size_control(src)
    cv_image = src

    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    

    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    label_image.config(image=imgtk)
    label_image.image = imgtk
    

# 알파 블렌딩
def alpha_blending():
    global label_image
    msgbox.showinfo("image", "첫번째 이미지를 선택해 주세요")
    img1 = image_load()

    msgbox.showinfo("image", "두번째 이미지를 선택해 주세요")
    img2 = image_load()
    
    alpha = 0.5 

    try:
        dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0) 
    except:
        msgbox.showerror("error", "같은 사이즈에 이미지를 선택해 주세요")
    else:
        show_image(dst)


# 이미지 회전
def rotation_90():
    global label_image, cv_image
    
    rows,cols = cv_image.shape[0:2]

    m90 = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1) 
    img90 = cv2.warpAffine(cv_image, m90, (cols, rows))

    show_image(img90)


cv_image = np.empty((1,1,3), dtype=np.uint8)

root = Tk()
root.title("그림판")
root.geometry("1300x800")   # 창 크기

# ==========================================================
# 상단 메뉴 창
menu = Menu(root)

menu_file = Menu(menu, tearoff=0)
menu_file.add_command(label="New File", command=create_new_file)
menu_file.add_command(label="Load Image", command=lambda: show_image(image_load()))
menu_file.add_separator()
menu_file.add_command(label="Exit", command=root.quit)

menu_transform = Menu(menu, tearoff=0)
menu_transform.add_command(label="시계반향으로 90도 회전", command=rotation_90)

menu.add_cascade(label="File", menu=menu_file)
menu.add_cascade(label="Transform", menu=menu_transform)

root.config(menu=menu)
# ==========================================================
# 왼쪽 메뉴 창
frame_left = Frame(root, background="gray") # relief : 테두리, bd : 외곽선 두께
frame_left.pack(side="left", fill="both")

Button(frame_left, text="alpha\nblending", padx=5, pady=5, width=6, height=2, command=alpha_blending).pack()

# ==========================================================
# 메인 프레임
frame_image = Frame(root)
frame_image.pack(side="right", fill="both", expand=True)

label_image = Label(frame_image)
label_image.pack()

# ==========================================================

root.mainloop()

