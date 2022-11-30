import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image

def create_new_file():
    print("create new file!!")

def image_load():   # 이미지 불러오기
    file = filedialog.askopenfilename(title="이미지 파일을 선택하세요", 
        filetypes=(("모든 파일", "*.*"), ("PNG 파일", "*.png"), ("JPG 파일", "*.jpg")),
        initialdir="C:/")

    cv2_img = cv2.imread(file)

    global photo
    #photo = PhotoImage(file=file, width=1000, height=790)
    photo = PhotoImage(file=file)

    

    global canvas
    canvas.create_image([0, 0], image=photo)

    

root = Tk()
root.title("그림판")
root.geometry("1300x800")   # 창 크기

# ==========================================================
# 상단 메뉴 창
menu = Menu(root)

menu_file = Menu(menu, tearoff=0)
menu_file.add_command(label="New File", command=create_new_file)
menu_file.add_command(label="Load Image", command=image_load)
menu_file.add_separator()
menu_file.add_command(label="Exit", command=root.quit)

menu.add_cascade(label="File", menu=menu_file)

root.config(menu=menu)
# ==========================================================
# 왼쪽 메뉴 창
frame_left = Frame(root, background="gray") # relief : 테두리, bd : 외곽선 두께
frame_left.pack(side="left", fill="both")

Button(frame_left, text="test", padx=5, pady=5, width=6, height=2).pack()

# ==========================================================
# 메인 프레임
frame_image = Frame(root)
frame_image.pack(side="right", fill="both", expand=True)

canvas = Canvas(frame_image, background="yellow", width=1000, height=790)
canvas.pack()
# ==========================================================

root.mainloop()

