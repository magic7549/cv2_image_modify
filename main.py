import os
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msgbox
from PIL import Image
from PIL import ImageTk




# 이미지 저장
def save_image():
    global cv_image
    
    # 이미지 쓰기
    cv2.imwrite("../result.jpg", cv_image)

    # 이미지가 저장된 폴더 열기
    path = os.path.realpath("../")
    os.startfile(path)

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

    # 사이즈가 같은 이미지가 선택되었는지 확인
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


# 볼록/오목 렌즈 효과
def remap_lens(x):
    global cv_image

    rows, cols = cv_image.shape[:2]

    # x == 0 -> 볼록 렌즈,       x == 1 -> 오목 렌즈
    if x == 0:
        exp = 1.5
    else :
        exp = 0.5
    scale = 1

    mapy, mapx = np.indices((rows, cols),dtype=np.float32)

    mapx = 2*mapx/(cols-1)-1
    mapy = 2*mapy/(rows-1)-1

    r, theta = cv2.cartToPolar(mapx, mapy)

    r[r< scale] = r[r<scale] **exp  

    mapx, mapy = cv2.polarToCart(r, theta)

    mapx = ((mapx + 1)*cols-1)/2
    mapy = ((mapy + 1)*rows-1)/2

    distorted = cv2.remap(cv_image,mapx,mapy,cv2.INTER_LINEAR)

    show_image(distorted)


# 모자이크
def mosaic():
    global cv_image

    ksize = 50
    while True:
        x,y,w,h = cv2.selectROI("mosaic", cv_image, False)
        if w > 0 and h > 0:
            roi = cv_image[y:y+h, x:x+w] 
            roi = cv2.blur(roi, (ksize, ksize)) 
            cv_image[y:y+h, x:x+w] = roi
        else:
            break

    cv2.destroyAllWindows()
    show_image(cv_image)


# 스케치 효과
def sketch():
    global cv_image

    img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
    edges = cv2.Laplacian(img_gray, -1, None, 5)
    _, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)
    sketch = cv2.medianBlur(sketch, 5)

    img_paint = cv2.blur(cv_image, (10,10) )
    img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)

    show_image(img_paint)


# 배경 제거
def grabcut():
    img = image_load()
    img_draw = img.copy()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    rect = [0,0,0,0]
    mode = cv2.GC_EVAL

    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)

    def onMouse(event, x, y, flags, param):
        nonlocal mode
        if event == cv2.EVENT_LBUTTONDOWN :
            if flags <= 1:
                mode = cv2.GC_INIT_WITH_RECT
                rect[:2] = x, y 
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON :
            if mode == cv2.GC_INIT_WITH_RECT:
                img_temp = img.copy()
                cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
                cv2.imshow("img", img_temp)
            elif flags > 1:
                mode = cv2.GC_INIT_WITH_MASK 
                cv2.imshow("img", img_draw)
        elif event == cv2.EVENT_LBUTTONUP:
            if mode == cv2.GC_INIT_WITH_RECT :
                rect[2:] =x, y
                cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255,0,0), 2)
                cv2.imshow("img", img_draw)
            cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
            img2 = img.copy()
            img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 255
            mode = cv2.GC_EVAL

            cv2.destroyAllWindows()
            show_image(img2)
      
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", onMouse)
    

# 문서 스캔
def paper_scan():
    img = image_load()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) 
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        vertices = cv2.approxPolyDP(c, 0.02 * peri, True) 
        if len(vertices) == 4:
            break
    pts = vertices.reshape(4, 2)

    sm = pts.sum(axis=1)
    diff = np.diff(pts, axis = 1)

    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0]) 
    w2 = abs(topRight[0] - topLeft[0]) 
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, mtrx, (width, height))

    cv2.destroyAllWindows()
    show_image(result)


# 시-토마시 코너 검출
def corner_goodFeature():
    img = image_load()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)
    corners = np.int32(corners)

    for corner in corners:
        x, y = corner[0]
        cv2.circle(img, (x, y), 5, (0,0,255), 1, cv2.LINE_AA)

    show_image(img)


# ORB 디스크립터 추출
def orb_desc():
    img = image_load()

    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(img, None)

    img_draw = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_image(img_draw)


# 노이즈 제거
def noise_remove(x):
    global cv_image
    
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    if x == 0:
        result = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, k)
    else:
        result = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, k)

    show_image(result)


# 리퀴파이 도구
def liquify():
    win_title = "Esc : Apply"
    half = 50 
    isDragging = False 
    cx1, cy1 = 0, 0

    def liquify_apply(img, cx1, cy1, cx2, cy2):
        x, y, w, h = cx1-half, cy1-half, half*2, half*2

        roi = img[y:y+h, x:x+w].copy()
        out = roi.copy()

        offset_cx1,offset_cy1 = cx1-x, cy1-y
        offset_cx2,offset_cy2 = cx2-x, cy2-y
        
        tri1 = [[ (0, 0), (w, 0), (offset_cx1, offset_cy1)],
                [ [0, 0], [0, h], [offset_cx1, offset_cy1]],
                [ [w, 0], [offset_cx1, offset_cy1], [w, h]],
                [ [0, h], [offset_cx1, offset_cy1], [w, h]]]

        tri2 = [[ [0, 0], [w, 0], [offset_cx2, offset_cy2]],
                [ [0, 0], [0, h], [offset_cx2, offset_cy2]],
                [ [w, 0], [offset_cx2, offset_cy2], [w, h]],
                [ [0, h], [offset_cx2, offset_cy2], [w, h]]]

        
        for i in range(4):
            matrix = cv2.getAffineTransform(np.float32(tri1[i]), np.float32(tri2[i]))
            warped = cv2.warpAffine(roi.copy(), matrix, (w, h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask = np.zeros((h, w), dtype = np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255,255,255))

            warped = cv2.bitwise_and(warped, warped, mask=mask)
            out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
            out = out + warped

        img[y:y+h, x:x+w] = out
        return img 

    def onMouse(event,x,y,flags,param):
        nonlocal cx1, cy1, isDragging, img  

        if event == cv2.EVENT_MOUSEMOVE:  
            if not isDragging :
                img_draw = img.copy()

                cv2.rectangle(img_draw, (x-half, y-half), (x+half, y+half), (0,255,0)) 
                cv2.imshow(win_title, img_draw)
        elif event == cv2.EVENT_LBUTTONDOWN :   
            isDragging = True 
            cx1, cy1 = x, y
        elif event == cv2.EVENT_LBUTTONUP :
            if isDragging:
                isDragging = False

                liquify_apply(img, cx1, cy1, x, y)    
                cv2.imshow(win_title, img)

    img = image_load()

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, onMouse) 
    cv2.imshow(win_title, img)

    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    show_image(img)


# 노멀라이즈
def normalize():
    global cv_image

    img_norm = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
    show_image(img_norm)


# 노멀라이즈
def equalize():
    global cv_image

    img_yuv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    show_image(img)


# CLAHE
def clahe():
    global cv_image

    img_yuv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)

    img_eq = img_yuv.copy()
    img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

    show_image(img_clahe)



cv_image = np.empty((1,1,3), dtype=np.uint8)

root = Tk()
root.title("그림판")
root.geometry("1300x800")   # 창 크기

# ==========================================================
# 상단 메뉴 창
menu = Menu(root)

menu_file = Menu(menu, tearoff=0)
menu_file.add_command(label="Load File", command=lambda: show_image(image_load()))
menu_file.add_command(label="Save Image", command=save_image)
menu_file.add_separator()
menu_file.add_command(label="Exit", command=root.quit)

menu_transform = Menu(menu, tearoff=0)
menu_transform.add_command(label="시계반향으로 90도 회전", command=rotation_90)
menu_transform.add_separator()
menu_transform.add_command(label="볼록 렌즈 효과", command=lambda: remap_lens(0))
menu_transform.add_command(label="오목 렌즈 효과", command=lambda: remap_lens(1))
menu_transform.add_separator()
menu_transform.add_command(label="모자이크 효과", command=mosaic)
menu_transform.add_command(label="스케치 효과", command=sketch)
menu_transform.add_separator()
menu_transform.add_command(label="노이즈 제거(열림)", command=lambda: noise_remove(0))
menu_transform.add_command(label="노이즈 제거(닫힘)", command=lambda: noise_remove(1))
menu_transform.add_separator()
menu_transform.add_command(label="노멀라이즈", command=normalize)
menu_transform.add_command(label="이퀄라이즈", command=equalize)
menu_transform.add_command(label="CLAHE", command=clahe)

menu.add_cascade(label="File", menu=menu_file)
menu.add_cascade(label="Transform", menu=menu_transform)

root.config(menu=menu)
# ==========================================================
# 왼쪽 메뉴 창
frame_left = Frame(root, background="gray") # relief : 테두리, bd : 외곽선 두께
frame_left.pack(side="left", fill="both")

Button(frame_left, text="alpha\nblending", padx=5, pady=5, width=6, height=2, command=alpha_blending).pack()
Button(frame_left, text="배경 제거", padx=5, pady=5, width=6, height=2, command=grabcut).pack()
Button(frame_left, text="문서 스캔", padx=5, pady=5, width=6, height=2, command=paper_scan).pack()
Button(frame_left, text="코너 검출", padx=5, pady=5, width=6, height=2, command=corner_goodFeature).pack()
Button(frame_left, text="디스크립터\n추출", padx=5, pady=5, width=6, height=2, command=orb_desc).pack()
Button(frame_left, text="리퀴파이", padx=5, pady=5, width=6, height=2, command=liquify).pack()

# ==========================================================
# 메인 프레임
frame_image = Frame(root)
frame_image.pack(side="right", fill="both", expand=True)

label_image = Label(frame_image)
label_image.pack()

# ==========================================================

root.mainloop()

