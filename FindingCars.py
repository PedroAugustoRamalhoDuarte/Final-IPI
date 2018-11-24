import cv2 as cv
import numpy as np
from fuction import *
cap = cv.VideoCapture("overpass.mp4")

cont = 0
intervalo = 3
previus_frame = np.zeros((720, 1280), dtype=np.uint8)
background = np.zeros((720, 1280), dtype=np.uint8)
binary_mask3 = np.zeros((720, 1280, 3), dtype=np.uint8)
previus_background = np.zeros((720, 1280), dtype=np.uint8)
backgroundbom = np.zeros((720, 1280, 3), dtype=np.uint8)
aloha = np.zeros((720, 1280), dtype=np.uint8)
matriz = np.zeros((720, 1280), dtype=np.uint8)
ret = True
while (ret):
    ret, frame_bgr = cap.read()
    # 1 Passo- Converter o frame em cinza
    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    imgprint("frame cinza", frame_gray)
    # 2 Passo - Diferenca do pixel atual e o anterior
    diff = cv.absdiff(frame_gray, previus_frame)
    imgprint("diferrence", diff)
    # 3 Passo - Thresh Holder
    a, binary_mask = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)
    imgprint("mascaa binaria",binary_mask)
    # 4 Passo - Aplicando o algoritmo de detectar bordas
    edges = cv.Canny(binary_mask, 0, 255)
    imgprint("bordas", edges)
    # 5 Passo - Melhorando bordas
    edges = morph_dilatation(edges)
    a, edges = cv.threshold(edges, 70, 1, cv.THRESH_BINARY)
    background = frame_gray * (1 - edges)
    foreground = frame_gray * edges
    imgprint("bordas melhoradas", edges)
    imgprint("background", background)
    imgprint("forregound", foreground)
    # Fine-grained segmentation of Current image
    diff = cv.absdiff(foreground, previus_background)
    a, diff = cv.threshold(diff, 200, 255, cv.THRESH_BINARY)
    imgprint("melhorado", diff)
    # alfa = 0.6
    # matriz[:, :] = (1 - alfa) * frame_background[:, :] + alfa * diff[:, :]
    k = cv.waitKey(0) & 0xff
    print(cont)
    if k == 27:
        break
    cont += 1
    previus_frame = frame_gray[:, :]
    previus_background = background[:, :]
cap.release()
cv.destroyAllWindows()
