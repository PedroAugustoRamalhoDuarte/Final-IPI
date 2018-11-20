import cv2 as cv
import numpy as np


def subtracao(frame, previus_frame):
    linhas, colunas = frame.shape
    result = np.zeros((linhas, colunas), dtype=np.uint8)
    for i in range(0, linhas):
        for j in range(0, colunas):
            result[i][j] = frame[i][j] - previus_frame[i][j]
    return result

cap = cv.VideoCapture("teste.avi")

cont = 0
intervalo = 3
previus_frame = np.zeros((720, 1280), dtype=np.uint8)
background = np.zeros((720, 1280), dtype=np.uint8)
binary_mask3 = np.zeros((720, 1280, 3), dtype=np.uint8)
while(1):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if 1400 < cont < 1600:
        # 1 Passo- Converter o frame em cinza
        frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        # 2 Passo - Diferenca do pixel atual e o anteriro
        diff = cv.absdiff(frame_gray, previus_frame)
        # diff = cv.blur(diff, (7, 7))
        # 3 Passo - Thresh Holder
        a, binary_mask = cv.threshold(diff, 40, 255, cv.THRESH_BINARY)
        # 4 Passo - Aplicando o algoritmo de detectar bordas
        edges = cv.Canny(binary_mask, 40, 255)
        # Preparando mask binary para print
        for i in range(0, 3):
            binary_mask3[:, :, i] = binary_mask / 255
        # Obtendo background e foreground rgb
        background = frame_bgr * (1-binary_mask3)
        foreground = frame_bgr * binary_mask3
        cv.imshow("foreground", foreground)
        cv.imshow("background", background)
        cv.imshow("edge", edges)
        k = cv.waitKey(0) & 0xff
        print(cont)
        if k == 27:
            break
        previus_frame = frame_gray[:, :]
    cont += 1

cap.release()
cv.destroyAllWindows()
