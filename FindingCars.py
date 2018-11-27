from fuction import *
from background import *
from auxiliares.roadDetection import *


cap = cv.VideoCapture("videos/roadvideo2Trim.mp4")

# Captura o primeiro frame para definir o numero de linhas e colunas e definir o primeiro previus_background
ret, frame = cap.read()
linhas, colunas, tipo = frame.shape
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
previus_background = frame_gray

# Contador de frames
cont = 0

# Imagens
previus_frame = np.zeros((linhas, colunas), dtype=np.uint8)
background = np.zeros((linhas, colunas), dtype=np.uint8)
background2 = np.zeros((linhas, colunas), dtype=np.uint8)
previus_background2 = np.zeros((linhas, colunas), dtype=np.uint8)
previus_foreground = np.zeros((linhas, colunas), dtype=np.uint8)
ground = np.zeros((linhas, colunas), dtype=np.uint8)
matriz = np.zeros((linhas, colunas), dtype=np.uint8)
previous_bgr = np.zeros((linhas, colunas, 3), dtype=np.uint8)

while (ret):
    # Captura frame a frame do video
    ret, frame_bgr = cap.read()

    # 1 Passo- Converter o frame em cinza
    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    imgprint("frame_gray", frame_gray)
    cont += 1

    # 2 Passo - Diferenca do pixel atual e o anterior
    diff = cv.absdiff(frame_gray, previus_frame)
    imgprint("diff", diff)

    # 3 Passo - Thresh Holder
    T = int(0.1 * np.amax(diff))
    a, binary_mask = cv.threshold(diff, T, 255, cv.THRESH_BINARY)

    # 4 Passo - Aplicando o algoritmo de detectar bordas
    edges = cv.Canny(binary_mask, 300, 350)
    imgprint("canny", edges)

    # 5 Passo - Melhorando bordas
    edges = morph_dilatation(edges)
    _, edges = cv.threshold(edges, 0, 1, cv.THRESH_BINARY)
    imgprint("edges", edges * 255)

    # Calculando o background e foreground da primeira segmentação
    background = frame_gray * (1 - edges)
    foreground = frame_gray * edges

    # Imprimindo Resultados
    imgprint("background", background)
    imgprint("forregound", foreground)

    # Fine-grained segmentation of Current image
    _, mask = cv.threshold(previus_background, T, 1, cv.THRESH_BINARY_INV)
    diff2 = cv.absdiff(foreground, previus_foreground * mask)
    _, diff2 = cv.threshold(diff2, T, 1, cv.THRESH_BINARY)
    background_fine = frame_gray * (1 - diff2)
    foreground_fine = frame_gray * diff2
    imgprint("edges", edges * 255)
    imgprint("Back melhor", background_fine)
    imgprint("Fore melhor", foreground_fine)

    # Self adaptative background
    # alfa = alfa_iluminacao(frame_gray, previus_frame)
    alfa = 0.1
    matriz[:, :] = (1 - alfa) * previus_background[:, :] + alfa * frame_gray[:, :]
    self_adap_background = self_background(binary_mask, matriz, previus_background)
    background = np.asarray(self_adap_background)
    imgprint("background10", background)

    # Contornos
    _, contornos, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        rect = cv.minAreaRect(contorno)
        box = cv.boxPoints(rect)
        box = np.uint0(box)
        frame_bgr = cv.drawContours(frame_bgr, [box], 0, (255, 0, 255), 3)
    imgprint("frame segmentado", frame_bgr)

    # Previus
    previus_frame = frame_gray[:, :]
    previus_background = background[:, :]
    previus_foreground = foreground[:, :]
    previous_bgr = frame_bgr[:, :]

    if cv.waitKey(0) & 0xFF == ord('s'):
        break


cap.release()
cv.destroyAllWindows()
