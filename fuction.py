import cv2 as cv
import numpy as np

n = 0.5


def imgprint(name, img):
    cv.imshow(name, img)


def alfa_iluminacao(frame, previus):
    """
    Calcula a parcela do alfa que varia de acordo a luminosidade
    :param frame: Frame atual
    :param previus: Frame anterios
    :return: Parcela do alfa que varia de acordo com a luminosidade
    """

    mean1 = np.mean(frame)
    mean2 = np.mean(previus)
    max_mean = max(mean1, mean2)
    alfa = n * (1 - abs(mean1 - mean2) / max_mean)
    return alfa


def preencher_interior(img):
    h, w = img.shape
    seed = (w / 2, h / 2)

    mask = np.zeros((h + 2, w + 2), np.uint8)

    floodflags = 4
    floodflags |= cv.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    num, im, mask, rect = cv.floodFill(img, mask, seed, (255, 0, 0), (10,) * 3, (10,) * 3, floodflags)

    cv.imwrite("seagull_flood.png", mask)
    return mask


def morph_dilatation(img):
    # Rect Kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    # imgprint(img)
    # img = cv.dilate(img, kernel_dilate, iterations=3)
    # img = cv.erode(img, kernel_close, iterations=3)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_close)
    img = cv.GaussianBlur(img, (9, 9), 0)
    # imgprint(img)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # imgprint(img)
    return img


'''
def morph_dilatation(img):
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    #noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    #sure background
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    #markers[unknown == 255] = 0

    # markers = cv.watershed(img, markers)
    #img[markers == -1] = [255, 0, 0]
    cv.imshow("TEEEEEESTE0", img)
    return img
'''

def filtro_cinza(img):
    linhas , colunas , tipo = img.shape
    binary_mask = np.zeros((linhas, colunas))
    intervalo = 17
    for i in range(0, linhas):
        for j in range(0, colunas):
            valor = img[i, j, 0]
            if valor - intervalo < img[i, j, 1] < valor + intervalo:
                if valor - intervalo < img[i, j, 2] < valor + intervalo:
                    binary_mask[i, j] = 255
    imgprint("bus", binary_mask)
    return binary_mask


def estrada(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = img[:, :]
    canny = cv.Canny(img,50, 150, apertureSize = 3)
    lines = cv.HoughLines(canny, 1, np.pi/180, 200, )
    minLineLength = 10
    maxLineGap = 100
    lines2 = cv.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for x1, y1, x2, y2 in lines2[0]:
        cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("imagem", img2)
    cv.waitKey(0)
    return img


if __name__ == "__main__":
    img = cv.imread("estrada.png")
