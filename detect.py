import cv2
import numpy as np
def detect(image_path):
    # Загрузка изображения сварки
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение фильтра Гаусса для сглаживания
    blurred_image = cv2.GaussianBlur(gray_image, (1, 1), 0)

    # Применение алгоритма Кэнни для обнаружения границ
    edges = cv2.Canny(blurred_image, 50, 150)

    # Нахождение контуров на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отбор контуров, которые могут быть сварочным швом
    welding_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2:  # Пороговое значение площади (настройте под свою задачу)
            welding_contours.append(contour)

    Cord=[]

    # Рисование прямоугольников вокруг сварочных швов
    for contour in welding_contours:
        x, y, w, h = cv2.boundingRect(contour)
        Cord.append(x)
        Cord.append(y)

    sumx=0
    sumy=0

    for i in range(0,len(Cord)):
        if i%2!=0:
            sumx+=Cord[i]
        else:
            sumy+=Cord[i]
    ff = ("x = " + str(sumx//(len(Cord)//2)) + ", y = " + str(sumy//(len(Cord)//2)))
    return ff