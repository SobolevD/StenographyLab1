import cv2
import numpy as np


def get_channel_array(image, channel_num):
    b, g, r = cv2.split(image)
    if channel_num == 1:
        return b
    if channel_num == 2:
        return g
    if channel_num == 3:
        return r


def get_plane(channel_image, plane_num):
    return channel_image & (2 ** (plane_num - 1))


def convert_to_binary(plane_raw, plane_num):
    return plane_raw >> (plane_num - 1)

if __name__ == '__main__':

    baboon_image = cv2.imread('baboon.tif')

    # 1. Вычленим синий канал
    blue_channel = get_channel_array(baboon_image, 1)

    # 2. Занулим 4-й битовый слой (сделаем белым), чтобы на него нанести миккимауса. Для этого умножим битово картинку
    # на число, у которого 4-й бит = 0: 1111 0111 = 247
    blue_channel_4_plate_filled = blue_channel & 247 # Занулим, так как 247 = 1111 0111, т.е. 4-й слой будет нулевым

    # 3. Поскольку модифицируется 4-й слой, то миккимаус должен быть матрицей, в которой есть только числа 0 и 8 (8 = 0000 1000)
    # 0 - белый пиксель, 8 - черный пиксель. Тогда при наложении микки на 4-й слой картинка 4-го слоя станет миккимаусом (побитовое сложение)
    mickey_image = ((cv2.imread('mickey.tif') / 255) * 8).astype(np.uint8)

    mickey_image = get_channel_array(mickey_image, 3)

    # 4. Наложение микки мауса на 4-й слой путем побитового сложения ("на белый лист кладем форму миккимауса")
    result = blue_channel_4_plate_filled | mickey_image

    r = get_channel_array(baboon_image, 3)
    g = get_channel_array(baboon_image, 2)

    # 5. Сольем каналы обратно
    final_result = cv2.merge([result, g, r])

    cv2.imshow("Original", baboon_image)
    cv2.imshow("With info", final_result)

    cv2.waitKey(0)

