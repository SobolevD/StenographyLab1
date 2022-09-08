import cv2
import numpy as np

VAR = 13
DELTA = (4 + 4 * VAR) % 3


def get_channel(image, channel_num):
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
    blue_channel = get_channel(baboon_image, 1)

    # 2. Занулим 4-й битовый слой (сделаем белым), чтобы на него нанести миккимауса. Для этого умножим битово картинку
    # на число, у которого 4-й бит = 0: 1111 0111 = 247
    blue_channel_4_plate_filled = blue_channel & 247  # Занулим, так как 247 = 1111 0111, т.е. 4-й слой будет нулевым

    # 3. Поскольку модифицируется 4-й слой, то миккимаус должен быть матрицей, в которой есть только числа 0 и 8 (8 = 0000 1000)
    # 0 - белый пиксель, 8 - черный пиксель. Тогда при наложении микки на 4-й слой картинка 4-го слоя станет миккимаусом (побитовое сложение)
    mickey_image = ((cv2.imread('ornament.tif') / 255) * 8).astype(np.uint8)

    mickey_image = get_channel(mickey_image, 3)

    # 4. Наложение микки мауса на 4-й слой путем побитового сложения ("на белый лист кладем форму миккимауса")
    svi_1_result_blue = blue_channel_4_plate_filled | mickey_image

    r = get_channel(baboon_image, 3)
    g = get_channel(baboon_image, 2)

    # 5. Сольем каналы обратно
    svi_1_result = cv2.merge([svi_1_result_blue, g, r])

    # СВИ-1 - декодирование (для проверки того, что мы получили реального микимауса,
    # можно попробовать на 8-й битовой плоскости. Т.е. 247 заменить на 127, а 8 на 128
    svi_1_result_blue_channel = get_channel(svi_1_result, 1)
    svi_1_decoded = get_plane(svi_1_result_blue_channel, 4)

    cv2.imshow("Original", baboon_image)
    cv2.imshow("SVI-1 Encoded", svi_1_result)
    cv2.imshow("SVI-1 Decoded", svi_1_decoded)

    cv2.waitKey(0)

    # СВИ-4
    baboon_image = cv2.imread('baboon.tif')

    noise = np.empty((512, 512), dtype="uint8")
    cv2.randn(noise, 0, DELTA - 1)

    cv2.imshow("Original", noise)

    green_channel = get_channel(baboon_image, 2)

    mickey_image = cv2.imread('ornament.tif')
    mickey_image = get_channel(mickey_image, 2)

    changed_green_channel = (green_channel // (2 * DELTA) * (2 * DELTA)) + mickey_image * DELTA + noise

    cv2.imshow("Result", changed_green_channel)

    r = get_channel(baboon_image, 3)
    b = get_channel(baboon_image, 1)

    svi_4_result = cv2.merge([b, changed_green_channel, r])

    cv2.imshow("Result", svi_4_result)

    cv2.waitKey(0)

    # СВИ-4 декодирование
    svi_4_result_green = get_channel(svi_4_result, 2)
    baboon_image_green = get_channel(baboon_image, 2)

    svi_4_decoded = (svi_4_result_green - noise - (baboon_image_green // (2 * DELTA) * 2 * DELTA)) / DELTA
    cv2.imshow("Result SVI-4", svi_4_decoded)

    cv2.waitKey(0)


