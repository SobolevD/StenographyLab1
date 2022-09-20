import cv2
import numpy as np

VAR = 13
DELTA = (4 + 4 * VAR) % 3


def get_channel(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'blue':
        return b
    if channel == 'green':
        return g
    if channel == 'red':
        return r


def get_plane(channel_image, plane_num):
    return channel_image & (2 ** (plane_num - 1))


def svi_1_encode(original_image, watermark, color_channel, bit_plate_num):
    num_for_clear_bit_plate = 255 - (2 ** (bit_plate_num - 1))

    prepared_watermark_colored = ((watermark / 255) * (2 ** (bit_plate_num - 1))).astype(np.uint8)
    single_channel_watermark = get_channel(prepared_watermark_colored, color_channel)

    channel_with_empty_bit_plate = get_channel(original_image, color_channel) & num_for_clear_bit_plate

    channel_result = channel_with_empty_bit_plate | single_channel_watermark

    cv2.imshow(" Channel result", channel_result)

    r = get_channel(baboon_image, 'red')
    g = get_channel(baboon_image, 'green')
    b = get_channel(baboon_image, 'blue')

    # 5. Сольем каналы обратно
    if color_channel == 'blue':
        return cv2.merge([channel_result, g, r])
    if color_channel == 'red':
        return cv2.merge([b, g, channel_result])
    if color_channel == 'green':
        return cv2.merge([b, channel_result, r])


def svi_1_decode(encoded_image, color_channel, bit_plate_num):
    encoded_image_channel = get_channel(encoded_image, color_channel)
    return get_plane(encoded_image_channel, bit_plate_num)


def svi_4_encode(original_image, watermark, color_channel, delta):
    height, width, channels = original_image.shape
    noise = (np.round(np.random.uniform(0.0, DELTA - 1, (height, width)))).astype("uint8")

    noise_to_show = noise > 0.5
    noise_to_show = (noise_to_show * 255).astype(np.uint8)

    cv2.imshow("Noise", noise_to_show)

    extracted_channel = get_channel(original_image, color_channel)
    binary_watermark = get_channel(watermark, color_channel)
    changed_channel = (extracted_channel // (2 * delta) * (2 * delta)) + binary_watermark * delta + noise

    cv2.imshow(" Channel result", changed_channel)
    r = get_channel(original_image, 'red')
    g = get_channel(original_image, 'green')
    b = get_channel(original_image, 'blue')

    if color_channel == 'blue':
        return noise, cv2.merge([changed_channel, g, r])
    if color_channel == 'red':
        return noise, cv2.merge([b, g, changed_channel])
    if color_channel == 'green':
        return noise, cv2.merge([b, changed_channel, r])


def svi_4_decode(encode_image, original_image, noise, color_channel, delta):
    encoded_image_channel = get_channel(encode_image, color_channel)
    original_image_channel = get_channel(original_image, color_channel)
    return (encoded_image_channel - noise - (original_image_channel // (2 * delta) * 2 * delta)) / delta


if __name__ == '__main__':
    baboon_image = cv2.imread('baboon.tif')
    ornament = cv2.imread('ornament.tif')

    # СВИ-1
    svi_1_result = svi_1_encode(baboon_image, ornament, 'blue', 4)
    svi_1_decode = svi_1_decode(svi_1_result, 'blue', 4)

    # пороговая обработка
    svi_1_decode = svi_1_decode > 7
    svi_1_decode = (svi_1_decode * 255).astype(np.uint8)

    cv2.imshow("Original", baboon_image)
    cv2.imshow("SVI-1 Encoded", svi_1_result)
    cv2.imshow("SVI-1 Decoded", svi_1_decode)

    cv2.waitKey(0)

    # СВИ-4
    result_noise, svi_4_result = svi_4_encode(baboon_image, ornament, 'green', DELTA)
    svi_4_decode = svi_4_decode(svi_4_result, baboon_image, result_noise, 'green', DELTA)

    cv2.imshow("Original", baboon_image)
    cv2.imshow("SVI-4 Encoded", svi_4_result)
    cv2.imshow("SVI-4 Decoded", svi_4_decode)

    cv2.waitKey(0)
