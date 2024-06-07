def get_image(data_batch, index):
    return data_batch['data'][index]


def get_y_value(image):
    image = image.reshape(3, 32, 32).transpose(1, 2, 0)
    asd = []
    for i in range(32):
        for j in range(32):
            y = ((299 * image[i][j][0]) + (587 * image[i][j][1]) + (114 * image[i][j][2])) / 1000
            asd.append(y)
    return asd


def get_h_value(image):
    image = image.reshape(3, 32, 32).transpose(1, 2, 0)
    asd = []
    for i in range(32):
        for j in range(32):
            pixel = image[i][j] / 255.0
            max_val = max(pixel)
            min_val = min(pixel)
            diff = max_val - min_val
            if max_val == min_val:
                asd.append(0)

            elif max_val == pixel[0]:
                asd.append((60 * ((pixel[1] - pixel[2]) / diff) + 360) % 360)

            elif max_val == pixel[1]:
                asd.append((60 * ((pixel[2] - pixel[0]) / diff) + 120) % 360)

            elif max_val == pixel[2]:
                asd.append((60 * ((pixel[0] - pixel[1]) / diff) + 240) % 360)
    return asd
