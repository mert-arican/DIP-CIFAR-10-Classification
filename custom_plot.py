import matplotlib.pyplot as plt

ii = 0


def add_image_to_plot(image, title, dimension, rows, cols, index, fig, is_gray=False):
    image = image.reshape(dimension, 32, 32).transpose(1, 2, 0)
    fig.add_subplot(rows, cols, index)
    plt.imshow(image) if not is_gray else plt.imshow(image, cmap='gray')
    titles = title.split('\n')
    plt.title(f'{titles[0]}'.capitalize())
    if len(titles) > 1:
        plt.xlabel(f'{titles[1]}')


def write_plots_to_file(test_class, images, width, height, rows, cols, titles=None, title=None, filename=None):
    global ii
    titles_received = titles is not None
    rows, cols = rows, cols
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)
    for i in range(1, cols*rows + 1):
        x = int(len(images[i-1])/(width*height))
        add_image_to_plot(images[i-1], titles[i-1] if titles_received else i-1, x, rows, cols, i, fig, x == 1)
    ii += 1
    plt.savefig(f'/Users/mertarican/Developer/Python/{test_class}/{ii if filename is None else filename}.jpg')
    plt.close(fig)
