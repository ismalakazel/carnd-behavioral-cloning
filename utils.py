from PIL import Image
from random import randint
from keras.preprocessing import image as keras_image

def open_image(path):
    return Image.open(path)

def to_keras(image):
    return keras_image.img_to_array(image)

def crop(image):
    LEFT = 60
    TOP = 47
    WIDTH = 320
    HEIGHT = 160
    box = (LEFT, TOP, WIDTH-LEFT, HEIGHT-TOP)
    return image.crop(box)

def resize(image):
    WIDTH = 200
    HEIGHT = 66
    return image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

def translate(image):
    rand1 = randint(-10, 10)
    rand2 = randint(-10, 10)
    return image.rotate(0, translate=(rand1, rand2))

def brigthness(image, constant=randint(1,3)):
    source = image.split()

    R, G, B = 0, 1, 2

    Red = source[R].point(lambda i: i/constant)
    Green = source[G].point(lambda i: i/constant)
    Blue = source[B].point(lambda i: i/constant)

    image = Image.merge(image.mode, (Red, Green, Blue))
    return image

def pre_process(image):
    image = crop(image)
    image = resize(image)
    return image

def augment(image):
    image = translate(image)
    image = brigthness(image, randint(1, 3))
    return image

