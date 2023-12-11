from pygame import image as pyg_image

SPRITES_PATH = "./sprites"

def load_sprite(filename, convert=True, alpha=True):
    img = pyg_image.load(f"{SPRITES_PATH}/{filename}")
    return img.convert_alpha() if convert and alpha else img.convert() if convert else img