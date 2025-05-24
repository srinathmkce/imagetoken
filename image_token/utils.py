from PIL import Image


def read_image_dims(path: str) -> tuple[int, int]:
    img = Image.open(path)
    width, height = img.size
    return width, height
