import os

import numpy as np
from PIL import Image, ImageDraw


def print_image(image_paths, accepted_pairs, filename):
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    x_offset = widths[0]

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (x_offset, 0))

    draw = ImageDraw.Draw(new_im)
    for pair in accepted_pairs:
        x1 = pair[0].coords[0]
        y1 = pair[0].coords[1]
        x2 = pair[1].coords[0] + x_offset
        y2 = pair[1].coords[1]
        color = tuple(np.random.randint(256, size=3))
        draw.line((x1, y1, x2, y2), fill=color)
    for pair in accepted_pairs:
        x1 = pair[0].coords[0]
        y1 = pair[0].coords[1]
        x2 = pair[1].coords[0] + x_offset
        y2 = pair[1].coords[1]
        color = tuple(np.random.randint(256, size=3))
        draw.line((x1, y1, x2, y2), fill=color)
    result_path = os.path.join(os.path.dirname(image_paths[0]), filename)
    new_im.save(result_path)


def print_all_image(image_paths, key_points_1, key_points_2, accepted_pairs, ransac_pairs, filename):
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    x_offset = images[0].size[0]

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (x_offset, 0))
    draw = ImageDraw.Draw(new_im)

    for point in key_points_1:
        add_key_point(draw, point, color='red')
    for point in key_points_2:
        add_key_point(draw, point, color='red', x_offset=x_offset)
    for pair in accepted_pairs:
        add_key_point(draw, pair[0], color='blue', scale=500)
        add_key_point(draw, pair[1], color='blue', x_offset=x_offset, scale=400)
    for pair in accepted_pairs:
        add_lines(draw, pair, x_offset, color='green')
    for pair in ransac_pairs:
        add_lines(draw, pair, x_offset, color='yellow')

    result_path = os.path.join(os.path.dirname(image_paths[0]), filename)
    new_im.save(result_path)


# def add_lines(image_draw, pair, x_offset, scale=200):
def add_lines(image_draw, pair, x_offset, color):
    x1 = pair[0].coords[0]
    y1 = pair[0].coords[1]
    x2 = pair[1].coords[0] + x_offset
    y2 = pair[1].coords[1]
    image_draw.line((x1, y1, x2, y2), fill=color)


def add_key_point(image_draw, point, color, x_offset=0, scale=10000):
    r = max(1, sum(image_draw.im.size) // 2 // scale)
    add_circle(image_draw=image_draw, x=point.coords[0] + x_offset, y=point.coords[1], radius=r, color=color)


def add_circle(image_draw, x, y, radius, color):
    image_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
