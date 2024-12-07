#pip install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation

def show(img):
    plt.imshow(img)
    plt.show()

def compare(imgs=[]):
    if len(imgs) == 0:
        return
    
    f, ax_arr = plt.subplots(1,len(imgs))
    for i, img in enumerate(imgs):
        ax_arr[i].imshow(img)

    plt.show()

img_src = cv2.imread("./source_images/arm1.jpg")
img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

compare([img, img_src, hsv_img])

lower_skin_color_range = np.array([0, 48, 80], dtype=np.uint8)
upper_skin_color_range = np.array([20, 255, 255], dtype=np.uint8)


skin_mask = cv2.inRange(hsv_img ,lower_skin_color_range, upper_skin_color_range)
show(skin_mask)

skin_mask_filled = ndi.binary_fill_holes(skin_mask).astype(np.uint8) * 255

compare([skin_mask, skin_mask_filled])

skin_mask_filled = ndi.binary_fill_holes(skin_mask).astype(np.uint8) * 255
skin_region = cv2.bitwise_and(img, img, mask=skin_mask_filled)

# CENTER OF MASS
center_y, center_x = [np.average(indices) for indices in np.where(skin_mask >= 255)]

# CONSTANTS
AVG_ARM_SIZE = 9 #cm
SQUARE_WIDTH = 2.5 #cm

image_heigh, image_width, _ = img.shape

# Proporção do tamanho do braço em relação a imagem
ratio = (center_x * 2) / image_width
# Projeção do tamanho do quadrado a ser desenhado em relação ao tamanho do braço
square_size = int(((image_width * ratio) / AVG_ARM_SIZE) * SQUARE_WIDTH)

drawable_img = img.copy()

x_top_left = int(center_x - square_size / 2)
y_top_left = int(center_y - square_size / 2)
x_bottom_right = int(center_x + square_size / 2)
y_bottom_right = int(center_y + square_size / 2)

drawable_img = cv2.rectangle(
    drawable_img, 
    (x_top_left, y_top_left), 
    (x_bottom_right, y_bottom_right), 
    (0, 255, 0), 
    2
)
compare([img, drawable_img])
show(drawable_img)

square_mask = np.zeros(img.shape[:2], dtype=np.uint8)
square_mask = cv2.rectangle(
    square_mask,
    (x_top_left, y_top_left),
    (x_bottom_right, y_bottom_right),
    255,
    -1
)
show(square_mask)

selected_area = cv2.bitwise_and(img, img, mask=square_mask)
compare([img, selected_area])

blue, green, red = cv2.split(img)
show(red)

non_black_pixels = red[red > 0]
hist = cv2.calcHist([non_black_pixels], [0], None, [256], [0, 256])

plt.plot(hist, color='black')

red_channel_area = cv2.bitwise_and(red, red, mask=square_mask)
show(red_channel_area)

binary_mask = np.where((red > 10) & (red < 80), 255, 0).astype(np.uint8)
, 255, 0).astype(np.uint8)

red_with_mask = cv2.bitwise_and(red_channel_area, red_channel_area, mask=binary_mask)

show(red_with_mask)

final_img = cv2.bitwise_and(binary_mask, binary_mask, mask=square_mask)

contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = img.copy()

for contour in contours:
    # Ignorar contornos muito pequenos (ruídos)
    if cv2.contourArea(contour) >= 0:  # Ajuste o limite conforme necessário
        # Encontrar o círculo mínimo que circunscreve o contorno
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Desenhar o círculo na imagem
        cv2.circle(output_img, center, radius, (0, 255, 0), 5)  # Azul com espessura 2


show(output_img)

cv2.rectangle(
    output_img, 
    (x_top_left, y_top_left), 
    (x_bottom_right, y_bottom_right), 
    (0, 255, 0), 
    2
)

compare([img, output_img])

show(output_img)

show(img)