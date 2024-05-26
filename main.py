import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    sift = cv.SIFT.create()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    value_weight = 0.55

    images = []
    gray_imgs = []
    key_poins = []
    descriptors = []

    for img_name in os.listdir('data'):
        img = cv.cvtColor(cv.imread('data/' + img_name), cv.COLOR_BGR2RGB)
        images.append(img)

        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray_img, None)

        gray_imgs.append(gray_img)
        key_poins.append(kp)
        descriptors.append(des)

    # Juntando primeira e segunda imagem
    match_one_to_two = bf.match(descriptors[0], descriptors[1])

    src_pts = np.float32([key_poins[0][m.queryIdx].pt for m in match_one_to_two])
    dst_pts = np.float32([key_poins[1][m.trainIdx].pt for m in match_one_to_two])

    homografia_one, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    altura, largura = images[1].shape[:2]
    aligned_img_one = cv.warpPerspective(images[0], homografia_one, (largura, altura))
    res_one = cv.addWeighted(aligned_img_one, value_weight, images[1], value_weight, 0)

    # Juntando terceira imagem
    match_two_to_three = bf.match(descriptors[1], descriptors[2])

    src_pts = np.float32([key_poins[1][m.queryIdx].pt for m in match_two_to_three])
    dst_pts = np.float32([key_poins[2][m.trainIdx].pt for m in match_two_to_three])

    homografia_two, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    altura, largura = images[2].shape[:2]
    aligned_img_two = cv.warpPerspective(res_one, homografia_two, (largura, altura))
    res_two = cv.addWeighted(aligned_img_two, value_weight, images[2], value_weight, 0)

    last_img_resized = cv.resize(images[2], (largura, altura))
    final_result = cv.addWeighted(res_two, value_weight, last_img_resized, value_weight, 0)

    plt.imshow(final_result)
    plt.savefig('final_result.jpg', format='jpg')
    plt.show()
