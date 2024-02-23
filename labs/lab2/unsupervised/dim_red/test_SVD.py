import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from temp_SVD import SVD

face_img = imread("../../imgs/personal/Moises_Guerrero.jpeg")
resized_img = resize(face_img, (256,256,3))
grayscale_img = rgb2gray(resized_img)

# plt.imshow(grayscale_img, cmap="gray")
# plt.axis('off')
# plt.show()

svd_test = SVD(nsing_vals=256)
transformed_img = svd_test.fit_transform(grayscale_img)

# print(np.abs(svd_test.U))
# print("-"*20)
# print(np.angle(svd_test.U))
# print("-"*20)
# print(np.abs(svd_test.U) * np.cos(np.angle(svd_test.U)))

plt.imshow(transformed_img, cmap="gray")
plt.axis('off')
plt.show()

# np.savetxt("iscomplex.txt", np.angle(svd_test.U), delimiter='|')

# print("-"*20)

# U, S, Vt = np.linalg.svd(grayscale_img)
# print(U)