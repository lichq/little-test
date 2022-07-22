import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

def mattranspose(mat):
    mat[0][-1] = mat[0][-1] % 5
    mat[1][-1] = mat[1][-1] % 5
    print(mat)
    return mat
    
def scale(Intrin_Mat, Extrin_Mat, height, width, filename):
    inv_Intrin_Mat = np.linalg.inv(Intrin_Mat)
    inv_Extrin_Mat = np.linalg.inv(Extrin_Mat)
    
    img = np.zeros((512,512))
    dep = 5
    for i in range(0, height):
        for j in range(0, width):
            idx = np.array([ i, j , 1])
            # print(idx)
            [u, v, w] = np.dot(inv_Intrin_Mat, idx) * dep
            point  = np.array([ u, v , w, 1])
            [x, y, z, _] = np.dot(Extrin_Mat, point)
            if x < 0:
                print(x)
            # print(x, y)
            # x = x % 5
            # y = y % 5
            while(x >= 2.5):
                x = x - 5
            while (x <= -2.5):
                x = x + 5
            while(y >= 2.5):
                y = y - 5
            while (y <= -2.5):
                y = y + 5
            if -0.5 < x < 0.5 and -0.5 < y < 0.5 :
                img[i][j] = 210
            else:
                [u, v, w] = np.dot(inv_Intrin_Mat, idx) * (dep+5)
                point = np.array([ u, v , w, 1])
                [x, y, z, _] = np.dot(Extrin_Mat, point)
                # print(z)
                while(x >= 2.5):
                    x = x - 5
                while (x <= -2.5):
                    x = x + 5
                while(y >= 2.5):
                    y = y - 5
                while (y <= -2.5):
                    y = y + 5
                if -2  < x < 2   and -2  < y < 2 :
                    img[i][j] = 128
                else:
                    img[i][j] = 50

    im = Image.fromarray(img)
    im = im.convert('L')
    # im.save(filename)   
    im.show(filename)

Extrin_Mat = np.array([[1, 0, 0, 3.5],
                        [0,1,0, 3.5],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

Intrin_Mat = np.array([[1000, 0, 256],
                        [0, 1000, 256],
                        [0, 0, 1]])

# Extrin_Mat = mattranspose(Extrin_Mat)

scale(Intrin_Mat, Extrin_Mat, 512, 512, "try111.png")
# for sc in np.arange(-0.1, 0.1, 0.01):
#     movex = round(sc, 2)
#     movey = round(movex + 0.05, 2)

#     scale(1, movex, movey, 'onlymove x{} y{}.png' .format(movex, movey))
