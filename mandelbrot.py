import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
import numba as nb
import sys
import math
import pickle



with open("color_list.pkl", "rb") as f:
    color_list = pickle.load(f)
def distinct_colors(n):
    return color_list[0:n]


def complex_matrix(corner1=-1 + 1j, corner2=1 - 1j, x_num=100, y_num=100):
    x_s = np.linspace(corner1.real, corner2.real, num=x_num)
    y_s = np.linspace(corner1.imag, corner2.imag, num=y_num)
    R, I = np.meshgrid(x_s, y_s)
    return R + I * 1j


@nb.njit(parallel=True)
def complex_matrix_to_data(matrix):
    return matrix.real.ravel(), matrix.imag.ravel()

class mandelbrot():
    def __init__(self,corner1=-1.5 + 1.5*1j, corner2=1.5 - 1.5*1j, x_num=1000, y_num=1000,explosion_boundary = 2,power = 2):
        self.matrix = complex_matrix(corner1, corner2, x_num, y_num)
        self.power = power

        self.explosion_boundary = explosion_boundary
        self.mask = np.ones_like(self.matrix,dtype = bool)
        self.corners = (corner1 , corner2)
        self.counter = np.zeros_like(self.matrix,dtype = int)
    def mandelbrot_validity(self,z):
        return np.abs(z)<self.explosion_boundary

    def run(self,iter):
        z = np.zeros_like(self.matrix)
        c  = self.matrix[:]
        count = np.zeros_like(self.matrix)
        for _ in range(iter):

            z[self.mask] = z[self.mask]**self.power + c[self.mask]
            self.mask = self.mandelbrot_validity(z)
            count[self.mask] = count[self.mask] + 1
        self.counter = count
    def reset(self):
        self.mask = np.ones_like(self.matrix,dtype = bool)
    def plot_mandelbrot(self):
        X,Y = complex_matrix_to_data(self.matrix*self.mask)
        plt.scatter(X,Y)
        plt.axis('equal')
        plt.show()

    #Saves the mandelbrot
    def image(self,color = None,colorBG=None):
        if color is None:
            color = (255,255,255)
        if colorBG is None:
            colorBG = (0,0,0)
        h, b = self.matrix.shape
        img = Image.new(mode="RGB", size=(b,h), color=colorBG)
        for i in range(h):
            for j in range(b):
                if (self.mask[i][j])    :
                    img.putpixel((j,i),color)
        return img

    def divergence_image(self,color_n = 30):
        h, b = self.matrix.shape
        img = Image.new(mode="RGB", size=(b, h), color=(0, 0, 0))
        max_iter = np.max(self.counter)

        for i in range(h):
            for j in range(b):
                ratio = self.counter[i][j] / (max_iter + 1)
                #print(ratio)
                col_temp = tuple(color_list[math.floor(ratio * color_n)])
                img.putpixel((j, i), col_temp)
        return img

    #generates the mandelbrot in batches rather than making it all at once
    def stitch_image(self,patch_dim = 100 , patch_num_x = 5 , patch_num_y = 5,iter_per_patch = 200,color = None,colorBG = None):
        super_img = Image.new(mode="RGB", size=(patch_num_x * patch_dim, patch_num_y* patch_dim), color=(0, 0, 0))
        boundary_vertices = complex_matrix(self.corners[0],self.corners[1],x_num=patch_num_x+1,y_num=patch_num_y+1)
        indx = 0
        total = patch_num_x*patch_num_y
        for i in range(patch_num_x):
            for j in range(patch_num_y):
                c1,c2 = boundary_vertices[j][i],boundary_vertices[j+1][i+1]
                temp = mandelbrot(c1,c2,x_num=patch_dim,y_num=patch_dim,power=self.power,explosion_boundary=self.explosion_boundary)
                temp.run(iter=iter_per_patch)
                temp = temp.image(color = color,colorBG = colorBG)
                super_img.paste(temp,(i*patch_dim,j*patch_dim))
                del temp
                sys.stdout.write(f'\rProgress: {indx*100/total}%')
                sys.stdout.flush()
                indx += 1
        return  super_img

    def zoom_sequence(self,n_frames = 100,zoom_center = 0,zoom_per_frame = 0.95,
                 frame_iter = 100,frame_x_num = 500, frame_y_num = 500,corner1=-1+1j, corner2=1-1j,
                      color = None,colorBG = None):
        if corner1 is None:
            corner1 = self.corners[0]
        if corner2 is None:
            corner2 = self.corners[1]

        frames = []

        for _ in range(n_frames):
            temp = mandelbrot(corner1,corner2,x_num=frame_x_num,y_num=frame_y_num,power=self.power,explosion_boundary=self.explosion_boundary)
            temp.run(iter = frame_iter)
            f = temp.image(color = color,colorBG = colorBG)
            frames.append(f)
            del temp
            sys.stdout.write(f"\rProgress: {_ * 100 / n_frames:.2f}%")
            sys.stdout.flush()
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
            #print(_)
        return frames

    def zoom_sequence_divergence(self, n_frames=100, zoom_center=0, zoom_per_frame=0.95,
                                 frame_iter=100, frame_x_num=500, frame_y_num=500,
                                 corner1=None, corner2=None, color_n=30):
        if corner1 is None:
            corner1 = self.corners[0]
        if corner2 is None:
            corner2 = self.corners[1]

        frames = []

        for _ in range(n_frames):
            temp = mandelbrot(corner1, corner2, x_num=frame_x_num, y_num=frame_y_num,
                              power=self.power, explosion_boundary=self.explosion_boundary)
            temp.run(iter=frame_iter)
            f = temp.divergence_image(color_n=color_n)
            frames.append(f)
            del temp
            sys.stdout.write(f"\rProgress: {_ * 100 / n_frames:.2f}%")
            sys.stdout.flush()
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center

        return frames

    def zoom_sequence_patched(self, n_frames=100, zoom_center=0, zoom_per_frame=0.95,
                          frame_iter=100, patch_dim = 100,patch_num_x = 5,patch_num_y = 5, corner1=None, corner2=None,
                              color = None,colorBG = None):
        if corner1 is None:
            corner1 = self.corners[0]
        if corner2 is None:
            corner2 = self.corners[1]

        frames = []
        for _ in range(n_frames):
            temp = mandelbrot(corner1, corner2, power=self.power,
                              explosion_boundary=self.explosion_boundary)
            f = temp.stitch_image(patch_dim = patch_dim,patch_num_x = patch_num_x , patch_num_y = patch_num_y , iter_per_patch = frame_iter,color = color,colorBG = colorBG)
            frames.append(f)
            del temp
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
            #print(_)

        return frames
    




