import math
from mandelbrot import complex_matrix,complex_matrix_to_data,distinct_colors,color_list
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

class julia:
    def __init__(self,corner1 = -2 + 2j,corner2 = 2-2j,x_num=1000,y_num=1000,explosion_boundary = 2,power = 2,c = 0.5 + 0.355j):
        self.corners = (corner1,corner2)
        self.size = (x_num,y_num)
        self.power = power

        self.matrix = complex_matrix(corner1,corner2,x_num,y_num)
        self.mask = np.ones_like(self.matrix,dtype = bool)
        self.counter = 0
        self.explosion_boundary = explosion_boundary
        self.c = c

    def julia_validity(self,z):
        return np.abs(z)<self.explosion_boundary

    def run(self,iter):
        z = self.matrix[:]
        cs = np.ones_like(z)*self.c
        count = np.zeros_like(z)
        for _ in range(iter):
            z[self.mask] = z[self.mask]**self.power + cs[self.mask]
            self.mask = self.julia_validity(z)
            count[self.mask] = count[self.mask] + 1
        self.counter = count.real
    def reset(self):
        self.mask = np.ones_like(self.matrix)
        self.counter *= 0

    def plot_julia(self):
        X,Y = complex_matrix_to_data(self.matrix*self.mask)
        plt.scatter(X,Y)
        plt.axis('equal')
        plt.show()

    #julia set image
    def image(self,color = None,colorBG=None):
        if color is None:
            color = (255,255,255)
        if colorBG is None:
            colorBG = (0,0,0)
        h, b = self.matrix.shape
        img = Image.new(mode="RGB", size=(b,h), color=colorBG)
        max_iter = np.max(self.counter)

        for i in range(h):
            for j in range(b):
                    ratio = self.counter[i][j]/max_iter
                    col_temp = tuple(int(x*ratio) for x in color)
                    img.putpixel((j,i),col_temp)
        return img

    def divergence_image(self,color_n = 10,global_max = None):
        h, b = self.matrix.shape
        img = Image.new(mode="RGB", size=(b,h), color=(0,0,0))
        max_iter = np.max(self.counter) if global_max is None else global_max

        for i in range(h):
            for j in range(b):
                    ratio = self.counter[i][j]/(max_iter+1)
                    #print(ratio)
                    col_temp =  tuple(color_list[math.floor(ratio*color_n)])
                    img.putpixel((j,i),col_temp)
        return img

    #generates the julia set in batches rather than making it all at once
    def stitch_image(self,patch_dim = 100 , patch_num_x = 5 , patch_num_y = 5,iter_per_patch = 200,color = None,colorBG = None):
        super_img = Image.new(mode="RGB", size=(patch_num_x * patch_dim, patch_num_y* patch_dim), color=(0, 0, 0))
        boundary_vertices = complex_matrix(self.corners[0],self.corners[1],x_num=patch_num_x+1,y_num=patch_num_y+1)
        indx = 0
        total = patch_num_x*patch_num_y
        for i in range(patch_num_x):
            for j in range(patch_num_y):
                c1,c2 = boundary_vertices[j][i],boundary_vertices[j+1][i+1]
                temp = julia(c1,c2,x_num=patch_dim,y_num=patch_dim,power=self.power,explosion_boundary=self.explosion_boundary,c = self.c)
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
            temp = julia(corner1,corner2,x_num=frame_x_num,y_num=frame_y_num,power=self.power,explosion_boundary=self.explosion_boundary,c = self.c)
            temp.run(iter = frame_iter)
            f = temp.image(color = color,colorBG = colorBG)
            frames.append(f)
            del temp
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
            #print(_)
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
            temp = julia(corner1, corner2, power=self.power,
                              explosion_boundary=self.explosion_boundary,c = self.c)
            f = temp.stitch_image(patch_dim = patch_dim,patch_num_x = patch_num_x , patch_num_y = patch_num_y , iter_per_patch = frame_iter,color = color,colorBG = colorBG)
            frames.append(f)
            del temp
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
            #print(_)

        return frames

    #WORKING BUT THE CONTRAST IS NOT THAT CLEAR
    def stitch_image_divergence(self,patch_dim = 100 , patch_num_x = 5 , patch_num_y = 5,
                                iter_per_patch = 200, color_n = 50):
        super_img = Image.new(mode="RGB", size=(patch_num_x * patch_dim, patch_num_y * patch_dim), color=(0, 0, 0))
        boundary_vertices = complex_matrix(self.corners[0], self.corners[1], x_num=patch_num_x+1, y_num=patch_num_y+1)
        indx = 0
        total = patch_num_x * patch_num_y
        for i in range(patch_num_x):
            for j in range(patch_num_y):
                c1, c2 = boundary_vertices[j][i], boundary_vertices[j+1][i+1]
                temp = julia(c1, c2, x_num=patch_dim, y_num=patch_dim,
                             power=self.power, explosion_boundary=self.explosion_boundary, c=self.c)
                temp.run(iter=iter_per_patch)
                temp_img = temp.divergence_image(color_n=color_n,global_max = iter_per_patch)
                super_img.paste(temp_img, (i*patch_dim, j*patch_dim))
                del temp, temp_img
                sys.stdout.write(f'\rProgress: {indx*100/total:.2f}%')
                sys.stdout.flush()
                indx += 1
        return super_img

    # zoom sequence with divergence_image
    def zoom_sequence_divergence(self, n_frames=100, zoom_center=0, zoom_per_frame=0.95,
                                 frame_iter=100, frame_x_num=500, frame_y_num=500,
                                 corner1=None, corner2=None, color_n = 50):
        if corner1 is None:
            corner1 = self.corners[0]
        if corner2 is None:
            corner2 = self.corners[1]

        frames = []
        for _ in range(n_frames):
            temp = julia(corner1, corner2, x_num=frame_x_num, y_num=frame_y_num,
                         power=self.power, explosion_boundary=self.explosion_boundary, c=self.c)
            temp.run(iter=frame_iter)
            f = temp.divergence_image(color_n=color_n)
            frames.append(f)
            del temp
            corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
            corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
        return frames

    #NOT WORKING UNLESS STITCH PATCHED STARTS WORKING
    # patched zoom sequence with divergence_image
    # def zoom_sequence_patched_divergence(self, n_frames=100, zoom_center=0, zoom_per_frame=0.95,
    #                                      frame_iter=100, patch_dim=100, patch_num_x=5, patch_num_y=5,
    #                                      corner1=None, corner2=None, color_n = 50):
    #     if corner1 is None:
    #         corner1 = self.corners[0]
    #     if corner2 is None:
    #         corner2 = self.corners[1]
    #
    #     frames = []
    #     for _ in range(n_frames):
    #         temp = julia(corner1, corner2, power=self.power,
    #                      explosion_boundary=self.explosion_boundary, c=self.c)
    #         f = temp.stitch_image_divergence(patch_dim=patch_dim, patch_num_x=patch_num_x,
    #                                          patch_num_y=patch_num_y, iter_per_patch=frame_iter,
    #                                          color_n=color_n)
    #         frames.append(f)
    #         del temp
    #         corner1 = (corner1 - zoom_center) * zoom_per_frame + zoom_center
    #         corner2 = (corner2 - zoom_center) * zoom_per_frame + zoom_center
    #     return frames



