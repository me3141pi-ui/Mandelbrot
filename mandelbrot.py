import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numba as nb
import sys
import math

#FUNCTION TO SAVE A LIST OF PIL IMAGES AS A GIF
def save_images_as_gif(images, output_path, duration=30, loop=0):
    if not images:
        raise ValueError("The images list is empty.")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )

#FUNCTION TO CREATE A GRADIENT MAP OF VIBRANT COLORS FOR VISUALISATION OF DIVERGENCE IMAGE
def vibrant_colormap(n=256):
    x = np.linspace(0, 1, n)
    hues = (0.9 * np.sin(2 * np.pi * x) + 1) / 2
    hues = (hues + x) % 1
    saturation = 0.9 * np.ones_like(x)
    value = 0.9 + 0.1 * np.sin(4 * np.pi * x)
    colors = []
    for h, s, v in zip(hues, saturation, value):
        colors.append(tuple(int(c * 255) for c in plt.cm.hsv(h % 1)[:3]))
    return np.array(colors)
#CREATING A 256 COLOR LIST
color_list = vibrant_colormap(256)
#RETURNS A COMPLEX NUMBER MATRIX
def complex_matrix(corner1=-1 + 1j, corner2=1 - 1j, x_num=100, y_num=100):
    x_s = np.linspace(corner1.real, corner2.real, num=x_num)
    y_s = np.linspace(corner1.imag, corner2.imag, num=y_num)
    R, I = np.meshgrid(x_s, y_s)
    return R + I * 1j


@nb.njit(parallel=True)
#SEPERATES THE COMPLEX MATRIX INTO REAL AND IMAG PARTS
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

    #CHECKS WHETHER THE POINT HAS ESCAPED THE EXPLOSION BOUNDARY
    def mandelbrot_validity(self,z):
        return np.abs(z)<self.explosion_boundary

    #RUNS THE COMPLEX PLANE THROUGH THE ITERATIVE SEQUENCE
    #GENERATES THE MANDELBROT MASK (BOOLEAN MATRIX WHERE 1-> NON ESCAPING POINT ,0-> ESCAPING POINT)
    #GENERATES THE COUNTER MATRIX (COUNTS THE NUMBER OF ITERATIONS AFTER WHICH THE POINT HAS ESCAPED)
    def run(self,iter):
        z = np.zeros_like(self.matrix)
        c  = self.matrix[:]
        count = np.zeros_like(self.matrix)
        for _ in range(iter):

            z[self.mask] = z[self.mask]**self.power + c[self.mask]
            self.mask = self.mandelbrot_validity(z)
            count[self.mask] = count[self.mask] + 1
        self.counter = count

    #RESETS MASK AND COUNTER
    def reset(self):
        self.mask = np.ones_like(self.matrix,dtype = bool)
        self.counter = np.zeros_like(self.matrix,dtype = int)

    #PLOTS THE MANDELBROT IN MATPLOTLIB
    def plot_mandelbrot(self):
        X,Y = complex_matrix_to_data(self.matrix*self.mask)
        plt.scatter(X,Y)
        plt.axis('equal')
        plt.show()

    #CREATES A BINARY IMAGE OF THE MANDELBROT SET CREATED DURING RUN()
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

    #CREATES A VISUALLY STUNNING DIVERGENCE IMAGE OF THE MANDELBROT SET , WHERE THE COLOR OF EACH POINT INDICATES HOW FAST IT DIVERGES
    def divergence_image(self,color_n = 30,global_max = None):
        h, b = self.matrix.shape
        img = Image.new(mode="RGB", size=(b, h), color=(0, 0, 0))
        max_iter = np.max(self.counter) if global_max is None else global_max

        for i in range(h):
            for j in range(b):
                ratio = self.counter[i][j] / (max_iter + 1)
                #print(ratio)
                col_temp = tuple(color_list[math.floor(ratio * color_n)])
                img.putpixel((j, i), col_temp)
        return img

    #generates the mandelbrot set in batches rather than making it all at once
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

    #GENERATES A ZOOM SEQUENCE OF THE MANDELBROT SET WHERE IT INITIALLY STARTS WITH AN IMAGE HAVING CORNERS corner1, corner2 AND ZOOMS INTO THE zoom_center
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

    #SIMILAR TO ZOOM SEQUENCE BUT EACH FRAME IS GENERATED USING STITCH IMAGE
    #CAN BE USED TO CREATE HIGH RESOLUTION ZOOM SEQUENCES
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

    #patch wise stitched divergence image of mandelbrot set . The patchiness issue present in julia.stitch_image_divergence is not present here
    #as global max iter is iter_per_patch(as long as some points of the mandelbrot set lie within the image)
    def stitch_image_divergence(self,patch_dim = 100 , patch_num_x = 5 , patch_num_y = 5,
                                iter_per_patch = 200, color_n = 50):
        super_img = Image.new(mode="RGB", size=(patch_num_x * patch_dim, patch_num_y * patch_dim), color=(0, 0, 0))
        boundary_vertices = complex_matrix(self.corners[0], self.corners[1], x_num=patch_num_x+1, y_num=patch_num_y+1)
        indx = 0
        total = patch_num_x * patch_num_y
        for i in range(patch_num_x):
            for j in range(patch_num_y):
                c1, c2 = boundary_vertices[j][i], boundary_vertices[j+1][i+1]
                temp = mandelbrot(c1, c2, x_num=patch_dim, y_num=patch_dim,
                             power=self.power, explosion_boundary=self.explosion_boundary)
                temp.run(iter=iter_per_patch)
                temp_img = temp.divergence_image(color_n=color_n,global_max = iter_per_patch)
                super_img.paste(temp_img, (i*patch_dim, j*patch_dim))
                del temp, temp_img
                sys.stdout.write(f'\rProgress: {indx*100/total:.2f}%')
                sys.stdout.flush()
                indx += 1
        return super_img

    #SAME AS ZOOM SEQUENCE BUT USES divergence_image
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


# center = -0.7435669 + 0.1314023j
# diagonal = 0.01 + 0.01j   # initial zoom box size
#
# corner1 = center + diagonal
# corner2 = center - diagonal
#
# x = mandelbrot(corner1, corner2, x_num=900, y_num=900)
# f = []
# for i in range(3,100):
#     x.run(iter = 100)
#     f.append(x.divergence_image(color_n = i))
#     x.reset()
# save_images_as_gif(f,output_path='div.gif',duration=50)
