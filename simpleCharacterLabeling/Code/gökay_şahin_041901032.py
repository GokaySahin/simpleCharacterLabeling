# IMPORTED MODULES
#-----------------------------------------------------------------------------------
from PIL import Image, ImageDraw , ImageFont , ImageMath  # Python Imaging Library (PIL) modules
import PIL.ImageOps
import PySimpleGUI as sg
import numpy as np   # fundamental Python module for scientific computing
import math   # math module that provides mathematical functions
import os   # os module can be used for file and directory operations
from tkinter import *
import tkinter as tk
from tkinter import filedialog

# MAIN FUNCTION OF THE PROGRAM
#-----------------------------------------------------------------------------------
# Main function where this python script starts execution
# read color image from file
   #--------------------------------------------------------------------------------
   # path of the current directory where this program file is placed
   curr_dir = os.path.dirname(os.path.realpath(__file__))
   img_file = curr_dir +  '/thinABC123.jpg'
   # img_file = '/Users/gokmen/Pictures/t-small.png'
   img_color = Image.open(img_file)
   img_color.show() # display the color image
   # convert the color image to a grayscale image
   #--------------------------------------------------------------------------------
   img_gray = img_color.convert('L')
   img_gray.show() # display the grayscale image
   # create a binary image by thresholding the grayscale image
   #--------------------------------------------------------------------------------
   # convert the grayscale PIL Image to a numpy array
   arr_gray = np.asarray(img_gray)
   # values below the threshold are considered as ONE and values above the threshold
   # are considered as ZERO (assuming that the image has dark objects (e.g., letters
   # ABC or digits 123) on a light background)
   THRESH, ZERO, ONE = 165, 0, 255
   # the threshold function defined below returns the binary image as a numpy array
   arr_bin = threshold(arr_gray, THRESH, ONE, ZERO)
   # you can uncomment the line below to work on a 100x100 artificial binary image
   # that contains 3 lines, a square and a circle instead of an input image file
   # arr_bin = artificial_binary_image(ONE)
   # convert the numpy array of the binary image to a PIL Image
   img_bin = Image.fromarray(arr_bin)
   img_bin.show()# display the binary image
   # component (object) labeling based on 4-connected components
   #--------------------------------------------------------------------------------
   # blob_coloring_4_connected function returns a numpy array that contains labels
   # for the pixels in the input image, the number of different labels and the numpy
   # array of the image with colored blobs
   arr_labeled_img, num_labels, arr_blobs = blob_coloring_8_connected(arr_bin, ONE)
   # print the number of objects as the number of different labels
   print("There are " + str(num_labels) + " objects in the input image.")
   # write the values in the labeled image to a file
   labeled_img_file = curr_dir + '/labeled_img.txt'
   np.savetxt(labeled_img_file, arr_labeled_img, fmt='%d', delimiter=',')
   # convert the numpy array of the colored components (blobs) to a PIL Image
   img_blobs = Image.fromarray(arr_blobs)
   label_Indexes, label_Coordinates = min_max_finder(arr_labeled_img)
   #label_Coordinates=bi_inverter(label_Indexes,label_Coordinates, img_bin)
   line_printer(label_Indexes,label_Coordinates,img_blobs)
   img_blobs.show()# display the colored components (blobs)

# GENERATING AN ARTIFICIAL BINARY IMAGE
#-----------------------------------------------------------------------------------
# Function that creates and returns a 100x100 artificial binary image with 3 lines,
# a square and a circle (background pixels = 0 and shape pixels = HIGH)
# The returned image can be used for comparing 4-connected and 8-connected labeling
def artificial_binary_image(HIGH):
   # the generated image has the size 100 x 100
   n_rows = n_cols = 100
   # y and x are 2D arrays that store row and column indices for each pixel of the
   # artificial binary image
   y, x = np.indices((n_rows, n_cols))
   # code part that is used to generate the 3 lines on the artificial binary image
   #--------------------------------------------------------------------------------
   mask_lines = np.zeros(shape = (n_rows, n_cols))
   for i in range (50, 70):
      # code part that generates the mask for the thick \ shaped line on the right
      mask_lines[i][i] = 1
      mask_lines[i][i + 1] = 1
      mask_lines[i][i + 2] = 1
      mask_lines[i][i + 3] = 1
      # code part that generates the mask for the thin \ shaped line on the right
      # (this line can not be labeled correctly by using 4-connected labeling thus
      # it requires using 8-connected labeling)
      mask_lines[i][i + 6] = 1
      # code part that generates the mask for the thick / shaped line on the left
      mask_lines[i - 20][90 - i + 1] = 1
      mask_lines[i - 20][90 - i + 2] = 1
      mask_lines[i - 20][90 - i + 3] = 1
   # code part that is used to generate the masks for creating a square and a circle
   # on the artificial binary image
   #--------------------------------------------------------------------------------
   x0, y0, r0 = 30, 30, 5
   x1, y1, r1 = 70, 30, 5
   mask_square = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
   # the created circle can not be labeled correctly by using 4-connected labeling
   # thus it requires using 8-connected labeling
   mask_circle = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
   # an artificial binary image is created by applying the masks
   #--------------------------------------------------------------------------------
   mask_square_and_circle = np.logical_or(mask_square, mask_circle)
   a_bin_image = np.logical_or(mask_lines, mask_square_and_circle) * HIGH
   # the resulting artificial binary image is returned
   return a_bin_image

# BINARIZATION
#-----------------------------------------------------------------------------------
# Function for creating and returning a binary image as a numpy array by thresholding
# the given array of the grayscale image
def threshold(arr_gray_in, T, LOW, HIGH):
   # get the numbers of rows and columns in the array of the grayscale image
   n_rows, n_cols = arr_gray_in.shape
   # initialize the output (binary) array by using the same size as the input array
   # and filling with zeros
   arr_bin_out = np.zeros(shape = arr_gray_in.shape)
   # for each value in the given array of the grayscale image
   for i in range(n_rows):
      for j in range(n_cols):
         # if the value is smaller than the given threshold T
         if abs(arr_gray_in[i][j]) < T:
            # the corresponding value in the output (binary) array becomes LOW
            arr_bin_out[i][j] = LOW
         # if the value is greter than or equal to the given threshold T
         else:
            # the corresponding value in the output (binary) array becomes HIGH
            arr_bin_out[i][j] = HIGH
   # return the resulting output (binary) array
   return arr_bin_out

# CONNECTED COMPONENT LABELING AND BLOB COLORING
#-----------------------------------------------------------------------------------
# Function for labeling objects as 4-connected components in a binary image whose
# numpy array is given as an input argument and creating an image with randomly
# colored components (blobs)
def blob_coloring_8_connected(arr_bin, ONE):
   # get the numbers of rows and columns in the array of the binary image
   n_rows, n_cols = arr_bin.shape
   # max possible label value is set as 10000
   max_label = 10000
   # initially all the pixels in the image are labeled as max_label
   arr_labeled_img = np.zeros(shape = (n_rows, n_cols), dtype = int)
   for i in range(n_rows):
      for j in range(n_cols):
         arr_labeled_img[i][j] = max_label
   # keep track of equivalent labels in an array
   # initially this array contains values from 0 to max_label - 1
   equivalent_labels = np.arange(max_label, dtype = int)
   # labeling starts with k = 1
   k = 1
   # first pass to assign initial labels and update equivalent labels from conflicts
   # for each pixel in the binary image
   #--------------------------------------------------------------------------------
   for i in range(1, n_rows - 1):
      for j in range(1, n_cols - 1):
         c = arr_bin[i][j] # value of the current (center) pixel
         l = arr_bin[i][j - 1] # value of the left pixel
         label_l  = arr_labeled_img[i][j - 1] # label of the left pixel
         d = arr_bin[i-1][j-1]
         label_d = arr_labeled_img[i-1][j-1]
         r = arr_bin[i-1][j+1]
         label_r = arr_labeled_img[i-1][j+1]
         u = arr_bin[i - 1][j] # value of the upper pixel
         label_u  = arr_labeled_img[i - 1][j] # label of the upper pixel
         # only the non-background pixels are labeled
         if c == ONE:
            # get the minimum of the labels of the upper and left pixels
            min_label = min(label_u, label_l)
            # if both upper and left pixels are background pixels
            if min_label == max_label:
               # label the current (center) pixel with k and increase k by 1
               arr_labeled_img[i][j] = k
               k += 1
            # if at least one of upper and left pixels is not a background pixel
            else:
               # label the current (center) pixel with min_label
               arr_labeled_img[i][j] = min_label
               # if upper pixel has a bigger label and it is not a background pixel
               if min_label != label_u and label_u != max_label:
                  # update the array of equivalent labels for label_u
                  update_array(equivalent_labels, min_label, label_u)
               # if left pixel has a bigger label and it is not a background pixel
               if min_label != label_l and label_l != max_label:
                  # update the array of equivalent labels for label_l
                  update_array(equivalent_labels, min_label, label_l)
               if min_label != label_d and label_d != max_label:
                  update_array(equivalent_labels,  min_label,  label_d)
               if min_label != label_r and label_r != max_label:
                  update_array(equivalent_labels,  min_label,  label_r)
   # final reduction in the array of equivalent labels to obtain the min. equivalent
   # label for each used label (values from 1 to k - 1) in the first pass of labeling
   #--------------------------------------------------------------------------------
   for i in range(1, k):
      index = i
      while equivalent_labels[index] != index:
         index = equivalent_labels[index]
      equivalent_labels[i] = equivalent_labels[index]
   # create a color map for randomly coloring connected components (blobs)
   #--------------------------------------------------------------------------------
   color_map = np.zeros(shape = (k, 3), dtype = np.uint8)
   np.random.seed(0)
   for i in range(k):
      color_map[i][0] = np.random.randint(0, 255, 1, dtype = np.uint8)
      color_map[i][1] = np.random.randint(0, 255, 1, dtype = np.uint8)
      color_map[i][2] = np.random.randint(0, 255, 1, dtype = np.uint8)
   # create an array for the image to store randomly colored blobs
   arr_color_img = np.zeros(shape = (n_rows, n_cols, 3), dtype = np.uint8)
   # second pass to resolve labels by assigning the minimum equivalent label for each
   # label in arr_labeled_img and color connected components (blobs) randomly
   #--------------------------------------------------------------------------------
   for i in range(n_rows):
      for j in range(n_cols):
         # only the non-background pixels are taken into account
         if arr_bin[i][j] == ONE:
            arr_labeled_img[i][j] = equivalent_labels[arr_labeled_img[i][j]]
            arr_color_img[i][j][0] = color_map[arr_labeled_img[i][j], 0]
            arr_color_img[i][j][1] = color_map[arr_labeled_img[i][j], 1]
            arr_color_img[i][j][2] = color_map[arr_labeled_img[i][j], 2]
         # change the label values of background pixels from max_label to 0
         else:
            arr_labeled_img[i][j] = 0
   # obtain the set of different values of the labels used to label the image
   different_labels = set(equivalent_labels[1:k])
   # compute the number of different values of the labels used to label the image
   num_different_labels = len(different_labels)
   # return the labeled image as a numpy array, number of different labels and the
   # image with colored blobs (components) as a numpy array
   return arr_labeled_img, num_different_labels, arr_color_img

# Function for updating the equivalent labels array by merging label1 and label2
# that are determined to be equivalent
def update_array(equ_labels, label1, label2) :
   # determine the small and large labels between label1 and label2
   if label1 < label2:
      lab_small = label1
      lab_large = label2
   else:
      lab_small = label2
      lab_large = label1
   # starting index is the large label
   index = lab_large
   # using an infinite while loop
   while True:
      # update the label of the currently indexed array element with lab_small when
      # it is bigger than lab_small
      if equ_labels[index] > lab_small:
         lab_large = equ_labels[index]
         equ_labels[index] = lab_small
         # continue the update operation from the newly encountered lab_large
         index = lab_large
      # update lab_small when a smaller label value is encountered
      elif equ_labels[index] < lab_small:
         lab_large = lab_small # lab_small becomes the new lab_large
         lab_small = equ_labels[index] # smaller value becomes the new lab_small
         # continue the update operation from the new value of lab_large
         index = lab_large
      # end the loop when the currently indexed array element is equal to lab_small
      else: # equ_labels[index] == lab_small
         break
#which locates min-max coordinates of labels and label numbers as member of different arrays
def min_max_finder(labeled_array):
   # +y -x -y +x
   rows = len(labeled_array)
   columns = len(labeled_array[0])
   w, h = 5, 3001;
   labels_coordinates = [[0 for x in range(w)] for y in range(h)] # +y -x -y +x
   labels_accrossed = []
   for row in range(rows):
      for column in range(columns):
         if(labeled_array[row][column]!=0):
            if(labeled_array[row][column] not in labels_accrossed):
               labels_accrossed.append(labeled_array[row][column])
               labels_coordinates[labeled_array[row][column]][0] = column #-x
               labels_coordinates[labeled_array[row][column]][1] = row #-y
               labels_coordinates[labeled_array[row][column]][2] = column #+x
               labels_coordinates[labeled_array[row][column]][3] = row #+y
               # -x -y +x +y
            else:
               if ( column < labels_coordinates[labeled_array[row][column]][0] ):
                  labels_coordinates[labeled_array[row][column]][0] = column
               if (column > labels_coordinates[labeled_array[row][column]][2]):
                  labels_coordinates[labeled_array[row][column]][2] = column
               if (row < labels_coordinates[labeled_array[row][column]][3]):
                  labels_coordinates[labeled_array[row][column]][3] = row
               if (row > labels_coordinates[labeled_array[row][column]][1]):
                  labels_coordinates[labeled_array[row][column]][1] = row
   return(labels_accrossed,labels_coordinates)
#binary inverter which takes input labels types, coordinates and image to crop
def bi_inverter(labels_accrossed,labels_coordinates,image):
   for i in range(len(labels_accrossed)):
      max_pixel=255
      label=labels_accrossed[i]
      min_x = labels_coordinates[label][0]-2
      min_y = labels_coordinates[label][1]+2
      max_x = labels_coordinates[label][2]+2
      max_y = labels_coordinates[label][3]-2
      width=max_x-min_x
      height=min_y-max_y
      im12= image.crop((min_x,max_y,min_x+width,max_y+height))
      im12_inverted=PIL.ImageOps.invert(im12.convert("RGB"))
      im12_inverted=im12_inverted.convert('L')
      im12_arr = np.asarray(im12_inverted)
      labeled_im12, numberof_blobs, arr_im12 = blob_coloring_8_connected(im12_arr, ONE=255)
      labels_coordinates[label][4] = numberof_blobs

   return(labels_coordinates)
# which prints lines with correct coordinates
def line_printer(labels_accrossed,labels_coordinates,im):
   text_A = 'A'
   text_B = 'B'
   text_C = 'C'
   for label in labels_accrossed:
      min_x = labels_coordinates[label][0]
      min_y = labels_coordinates[label][1]
      max_x = labels_coordinates[label][2]
      max_y = labels_coordinates[label][3]
      shape = (min_x,min_y),(max_x,max_y)
      draw = ImageDraw.Draw(im)
      draw.rectangle(shape)
      font = ImageFont.truetype(r'C:\\Users\\gokay\\PycharmProjects\\helloworld\arial.ttf')
      #draw.text((min_x,max_y))

# main() function is specified as the entry point where the program starts running
if __name__=='__main__':
   main()