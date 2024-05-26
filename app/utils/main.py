# editing version--peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from scipy.signal import find_peaks

import tkinter.filedialog
import tkinter as tk

from ncempy.io import dm as DM

from scipy.signal import find_peaks

from PIL import Image

import sys

import datetime
import time

matplotlib.use('TkAgg')
np.set_printoptions(suppress=True)

window = tk.Tk()
window.geometry('920x550')
window.title('CNT direction measurement')

canvas = tk.Canvas(window, width=1850, height=700)
canvas.pack(fill='both', expand=True)
rect_select_img = canvas.create_rectangle(280, 10, 600, 130)
rect_step = canvas.create_rectangle(280, 140, 600, 280)
rect_image_size = canvas.create_rectangle(20, 10, 260, 110)
rect_processing_parameters = canvas.create_rectangle(20, 120, 260, 360)
rect_batch_recognize = canvas.create_rectangle(20, 370, 260, 530)
rect_partial_calculate = canvas.create_rectangle(620, 10, 900, 410)

# cut image size
lab_image_size = tk.Label(window, text='Image Size:', width=13, font=('Arial', 12))
lab_image_size.place(x=30, y=30, anchor='w')


def set_percentage():
    if not ('image_backup' in globals()):
        tkinter.messagebox.showerror(title='Error!', message='Choose image first！')
        return 0

    image_size_percentage.configure(state='disabled')
    image_size_pixel.configure(state='normal')

    x_min_pixel = int(x_size_min.get())
    x_max_pixel = int(x_size_max.get())
    y_min_pixel = int(y_size_min.get())
    y_max_pixel = int(y_size_max.get())

    x_min_percentage = int(np.round(x_min_pixel / np.shape(image_backup)[1] * 100))
    x_max_percentage = int(np.round(x_max_pixel / np.shape(image_backup)[1] * 100))
    y_min_percentage = int(np.round(y_min_pixel / np.shape(image_backup)[0] * 100))
    y_max_percentage = int(np.round(y_max_pixel / np.shape(image_backup)[0] * 100))

    x_size_min.set(x_min_percentage)
    x_size_max.set(x_max_percentage)
    y_size_min.set(y_min_percentage)
    y_size_max.set(y_max_percentage)


def set_pixel():
    if not ('image_backup' in globals()):
        tkinter.messagebox.showerror(title='Error!', message='Choose image first！')
        var_image_size.set(0)
        return 0

    image_size_percentage.configure(state='normal')
    image_size_pixel.configure(state='disabled')

    x_min_percentage = int(x_size_min.get())
    x_max_percentage = int(x_size_max.get())
    y_min_percentage = int(y_size_min.get())
    y_max_percentage = int(y_size_max.get())

    x_min_pixel = int(np.round(x_min_percentage / 100 * np.shape(image_backup)[1]))
    x_max_pixel = int(np.round(x_max_percentage / 100 * np.shape(image_backup)[1]))
    y_min_pixel = int(np.round(y_min_percentage / 100 * np.shape(image_backup)[0]))
    y_max_pixel = int(np.round(y_max_percentage / 100 * np.shape(image_backup)[0]))

    x_size_min.set(x_min_pixel)
    x_size_max.set(x_max_pixel)
    y_size_min.set(y_min_pixel)
    y_size_max.set(y_max_pixel)


var_image_size = tk.IntVar(value=0)
image_size_percentage = tk.Radiobutton(window, text='%', variable=var_image_size, state='disabled', value=0,
                                       font=('Arial', 12), command=set_percentage)
image_size_pixel = tk.Radiobutton(window, text='px', variable=var_image_size, value=1, state='normal',
                                  font=('Arial', 12), command=set_pixel)
image_size_percentage.place(x=140, y=30, anchor='w')
image_size_pixel.place(x=180, y=30, anchor='w')

lab_x_size = tk.Label(window, text='x:', font=('Arial', 12))
lab_x_size.place(x=40, y=60, anchor='w')

x_size_min = tk.StringVar(value='0')
entry_x_size_min = tk.Entry(window, font=('Arial', 14), width=5, textvariable=x_size_min)
entry_x_size_min.place(x=70, y=60, anchor='w')

lab_x_to = tk.Label(window, text='to', font=('Arial', 12))
lab_x_to.place(x=140, y=60, anchor='w')

x_size_max = tk.StringVar(value='100')
entry_x_size_max = tk.Entry(window, font=('Arial', 14), width=5, textvariable=x_size_max)
entry_x_size_max.place(x=170, y=60, anchor='w')

lab_y_size = tk.Label(window, text='y:', font=('Arial', 12))
lab_y_size.place(x=40, y=90, anchor='w')

y_size_min = tk.StringVar(value='0')
entry_y_size_min = tk.Entry(window, font=('Arial', 14), width=5, textvariable=y_size_min)
entry_y_size_min.place(x=70, y=90, anchor='w')

lab_y_to = tk.Label(window, text='to', font=('Arial', 12))
lab_y_to.place(x=140, y=90, anchor='w')

y_size_max = tk.StringVar(value='80')
entry_y_size_max = tk.Entry(window, font=('Arial', 14), width=5, textvariable=y_size_max)
entry_y_size_max.place(x=170, y=90, anchor='w')

# make sure all plat are new created
plot_number = 0

# input image
lab_select_image = tk.Label(window, text='Select Image:', width=30, font=('Arial', 12))
lab_select_image.place(x=300, y=30, anchor='w')


def select_image_DM():
    global image_type
    global filename
    global image_load
    global image_raw
    global image_backup  # used in resize image

    global real_size_x_min
    global real_size_x_max
    global real_size_y_min
    global real_size_y_max

    filename = tk.filedialog.askopenfilename()

    if filename != "":
        image_raw = DM.dmReader(filename)
        image_load = image_raw['data']
        image_backup = image_load.copy()
        plt.figure(figsize=(10, 8))
        plt.title("Input Image")
        plt.imshow(np.abs(image_load), cmap='gray')
        plt.show()

        real_size_x_min = 0
        real_size_x_max = 100
        real_size_y_min = 0
        real_size_y_max = 100


def select_image_img():
    global image_type
    global filename
    global image_load
    global image_raw
    global image_backup

    global real_size_x_min
    global real_size_x_max
    global real_size_y_min
    global real_size_y_max

    filename = tk.filedialog.askopenfilename()

    if filename != "":
        image_load = plt.imread(filename)
        if np.ndim(image_load) == 3:
            image_load = image_load[:, :, 0]
        image_backup = image_load.copy()

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title("Input Image")
        plt.imshow(np.abs(image_load), cmap='gray')

        # draw arrow test?
        # def onclick(event):
        #    global aaa
        #    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #    #      ('double' if event.dblclick else 'single', event.button,
        #    #       event.x, event.y, event.xdata, event.ydata))
        #    print('on, x={:.2f},y={:.2f}'.format(event.xdata,event.ydata))
        #    aaa=(event.xdata,event.ydata)
        #    print(aaa)
        #
        # def offclick(event):
        #    global bbb
        #    print('off, x={:.2f},y={:.2f}'.format(event.xdata,event.ydata))
        #    bbb=(event.xdata,event.ydata)
        #    print(bbb)
        #    plt.annotate(text='', xy=aaa, xytext=bbb, arrowprops=dict(arrowstyle='<->',color='red',lw=1.5))
        #    plt.show()
        #
        # cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # cid2 = fig.canvas.mpl_connect('button_release_event',offclick)

        plt.show()

        real_size_x_min = 0
        real_size_x_max = 100
        real_size_y_min = 0
        real_size_y_max = 100


buttun_select_image_DM = tk.Button(window, text='DM file', font=('Arial', 12), height=1, width=13,
                                   command=select_image_DM)
buttun_select_image_DM.place(x=300, y=60, anchor='w')

buttun_select_image_img = tk.Button(window, text='IMG file', font=('Arial', 12), height=1, width=13,
                                    command=select_image_img)
buttun_select_image_img.place(x=453, y=60, anchor='w')


def resize_img():
    global image_load
    global real_size_x_min
    global real_size_x_max
    global real_size_y_min
    global real_size_y_max

    raw_x = np.shape(image_backup)[0]
    raw_y = np.shape(image_backup)[1]

    image_size_type = var_image_size.get()

    if image_size_type == 0:
        image_load = image_backup[
                     int(0.01 * raw_x * float(y_size_min.get())):int(0.01 * raw_x * float(y_size_max.get())),
                     int(0.01 * raw_y * float(x_size_min.get())):int(0.01 * raw_y * float(x_size_max.get()))]
    elif image_size_type == 1:
        image_load = image_backup[int(y_size_min.get()):int(y_size_max.get()),
                     int(x_size_min.get()):int(x_size_max.get())]

    print("Resized image size: {} * {}".format(np.shape(image_load)[0], np.shape(image_load)[1]))

    real_size_x_min = x_size_min.get()
    real_size_x_max = x_size_max.get()
    real_size_y_min = y_size_min.get()
    real_size_y_max = y_size_max.get()

    plt.figure(figsize=(10, 8))
    plt.title("Resized Input Image")
    plt.imshow(np.abs(image_load), cmap='gray')
    plt.show()


buttun_resize_image = tk.Button(window, text='Resize Image', font=('Arial', 12), height=1, width=30, command=resize_img)
buttun_resize_image.place(x=300, y=100, anchor='w')

# show FFT
lab_processing_parameters = tk.Label(window, text='Processing Parameters :', width=23, font=('Arial', 12))
lab_processing_parameters.place(x=30, y=140, anchor='w')

lab_mask_inside_radius = tk.Label(window, text='Mask inside r (%) :', font=('Arial', 12))
lab_mask_inside_radius.place(x=30, y=170, anchor='w')

mask_inside_radius = tk.StringVar(value='10')
entry_mask_inside_radius = tk.Entry(window, font=('Arial', 14), width=5, textvariable=mask_inside_radius)
entry_mask_inside_radius.place(x=190, y=170, anchor='w')

lab_mask_outside_radius = tk.Label(window, text='Mask outside r (%) :', font=('Arial', 12))
lab_mask_outside_radius.place(x=30, y=210, anchor='w')

mask_outside_radius = tk.StringVar(value='60')
entry_mask_outside_radius = tk.Entry(window, font=('Arial', 14), width=5, textvariable=mask_outside_radius)
entry_mask_outside_radius.place(x=190, y=210, anchor='w')


def show_mask_FFT():
    global center_masked_FFT
    global masked_FFT
    global fftMagImage

    global whole_mask

    fftImage = np.fft.fft2(image_load)
    fftShiftImage = np.fft.fftshift(fftImage)
    fftMagImage = np.abs(fftShiftImage)

    # center_masked_FFT=fftMagImage.copy()
    # masked_FFT=fftMagImage.copy()

    inside_r_precentage = float(mask_inside_radius.get())
    outside_r_precentage = float(mask_outside_radius.get())

    inside_r = np.amin(np.shape(fftMagImage)) * inside_r_precentage / 100 / 2
    outside_r = np.amin(np.shape(fftMagImage)) * outside_r_precentage / 100 / 2

    # mask of center masked FFT
    center_mask = np.ones(np.shape(fftImage))

    # specify circle parameters: centre ij
    ci, cj = np.shape(center_mask)[0] / 2, np.shape(center_mask)[1] / 2

    # Create index arrays to z
    I, J = np.meshgrid(np.arange(center_mask.shape[1]), np.arange(center_mask.shape[0]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

    # Assign value of 0 to those points where dist<cr:
    center_mask[np.where(dist < inside_r)] = 0

    # mask of masked FFT
    whole_mask = center_mask.copy()
    whole_mask[np.where(dist > outside_r)] = 0

    # masked FFT and center masked FFT
    center_masked_FFT = np.multiply(center_mask, fftMagImage)
    masked_FFT = np.multiply(whole_mask, fftMagImage)

    # show FFT
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title("Center Masked FFT")
    plt.axis('off')
    plt.imshow(np.abs(center_masked_FFT), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Masked FFT")
    plt.axis('off')
    plt.imshow(np.abs(masked_FFT), cmap='gray')
    plt.show()


buttun_mask_FFT = tk.Button(window, text='Show Masked FFT', font=('Arial', 12), height=1, width=30,
                            command=show_mask_FFT)
buttun_mask_FFT.place(x=300, y=170, anchor='w')

# angle step size
lab_angle_step = tk.Label(window, text='Angle step:', font=('Arial', 12))
lab_angle_step.place(x=30, y=250, anchor='w')

angle_step = tk.StringVar(value='1')
entry_angle_step = tk.Entry(window, font=('Arial', 14), width=5, textvariable=angle_step)
entry_angle_step.place(x=190, y=250, anchor='w')


# calculate angle distribution
def angle_distribution():
    global ave_intensity
    global x1
    global sum_number
    part_sum = int(180 / float(angle_step.get()))

    center = [float((np.shape(fftMagImage)[0] - 1) / 2), float((np.shape(fftMagImage)[1] - 1) / 2)]

    sum_intensity = np.zeros(part_sum * 2)
    sum_number = np.zeros(part_sum * 2)
    ave_intensity = np.zeros(part_sum * 2)

    for i in range(np.shape(fftMagImage)[0]):
        for j in range(np.shape(fftMagImage)[1]):
            y_0 = i - center[0]
            x_0 = j - center[1]
            if whole_mask[i][j] == 1:
                if j - center[1] == 0:
                    part_number = int(part_sum / 2)
                    sum_intensity[part_number] = sum_intensity[part_number] + fftMagImage[i][j]
                    sum_number[part_number] = sum_number[part_number] + 1
                else:
                    angle = np.arctan(y_0 / x_0) / np.pi * 180

                    part_number = round(angle / float(angle_step.get()))
                    if part_number < 0:
                        part_number = part_number + part_sum
                    if part_number >= part_sum:
                        part_number = part_number - part_sum
                    sum_intensity[part_number] = sum_intensity[part_number] + fftMagImage[i][j]
                    sum_number[part_number] = sum_number[part_number] + 1

    # output pixel number-angle relationship (test part)
    # plt.figure(figsize=(10,8))
    # x1=np.array(range(part_sum))/part_sum*180
    # plt.title("pixel number")
    # plt.plot(x1,sum_number[0:int(np.shape(sum_number)[0]/2)])
    # plt.show()

    for i in range(part_sum):
        ave_intensity[i] = sum_intensity[i] / sum_number[i]

    for i in range(part_sum):
        ave_intensity[i + part_sum] = ave_intensity[i]

    # ave_intensity=np.flipud(ave_intensity)     #???

    # ave_intensity=ave_intensity-np.amin(ave_intensity)
    # ave_intensity=ave_intensity/np.amax(ave_intensity)     #normalize

    # show distribution
    plt.figure(figsize=(10, 8))
    x1 = np.array(range(part_sum * 2)) / part_sum * 180
    plt.title("Angle Distribution")
    plt.plot(x1, ave_intensity)
    plt.show()


buttun_distribution = tk.Button(window, text='Calculate Angle Distribution', font=('Arial', 12), height=1, width=30,
                                command=angle_distribution)
buttun_distribution.place(x=300, y=210, anchor='w')

# min_prominence
lab_min_prominence = tk.Label(window, text='Min prominence:', font=('Arial', 12))
lab_min_prominence.place(x=30, y=290, anchor='w')

min_prominence = tk.StringVar(value='0.2')
entry_min_prominence = tk.Entry(window, font=('Arial', 14), width=5, textvariable=min_prominence)
entry_min_prominence.place(x=190, y=290, anchor='w')

# min_width
lab_min_width = tk.Label(window, text='Min width:', font=('Arial', 12))
lab_min_width.place(x=30, y=330, anchor='w')

min_width = tk.StringVar(value='10')
entry_min_width = tk.Entry(window, font=('Arial', 14), width=5, textvariable=min_width)
entry_min_width.place(x=190, y=330, anchor='w')


# calculate Chebyshev orientation parameter(COP)

def cal_COP(array, peak_max, step):
    """
    step: angle step
    """
    numerator_COP = 0
    for i in range(np.shape(array)[0]):
        numerator_COP = numerator_COP + array[i] * ((np.cos((i * step - peak_max) / 180 * np.pi)) ** 2) * step

    denominator_COP = np.sum(array * step)
    COP = 2 * numerator_COP / denominator_COP - 1
    return COP


def cal_ODF_parameter(array, peak_max, step, type_para=1):
    """
    type 1: 1/(x^0.1+1)
    type 2:cos^2
    type n:1/(x^n+1),n<1
    """
    numerator_para = 0
    nn = 0.1
    # fun_para=np.zeros(np.shape(array)[0])
    # for i in range(np.shape(array)[0]):
    #    #piecewise function of weights
    #    if i < (np.shape(array)[0]/4):
    #        fun_para[i]=(1/(np.power(i*step+1,nn))-1/(np.power(91,nn)))*(np.power(91,nn))/(np.power(90,nn)-1)
    #    elif i < (np.shape(array)[0]/4)*2:
    #        fun_para[i]=(1/(np.power(180-i*step+1,nn))-1/(np.power(91,nn)))*(np.power(91,nn))/(np.power(90,nn)-1)
    #    elif i < (np.shape(array)[0]/4)*3:
    #        fun_para[i]=(1/(np.power(i*step-180+1,nn))-1/(np.power(91,nn)))*(np.power(91,nn))/(np.power(90,nn)-1)
    #    else:
    #        fun_para[i]=(1/(np.power(360-i*step+1,nn))-1/(np.power(91,nn)))*(np.power(91,nn))/(np.power(90,nn)-1)

    ##without n
    # if i < (np.shape(a)[0]/4):
    #    fun_para=(1/(i*step+1)-1/(90+1))*step*(90+1)/90
    # elif i < (np.shape(a)[0]/4)*2:
    #    fun_para=(1/(180-i*step+1)-1/(90+1))*step*(90+1)/90
    # elif i < (np.shape(a)[0]/4)*3:
    #    fun_para=(1/(i*step-180+1)-1/(90+1))*step*(90+1)/90
    # else:
    #    fun_para=(1/(360-i*step+1)-1/(90+1))*step*(90+1)/90

    fun_para = np.zeros(360)

    if type_para == 1:
        # 1/x type
        for i in range(360):
            # piecewise function of weights
            if i < (360 / 4):
                fun_para[i] = (1 / (np.power(i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)
            elif i < (360 / 4) * 2:
                fun_para[i] = (1 / (np.power(180 - i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)
            elif i < (360 / 4) * 3:
                fun_para[i] = (1 / (np.power(i - 180 + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                        np.power(91, nn) - 1)
            else:
                fun_para[i] = (1 / (np.power(360 - i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)

    elif type_para < 1:
        # 1/x type, changable n
        nn = type_para
        for i in range(360):
            # piecewise function of weights
            if i < (360 / 4):
                fun_para[i] = (1 / (np.power(i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)
            elif i < (360 / 4) * 2:
                fun_para[i] = (1 / (np.power(180 - i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)
            elif i < (360 / 4) * 3:
                fun_para[i] = (1 / (np.power(i - 180 + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)
            else:
                fun_para[i] = (1 / (np.power(360 - i + 1, nn)) - 1 / (np.power(91, nn))) * (np.power(91, nn)) / (
                            np.power(91, nn) - 1)

    elif type_para == 2:
        # cos^2 type
        for i in range(360):
            fun_para[i] = np.power(np.cos(i / 180 * np.pi), 2)

    # elif type_para==10:
    #    for i in range(360):
    #        if i<=10:
    #            fun_para[i]=1-i/10
    #        elif i<170:
    #            fun_para[i]=0
    #        elif i<180:
    #            fun_para[i]=i/10-17
    #        elif i<190:
    #            fun_para[i]=-i/10+19
    #        elif i<350:
    #            fun_para[i]=0
    #        elif i<360:
    #            fun_para[i]=i/10-35

    elif type_para > 2:
        # linear type
        for i in range(360):
            if i <= type_para:
                fun_para[i] = 1 - i / type_para
            elif i < 180 - type_para:
                fun_para[i] = 0
            elif i < 180:
                fun_para[i] = i / type_para + 1 - 180 / type_para
            elif i < 180 + type_para:
                fun_para[i] = -i / type_para + 1 + 180 / type_para
            elif i < 360 - type_para:
                fun_para[i] = 0
            elif i < 360:
                fun_para[i] = i / type_para + 1 - 360 / type_para

    fun_i = np.zeros(np.shape(array)[0])
    for i in range(np.shape(array)[0]):
        fun_i[i] = int(i * step - peak_max)

        while fun_i[i] < 0:
            fun_i[i] = fun_i[i] + 360
        while fun_i[i] >= 360:
            fun_i[i] = fun_i[i] - 360

        numerator_para = numerator_para + array[i] * fun_para[int(fun_i[i])] * step

    denominator_para = np.sum(array * step)
    ODF_parameter = numerator_para / denominator_para

    # plt.figure()
    # plt.plot(np.array(range(np.shape(fun_para)[0])),fun_para)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.array(range(np.shape(fun_i)[0])),fun_i)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.array(range(np.shape(array)[0])),array)
    # plt.show()

    # print("function:{}".format(fun_para))

    return ODF_parameter


# fit curve
def fit_curve():
    prominence = float(min_prominence.get())
    width = float(min_width.get()) / float(angle_step.get())
    part_sum = int(180 / float(angle_step.get()))

    prominence_real = (np.amax(ave_intensity) - np.amin(ave_intensity)) * prominence

    global peaks
    peaks = find_peaks(ave_intensity, prominence=prominence_real, width=width)
    print(peaks)

    global peak_center
    global peak_error
    global peak_max
    global peak_choose
    global COP
    global para_ODF

    if np.shape(peaks[0])[0] == 1:
        print("Peak Number:{}".format(np.shape(peaks[0])[0]))

        peak_max = float(peaks[0]) * float(angle_step.get())
        peak_center = (peaks[1]['left_ips'][0] / part_sum * 180 + peaks[1]['right_ips'][0] / part_sum * 180) / 2
        peak_error = float(peaks[1]['widths'] / 2) * float(angle_step.get())

        COP = cal_COP(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()))
        para_ODF = cal_ODF_parameter(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()),
                                     type_para=1)
        normalized_para_ODF = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_center,
                                                step=float(angle_step.get()), type_para=1)
        # show distribution with analyzed peak
        plt.figure(figsize=(10, 8))
        plt.title("Angle Distribution with Analyzed Peak")
        plt.plot(x1, ave_intensity)

        # plt.scatter(x1[int(peaks[0])],ave_intensity[int(peaks[0])],c="g",marker="x")
        # plt.scatter(x1[int(peaks[1]['left_ips'])],ave_intensity[int(peaks[1]['left_ips'])],c="g",marker="v")
        # plt.scatter(x1[int(peaks[1]['right_ips'])],ave_intensity[int(peaks[1]['right_ips'])],c="g",marker="v")

        plt.scatter(x1[int(peaks[0])], ave_intensity[int(peaks[0])], c="g", marker="x")
        plt.scatter(peaks[1]['left_ips'] / part_sum * 180, peaks[1]['width_heights'], c="g", marker="v")
        plt.scatter(peaks[1]['right_ips'] / part_sum * 180, peaks[1]['width_heights'], c="g", marker="v")

        if peak_center > 180:
            plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                     "Peak Center = {:.2f} ({:.2f})° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}\nParameter of ODF = {:.3f}\nNormalized OP1={:.3f}".format(
                         peak_center, peak_center - 180, peak_error, peak_max, COP, para_ODF, normalized_para_ODF),
                     fontsize=10)
        else:
            plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                     "Peak Center = {:.2f}° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}\nParameter of ODF = {:.3f}\nNormalized OP1={:.3f}".format(
                         peak_center, peak_error, peak_max, COP, para_ODF, normalized_para_ODF), fontsize=10)

        plt.show()

    elif np.shape(peaks[0])[0] == 2:  # if fits two peaks, choose the higher one
        print("Peak Number:{}".format(np.shape(peaks[0])[0]))

        if float(peaks[1]['prominences'][0]) > float(peaks[1]['prominences'][1]):
            peak_choose = 0
        else:
            peak_choose = 1
        peak_max = float(peaks[0][peak_choose]) * float(angle_step.get())
        peak_center = (peaks[1]['left_ips'][peak_choose] / part_sum * 180 + peaks[1]['right_ips'][
            peak_choose] / part_sum * 180) / 2
        peak_error = float(peaks[1]['widths'][peak_choose] / 2) * float(angle_step.get())

        COP = cal_COP(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()))
        para_ODF = cal_ODF_parameter(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()),
                                     type_para=1)
        normalized_para_ODF = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_center,
                                                step=float(angle_step.get()), type_para=1)

        # show distribution with analyzed peak
        plt.figure(figsize=(10, 8))
        plt.title("Angle Distribution with Analyzed Peak")
        plt.plot(x1, ave_intensity)

        # plt.scatter(x1[int(peaks[0][peak_choose])],ave_intensity[int(peaks[0][peak_choose])],c="g",marker="x")
        # plt.scatter(x1[int(peaks[1]['left_ips'][peak_choose])],ave_intensity[int(peaks[1]['left_ips'][peak_choose])],c="g",marker="v")
        # plt.scatter(x1[int(peaks[1]['right_ips'][peak_choose])],ave_intensity[int(peaks[1]['right_ips'][peak_choose])],c="g",marker="v")

        plt.scatter(x1[int(peaks[0][peak_choose])], ave_intensity[int(peaks[0][peak_choose])], c="g", marker="x")
        plt.scatter(peaks[1]['left_ips'][peak_choose] / part_sum * 180, peaks[1]['width_heights'][peak_choose], c="g",
                    marker="v")
        plt.scatter(peaks[1]['right_ips'][peak_choose] / part_sum * 180, peaks[1]['width_heights'][peak_choose], c="g",
                    marker="v")

        if peak_center > 180:
            plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                     "Peak Center = {:.2f} ({:.2f})° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}\nParameter of ODF = {:.3f}\nNormalized OP1 = {:.3f}".format(
                         peak_center, peak_center - 180, peak_error, peak_max, COP, para_ODF, normalized_para_ODF),
                     fontsize=10)
        else:
            plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                     "Peak Center = {:.2f}° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}\nParameter of ODF = {:.3f}\nNormalized OP1 = {:.3f}".format(
                         peak_center, peak_error, peak_max, COP, para_ODF, normalized_para_ODF), fontsize=10)

        plt.show()

    else:
        print("Peak Number:{}".format(np.shape(peaks[0])[0]))

        plt.figure(figsize=(10, 8))
        plt.title("Angle Distribution with Analyzed Peak")
        plt.plot(x1, ave_intensity)

        peak_max = 0
        for i in range(np.shape(ave_intensity)[0]):
            if ave_intensity[i] > ave_intensity[peak_max]:
                peak_max = i

        COP = cal_COP(array=ave_intensity, peak_max=peak_max,
                      step=float(angle_step.get()))  # 对于找不出峰的图像，直接采用全图最高点作为峰值计算COP
        para_ODF = cal_ODF_parameter(array=ave_intensity, peak_max=peak_max, step=float(angle_step.get()), type_para=1)
        normalized_para_ODF = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_max,
                                                step=float(angle_step.get()), type_para=1)

        for i in range(np.shape(peaks[0])[0]):
            plt.scatter(x1[int(peaks[0][i])], ave_intensity[int(peaks[0][i])], c="g", marker="x")
            plt.scatter(peaks[1]['left_ips'][i] / part_sum * 180, peaks[1]['width_heights'][i], c="g", marker="v")
            plt.scatter(peaks[1]['right_ips'][i] / part_sum * 180, peaks[1]['width_heights'][i], c="g", marker="v")
            plt.scatter(int(peaks[1]['left_bases'][i] / part_sum * 180), ave_intensity[int(peaks[1]['left_bases'][i])],
                        c="r", marker="v")
            plt.scatter(int(peaks[1]['right_bases'][i] / part_sum * 180),
                        ave_intensity[int(peaks[1]['right_bases'][i])], c="r", marker="v")

        plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                 "Peaks Number : {}\nCOP = {:.3f}\nParameter of ODF = {:.3f}\nNormalized OP1 = {:.3f}".format(
                     np.shape(peaks[0])[0], COP, para_ODF, normalized_para_ODF), fontsize=10)
        plt.show()


buttun_fit_curve = tk.Button(window, text='Fit Curve', font=('Arial', 12), height=1, width=30, command=fit_curve)
buttun_fit_curve.place(x=300, y=250, anchor='w')


# draw all
def draw_all():
    show_mask_FFT()
    angle_distribution()
    fit_curve()


buttun_draw_all = tk.Button(window, text='Draw All', font=('Arial', 12), height=1, width=30, command=draw_all)
buttun_draw_all.place(x=300, y=310, anchor='w')


# save image
def save_it():
    window_save = tk.Toplevel(window)
    window_save.geometry('350x340')
    window_save.title('Save It!')

    canvas = tk.Canvas(window_save, width=1850, height=700)
    canvas.pack(fill='both', expand=True)
    rect_save_it = canvas.create_rectangle(20, 60, 320, 240)

    def select_save_route():
        global save_route
        save_route = tk.filedialog.askdirectory()
        print("Save Route : {}".format(save_route))

    buttun_select_save_route = tk.Button(window_save, text='Select Save Route', font=('Arial', 12), height=1, width=30,
                                         command=select_save_route)
    buttun_select_save_route.place(x=30, y=30, anchor='w')

    def save_reshaped_image():
        if save_route == "":
            print("please choose save route!")
            return 0

        plt.imsave("{}/{}_reshaped_image.png".format(save_route, os.path.basename(filename)[0:-4]), np.abs(image_load),
                   cmap='gray')
        print("Reshaped image saved successfully!")

    buttun_save_reshaped_image = tk.Button(window_save, text='Save Reshaped Image', font=('Arial', 12), height=1,
                                           width=30, command=save_reshaped_image)
    buttun_save_reshaped_image.place(x=30, y=90, anchor='w')

    def save_masked_FFT():
        if save_route == "":
            print("please choose save route!")
            return 0
        # plt.figure(figsize=(10,10))
        # plt.axis('off')
        # plt.imshow(np.abs(masked_FFT), cmap='gray')
        # plt.savefig("{}/{}_masked_FFT.png".format(save_route,os.path.basename(filename)[0:-4]))
        # plt.clf()
        # plt.close()

        plt.imsave("{}/{}_center_masked_FFT.png".format(save_route, os.path.basename(filename)[0:-4]),
                   np.abs(center_masked_FFT), cmap='gray')
        plt.imsave("{}/{}_masked_FFT.png".format(save_route, os.path.basename(filename)[0:-4]), np.abs(masked_FFT),
                   cmap='gray')

        print("Masked FFT saved successfully!")

    buttun_save_masked_FFT = tk.Button(window_save, text='Save Masked FFT', font=('Arial', 12), height=1, width=30,
                                       command=save_masked_FFT)
    buttun_save_masked_FFT.place(x=30, y=130, anchor='w')

    def save_angle_distribution_plot():
        if save_route == "":
            print("please choose save route!")
            return 0
        part_sum = int(180 / float(angle_step.get()))

        plt.figure()
        plt.title("Angle Distribution with Analyzed Peak")
        plt.plot(x1, ave_intensity)

        if np.shape(peaks[0])[0] == 1:
            plt.scatter(x1[int(peaks[0])], ave_intensity[int(peaks[0])], c="g", marker="x")
            plt.scatter(peaks[1]['left_ips'] / part_sum * 180, peaks[1]['width_heights'], c="g", marker="v")
            plt.scatter(peaks[1]['right_ips'] / part_sum * 180, peaks[1]['width_heights'], c="g", marker="v")

            if peak_center >= 180:
                plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                         "Peak Center = {:.2f} ({:.2f})° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}".format(peak_center,
                                                                                                         peak_center - 180,
                                                                                                         peak_error,
                                                                                                         peak_max, COP),
                         fontsize=10)
            else:
                plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                         "Peak Center = {:.2f}° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}".format(peak_center, peak_error,
                                                                                                peak_max, COP),
                         fontsize=10)

        elif np.shape(peaks[0])[0] == 2:
            plt.scatter(x1[int(peaks[0][peak_choose])], ave_intensity[int(peaks[0][peak_choose])], c="g", marker="x")
            plt.scatter(peaks[1]['left_ips'][peak_choose] / part_sum * 180, peaks[1]['width_heights'][peak_choose],
                        c="g", marker="v")
            plt.scatter(peaks[1]['right_ips'][peak_choose] / part_sum * 180, peaks[1]['width_heights'][peak_choose],
                        c="g", marker="v")

            if peak_center >= 180:
                plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                         "Peak Center = {:.2f} ({:.2f})° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}".format(peak_center,
                                                                                                         peak_center - 180,
                                                                                                         peak_error,
                                                                                                         peak_max, COP),
                         fontsize=10)
            else:
                plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                         "Peak Center = {:.2f}° ± {:.2f}°\nPeak Max = {}°, COP = {:.3f}".format(peak_center, peak_error,
                                                                                                peak_max, COP),
                         fontsize=10)
        else:
            for i in range(np.shape(peaks[0])[0]):
                plt.scatter(x1[int(peaks[0][i])], ave_intensity[int(peaks[0][i])], c="g", marker="x")
                plt.scatter(peaks[1]['left_ips'][i] / part_sum * 180, peaks[1]['width_heights'][i], c="g", marker="v")
                plt.scatter(peaks[1]['right_ips'][i] / part_sum * 180, peaks[1]['width_heights'][i], c="g", marker="v")
                plt.scatter(int(peaks[1]['left_bases'][i] / part_sum * 180),
                            ave_intensity[int(peaks[1]['left_bases'][i])], c="r", marker="v")
                plt.scatter(int(peaks[1]['right_bases'][i] / part_sum * 180),
                            ave_intensity[int(peaks[1]['right_bases'][i])], c="r", marker="v")

            plt.text(0, 0.9 * (np.amax(ave_intensity) - np.amin(ave_intensity)) + np.amin(ave_intensity),
                     "Peaks Number : {}".format(np.shape(peaks[0])[0]), fontsize=10)

        plt.savefig("{}/{}_angle_distribution.png".format(save_route, os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        print("Angle distribution plot saved successfully!")

    buttun_save_angle_distribution_plot = tk.Button(window_save, text='Save Angle Distribution Plot',
                                                    font=('Arial', 12), height=1, width=30,
                                                    command=save_angle_distribution_plot)
    buttun_save_angle_distribution_plot.place(x=30, y=170, anchor='w')

    def save_angle_distribution_data():
        fo = file_angle_distribution = open(
            "{}/{}_angle_distribution_data.txt".format(save_route, os.path.basename(filename)[0:-4]), "w+")
        file_angle_distribution.write("Source image name : {}\n\n".format(filename))

        file_angle_distribution.write("Processing information:\n")
        file_angle_distribution.write(
            "Image size(%): x:{}-{} ; y:{}-{}\n".format(real_size_x_min, real_size_x_max, real_size_y_min,
                                                        real_size_y_max))
        file_angle_distribution.write("Mask inside radius (%): {}\n".format(mask_inside_radius.get()))
        file_angle_distribution.write("Mask outside radius (%): {}\n".format(mask_outside_radius.get()))
        file_angle_distribution.write("Angle step: {}\n".format(angle_step.get()))
        file_angle_distribution.write("Min prominence: {}\n".format(min_prominence.get()))
        file_angle_distribution.write("Min width: {}\n\n".format(min_width.get()))

        file_angle_distribution.write("Processing result:\n")
        if (np.shape(peaks[0])[0] == 2) or (np.shape(peaks[0])[0] == 1):
            if peak_center >= 180:
                file_angle_distribution.write(
                    "Peak Center = {:.2f} ({:.2f})° ± {:.2f}°\nPeak Max = {}°\nCOP = {:.7f}\n\n".format(peak_center,
                                                                                                        peak_center - 180,
                                                                                                        peak_error,
                                                                                                        peak_max, COP))
            else:
                file_angle_distribution.write(
                    "Peak Center = {:.2f}° ± {:.2f}°\nPeak Max = {}°\nCOP = {:.7f}\n\n".format(peak_center, peak_error,
                                                                                               peak_max, COP))
        else:
            file_angle_distribution.write("COP = {:.7f}\n\n".format(COP))

        file_angle_distribution.write("Peak fitting data: {}\n\n".format(peaks))

        file_angle_distribution.write("Angle distribution data:\n\n")
        file_angle_distribution.write("Angle Intensity\n")
        for i in range(np.shape(x1)[0]):
            file_angle_distribution.write("{:.0f} {}\n".format(x1[i], ave_intensity[i]))

        fo.close()
        print("Angle distribution data saved successfully!")

    buttun_save_angle_distribution_data = tk.Button(window_save, text='Save Angle Distribution Data',
                                                    font=('Arial', 12), height=1, width=30,
                                                    command=save_angle_distribution_data)
    buttun_save_angle_distribution_data.place(x=30, y=210, anchor='w')

    def save_all():
        save_reshaped_image()
        save_masked_FFT()
        save_angle_distribution_plot()
        save_angle_distribution_data()
        print("---------saved all-----------")

    buttun_save_all = tk.Button(window_save, text='Save All', font=('Arial', 12), height=1, width=30, command=save_all)
    buttun_save_all.place(x=30, y=270, anchor='w')

    # exit save window
    buttun_exit_save = tk.Button(window_save, text='Exit', font=('Arial', 12), height=1, width=30,
                                 command=window_save.destroy)
    buttun_exit_save.place(x=30, y=310, anchor='w')


buttun_save_image = tk.Button(window, text='Save', font=('Arial', 12), height=1, width=30, command=save_it)
buttun_save_image.place(x=300, y=350, anchor='w')

# partial calculate paramater
lab_parital_calculate = tk.Label(window, text='Parital Calculate:', font=('Arial', 12))
lab_parital_calculate.place(x=700, y=30, anchor='w')

lab_x_part = tk.Label(window, text='# x:', font=('Arial', 12))
lab_x_part.place(x=640, y=60, anchor='w')

x_part = tk.StringVar(value='8')
entry_x_partial = tk.Entry(window, font=('Arial', 14), width=5, textvariable=x_part)
entry_x_partial.place(x=680, y=60, anchor='w')

lab_y_part = tk.Label(window, text='# y:', font=('Arial', 12))
lab_y_part.place(x=770, y=60, anchor='w')

y_part = tk.StringVar(value='8')
entry_y_part = tk.Entry(window, font=('Arial', 14), width=5, textvariable=y_part)
entry_y_part.place(x=810, y=60, anchor='w')

most_cut_check = tk.IntVar(value=0)
# most_cut_check_button=tk.Checkbutton(window,text='Most Cut',variable=most_cut_check,onvalue=1,offvalue=0,font=('Arial', 12))
# most_cut_check_button.place(x=640,y=100,anchor='w')
#
# lab_cut_offset=tk.Label(window, text='Offset:', font=('Arial', 12))
# lab_cut_offset.place(x=750,y=100,anchor='w')
#
# cut_offset=tk.DoubleVar(value=0.5)
# entry_cut_offset=tk.Entry(window, font=('Arial', 14),width=5,textvariable=cut_offset)
# entry_cut_offset.place(x=810,y=100,anchor='w')

partial_image_output = tk.IntVar(value=1)
partial_image_output_button = tk.Checkbutton(window, text='Partial image output', variable=partial_image_output,
                                             onvalue=1, offvalue=0, font=('Arial', 12))
partial_image_output_button.place(x=640, y=100, anchor='w')

lab_partial_angle_step = tk.Label(window, text='Angle step:', font=('Arial', 12))
lab_partial_angle_step.place(x=640, y=140, anchor='w')

partial_angle_step = tk.StringVar(value='5')
entry_partial_angle_step = tk.Entry(window, font=('Arial', 14), width=5, textvariable=partial_angle_step)
entry_partial_angle_step.place(x=810, y=140, anchor='w')

lab_partial_min_prominence = tk.Label(window, text='Min prominence:', font=('Arial', 12))
lab_partial_min_prominence.place(x=640, y=180, anchor='w')

partial_min_prominence = tk.StringVar(value='0.7')
entry_partial_min_prominence = tk.Entry(window, font=('Arial', 14), width=5, textvariable=partial_min_prominence)
entry_partial_min_prominence.place(x=810, y=180, anchor='w')

lab_partial_min_width = tk.Label(window, text='Min width:', font=('Arial', 12))
lab_partial_min_width.place(x=640, y=220, anchor='w')

partial_min_width = tk.StringVar(value='5')
entry_partial_min_width = tk.Entry(window, font=('Arial', 14), width=5, textvariable=partial_min_width)
entry_partial_min_width.place(x=810, y=220, anchor='w')


# part get number
def calculate_most_part():
    # if not select "most cut", use the #x#y setting
    if most_cut_check.get() == 0:
        return int(x_part.get()), int(y_part.get())
    # if select "most cut", calculate #x#y
    part_offset = cut_offset.get()
    temp_x = 4
    # temp_y=int(np.shape(image_load)[1]/(np.shape(image_load)[0]/temp_x))
    temp_y = 4
    differ_value = 0  # to calculate the differ of the maximum and minimum of part number

    raw_mask_inside_radius = float(mask_inside_radius.get())
    raw_mask_outside_radius = float(mask_outside_radius.get())
    temp_angle_step = float(partial_angle_step.get())
    part_sum = int(180 / temp_angle_step)

    # for temp_x
    while differ_value < part_offset:
        temp_x = temp_x + 1
        temp_image = np.ones([int(np.shape(image_load)[0] / temp_x), int(np.shape(image_load)[1] / temp_y)])

        temp_inside_r = raw_mask_inside_radius / max(temp_x, temp_y)
        temp_outside_r = raw_mask_outside_radius / max(temp_x, temp_y)

        # set temp_image as a mask
        # specify circle parameters: centre ij
        ci, cj = np.shape(temp_image)[0] / 2, np.shape(temp_image)[1] / 2

        # Create index arrays to z
        I, J = np.meshgrid(np.arange(temp_image.shape[1]), np.arange(temp_image.shape[0]))

        # calculate distance of all points to centre
        dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

        # Assign value of 0 to those points where dist<cr:
        temp_image[np.where(dist < temp_inside_r)] = 0

        # mask of masked FFT
        temp_image[np.where(dist > temp_outside_r)] = 0

        center = [float((np.shape(temp_image)[0] - 1) / 2), float((np.shape(temp_image)[1] - 1) / 2)]
        partial_sum_number = np.zeros(part_sum)

        # test
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(temp_image), cmap='gray')
        plt.show()

        for i in range(np.shape(temp_image)[0]):
            for j in range(np.shape(temp_image)[1]):

                y_0 = i - center[0]
                x_0 = j - center[1]

                if temp_image[i][j] == 1:
                    if j - center[1] == 0:
                        part_number = int(part_sum / 2)
                        partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                    else:
                        angle = np.arctan(y_0 / x_0) / np.pi * 180

                        part_number = round(angle / temp_angle_step)
                        if part_number < 0:
                            part_number = part_number + part_sum
                        if part_number >= part_sum:
                            part_number = part_number - part_sum
                        partial_sum_number[part_number] = partial_sum_number[part_number] + 1
        differ_value = ((np.amax(partial_sum_number) - np.amin(partial_sum_number)) / np.amin(partial_sum_number))

        print("x:{},y:{},differ:{}".format(temp_x, temp_y, differ_value))

        if temp_x > 100:
            return 0
    temp_x_save = temp_x - 1
    temp_x = 4
    differ_value = 0

    # for temp_y
    while differ_value < part_offset:
        temp_y = temp_y + 1
        temp_image = np.ones([int(np.shape(image_load)[0] / temp_x), int(np.shape(image_load)[1] / temp_y)])

        temp_inside_r = raw_mask_inside_radius / max(temp_x, temp_y)
        temp_outside_r = raw_mask_outside_radius / max(temp_x, temp_y)

        # set temp_image as a mask
        # specify circle parameters: centre ij
        ci, cj = np.shape(temp_image)[0] / 2, np.shape(temp_image)[1] / 2

        # Create index arrays to z
        I, J = np.meshgrid(np.arange(temp_image.shape[1]), np.arange(temp_image.shape[0]))

        # calculate distance of all points to centre
        dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

        # Assign value of 0 to those points where dist<cr:
        temp_image[np.where(dist < temp_inside_r)] = 0

        # mask of masked FFT
        temp_image[np.where(dist > temp_outside_r)] = 0

        center = [float((np.shape(temp_image)[0] - 1) / 2), float((np.shape(temp_image)[1] - 1) / 2)]
        partial_sum_number = np.zeros(part_sum)

        # test
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(temp_image), cmap='gray')
        plt.show

        for i in range(np.shape(temp_image)[0]):
            for j in range(np.shape(temp_image)[1]):

                y_0 = i - center[0]
                x_0 = j - center[1]

                if temp_image[i][j] == 1:
                    if j - center[1] == 0:
                        part_number = int(part_sum / 2)
                        partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                    else:
                        angle = np.arctan(y_0 / x_0) / np.pi * 180

                        part_number = round(angle / temp_angle_step)
                        if part_number < 0:
                            part_number = part_number + part_sum
                        if part_number >= part_sum:
                            part_number = part_number - part_sum
                        partial_sum_number[part_number] = partial_sum_number[part_number] + 1
        differ_value = ((np.amax(partial_sum_number) - np.amin(partial_sum_number)) / np.amin(partial_sum_number))
        print("x:{},y:{},differ:{}".format(temp_x, temp_y, differ_value))
        if temp_y > 100:
            return 0
    temp_y_save = temp_y - 1

    # check temp_y_save and temp_x_save
    temp_image = np.ones([int(np.shape(image_load)[0] / temp_x_save), int(np.shape(image_load)[1] / temp_y_save)])

    temp_inside_r = raw_mask_inside_radius / max(temp_x_save, temp_y_save)
    temp_outside_r = raw_mask_outside_radius / max(temp_x_save, temp_y_save)

    # set temp_image as a mask
    # specify circle parameters: centre ij
    ci, cj = np.shape(temp_image)[0] / 2, np.shape(temp_image)[1] / 2

    # Create index arrays to z
    I, J = np.meshgrid(np.arange(temp_image.shape[1]), np.arange(temp_image.shape[0]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

    # Assign value of 0 to those points where dist<cr:
    temp_image[np.where(dist < temp_inside_r)] = 0

    # mask of masked FFT
    temp_image[np.where(dist > temp_outside_r)] = 0

    center = [float((np.shape(temp_image)[0] - 1) / 2), float((np.shape(temp_image)[1] - 1) / 2)]
    partial_sum_number = np.zeros(part_sum)

    # test
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(temp_image), cmap='gray')
    plt.show()

    for i in range(np.shape(temp_image)[0]):
        for j in range(np.shape(temp_image)[1]):

            y_0 = i - center[0]
            x_0 = j - center[1]

            if temp_image[i][j] == 1:
                if j - center[1] == 0:
                    part_number = int(part_sum / 2)
                    partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                else:
                    angle = np.arctan(y_0 / x_0) / np.pi * 180

                    part_number = round(angle / temp_angle_step)
                    if part_number < 0:
                        part_number = part_number + part_sum
                    if part_number >= part_sum:
                        part_number = part_number - part_sum
                    partial_sum_number[part_number] = partial_sum_number[part_number] + 1
    differ_value = ((np.amax(partial_sum_number) - np.amin(partial_sum_number)) / np.amin(partial_sum_number))
    print("x:{},y:{},differ:{}".format(temp_x_save, temp_y_save, differ_value))

    return temp_x_save, temp_y_save


def calculate_most_part_1():
    # if not select "most cut", use the #x#y setting
    if most_cut_check.get() == 0:
        return int(x_part.get()), int(y_part.get())
    # if select "most cut", calculate #x#y
    part_offset = cut_offset.get()
    differ_value = 0  # to calculate the differ of the maximum and minimum of part number

    raw_mask_inside_radius = float(mask_inside_radius.get())
    raw_mask_outside_radius = float(mask_outside_radius.get())
    temp_angle_step = float(partial_angle_step.get())
    part_sum = int(180 / temp_angle_step)

    differ_matrix = np.ones([30, 30]) * 10
    select_point = np.array([0, 0])  # save the select point of cut part number

    for temp_x in range(4, 30):
        for temp_y in range(4, 30):
            temp_image = np.ones([int(np.shape(image_load)[0] / temp_x), int(np.shape(image_load)[1] / temp_y)])

            round_shrink = np.shape(temp_image)[0] / np.amin(
                [np.shape(temp_image)[0] / temp_x, np.shape(temp_image)[1] / temp_y])
            temp_inside_r = raw_mask_inside_radius / round_shrink
            temp_outside_r = raw_mask_outside_radius / round_shrink

            # set temp_image as a mask
            # specify circle parameters: centre ij
            ci, cj = np.shape(temp_image)[0] / 2, np.shape(temp_image)[1] / 2

            # Create index arrays to z
            I, J = np.meshgrid(np.arange(temp_image.shape[1]), np.arange(temp_image.shape[0]))

            # calculate distance of all points to centre
            dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

            # Assign value of 0 to those points where dist<cr:
            temp_image[np.where(dist < temp_inside_r)] = 0

            # mask of masked FFT
            temp_image[np.where(dist > temp_outside_r)] = 0

            center = [float((np.shape(temp_image)[0] - 1) / 2), float((np.shape(temp_image)[1] - 1) / 2)]
            partial_sum_number = np.zeros(part_sum)

            for i in range(np.shape(temp_image)[0]):
                for j in range(np.shape(temp_image)[1]):

                    y_0 = i - center[0]
                    x_0 = j - center[1]

                    if temp_image[i][j] == 1:
                        if j - center[1] == 0:
                            part_number = int(part_sum / 2)
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                        else:
                            angle = np.arctan(y_0 / x_0) / np.pi * 180

                            part_number = round(angle / temp_angle_step)
                            if part_number < 0:
                                part_number = part_number + part_sum
                            if part_number >= part_sum:
                                part_number = part_number - part_sum
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
            if np.amin(partial_sum_number) == 0:  # useless
                differ_value = 10
            else:
                differ_value = (
                        (np.amax(partial_sum_number) - np.amin(partial_sum_number)) / np.amin(partial_sum_number))
                if differ_value > part_offset:  # useless
                    differ_value = 10
                else:
                    if select_point[0] * select_point[1] < temp_x * temp_y:  # choose the one which have the most part
                        select_point = np.array([temp_x, temp_y])

            differ_matrix[temp_x, temp_y] = differ_value

    # test
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(differ_matrix), cmap='gray')
    plt.show()

    # find minimum

    # test 2
    temp_x = select_point[0]
    temp_y = select_point[1]
    temp_image = np.ones([int(np.shape(image_load)[0] / temp_x), int(np.shape(image_load)[1] / temp_y)])

    round_shrink = np.shape(temp_image)[0] / np.amin(
        [np.shape(temp_image)[0] / temp_x, np.shape(temp_image)[1] / temp_y])
    temp_inside_r = raw_mask_inside_radius / round_shrink
    temp_outside_r = raw_mask_outside_radius / round_shrink

    # set temp_image as a mask
    # specify circle parameters: centre ij
    ci, cj = np.shape(temp_image)[0] / 2, np.shape(temp_image)[1] / 2

    # Create index arrays to z
    I, J = np.meshgrid(np.arange(temp_image.shape[1]), np.arange(temp_image.shape[0]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

    # Assign value of 0 to those points where dist<cr:
    temp_image[np.where(dist < temp_inside_r)] = 0

    # mask of masked FFT
    temp_image[np.where(dist > temp_outside_r)] = 0

    center = [float((np.shape(temp_image)[0] - 1) / 2), float((np.shape(temp_image)[1] - 1) / 2)]
    partial_sum_number = np.zeros(part_sum)

    # test
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(temp_image), cmap='gray')
    plt.show()

    for i in range(np.shape(temp_image)[0]):
        for j in range(np.shape(temp_image)[1]):

            y_0 = i - center[0]
            x_0 = j - center[1]

            if temp_image[i][j] == 1:
                if j - center[1] == 0:
                    part_number = int(part_sum / 2)
                    partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                else:
                    angle = np.arctan(y_0 / x_0) / np.pi * 180

                    part_number = round(angle / temp_angle_step)
                    if part_number < 0:
                        part_number = part_number + part_sum
                    if part_number >= part_sum:
                        part_number = part_number - part_sum
                    partial_sum_number[part_number] = partial_sum_number[part_number] + 1
    # test
    plt.figure(figsize=(10, 8))
    x0 = np.array(range(part_sum)) / part_sum * 180
    plt.title("pixel number")
    plt.plot(x0, partial_sum_number)
    plt.show()
    print("total pixel numer:{}".format(np.sum(partial_sum_number)))

    differ_value = ((np.amax(partial_sum_number) - np.amin(partial_sum_number)) / np.amin(partial_sum_number))
    print("x:{},y:{},differ:{}".format(temp_x, temp_y, differ_value))

    return select_point[0], select_point[1]


# show partial image
def show_partial_image():
    global partial_image
    global x_part_number
    global y_part_number
    x_part_number, y_part_number = calculate_most_part_1()
    print("Cut number:{}*{}".format(x_part_number, y_part_number))

    partial_image = []

    for i in range(x_part_number):
        for j in range(y_part_number):
            partial_image.append(image_load[int(np.shape(image_load)[0] * i / x_part_number):int(
                np.shape(image_load)[0] * (i + 1) / x_part_number),
                                 int(np.shape(image_load)[1] * j / y_part_number):int(
                                     np.shape(image_load)[1] * (j + 1) / y_part_number)])

    if (partial_image_output.get()) == 1:
        start_time = time.time()
        # show partial image
        plt.figure(figsize=(10, 8))

        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_image[i * y_part_number + j]), cmap='gray')
                progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

        plt.suptitle('Partial Image ( {} * {} )'.format(x_part_number, y_part_number))
        plt.show()
    print("\n-------------------------Partial image shown----------------------------\n")


buttun_show_partial_image = tk.Button(window, text='Show Parital Image', font=('Arial', 12), height=1, width=25,
                                      command=show_partial_image)
buttun_show_partial_image.place(x=640, y=260, anchor='w')


# show partial FFT
def show_partial_FFT():
    global partial_FFT
    global partial_mask

    partial_FFT = []
    partial_mask = []

    inside_r_precentage = float(mask_inside_radius.get())
    outside_r_precentage = float(mask_outside_radius.get())

    start_time = time.time()
    for i in range(x_part_number):
        for j in range(y_part_number):
            partial_fftImage = np.fft.fft2(partial_image[i * y_part_number + j])
            partial_fftShiftImage = np.fft.fftshift(partial_fftImage)
            partial_fftMagImage = np.abs(partial_fftShiftImage)

            inside_r = np.amin(np.shape(partial_fftMagImage)) * inside_r_precentage / 100 / 2
            outside_r = np.amin(np.shape(partial_fftMagImage)) * outside_r_precentage / 100 / 2

            # mask of center masked FFT
            partial_center_mask = np.ones(np.shape(partial_fftMagImage))

            # specify circle parameters: centre ij
            ci, cj = np.shape(partial_center_mask)[0] / 2, np.shape(partial_center_mask)[1] / 2

            # Create index arrays to z
            I, J = np.meshgrid(np.arange(partial_center_mask.shape[1]), np.arange(partial_center_mask.shape[0]))

            # calculate distance of all points to centre
            dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

            # Assign value of 0 to those points where dist<cr:
            partial_center_mask[np.where(dist < inside_r)] = 0

            # mask of masked FFT
            partial_whole_mask = partial_center_mask.copy()
            partial_whole_mask[np.where(dist > outside_r)] = 0

            masked_partial_FFT = np.multiply(partial_whole_mask, partial_fftMagImage)

            partial_mask.append(partial_whole_mask)
            partial_FFT.append(masked_partial_FFT)
            progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

    print("\n-------------------------Partial FFT calculated----------------------------\n")

    # show partial FFT
    if (partial_image_output.get()) == 1:
        plt.figure(figsize=(10, 8))
        start_time = time.time()

        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_FFT[i * y_part_number + j]), cmap='gray')
                progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

        plt.suptitle('Partial FFT ( {} * {} )'.format(x_part_number, y_part_number))
        plt.show()
        print("\n-------------------------Partial FFT shown----------------------------\n")


buttun_show_partial_FFT = tk.Button(window, text='Show Partial FFT', font=('Arial', 12), height=1, width=25,
                                    command=show_partial_FFT)
buttun_show_partial_FFT.place(x=640, y=300, anchor='w')


# Fit partial direction
def fit_partial_direction():
    global partial_direciton
    global partial_peak_number_sum
    global partial_peak_sum
    global partial_curve

    partial_peak_number_sum = np.zeros([x_part_number, y_part_number])
    partial_direciton = np.zeros([x_part_number, y_part_number])
    partial_intensity_sum = np.zeros([x_part_number, y_part_number])

    part_sum = int(180 / float(partial_angle_step.get()))

    partial_direction = []
    partial_peak_sum = []
    partial_curve = []

    fig_partial_direction, ax_partial_direction = plt.subplots(figsize=(10, 8))
    ax_partial_direction.axis('off')
    x1 = np.array(range(part_sum * 2)) / part_sum * 180
    ax_partial_direction_list = [[None for _ in range(y_part_number)] for _ in range(x_part_number)]

    start_time = time.time()
    for i in range(x_part_number):
        for j in range(y_part_number):
            center = [float((np.shape(partial_FFT[i * y_part_number + j])[0] - 1) / 2),
                      float((np.shape(partial_FFT[i * y_part_number + j])[1] - 1) / 2)]

            partial_sum_intensity = np.zeros(part_sum * 2)
            partial_sum_number = np.zeros(part_sum * 2)
            partial_ave_intensity = np.zeros(part_sum * 2)

            # main part to sum intensity
            for ii in range(np.shape(partial_FFT[i * y_part_number + j])[0]):
                for jj in range(np.shape(partial_FFT[i * y_part_number + j])[1]):

                    y_0 = ii - center[0]
                    x_0 = jj - center[1]

                    if partial_mask[i * y_part_number + j][ii][jj] == 1:
                        if jj - center[1] == 0:
                            part_number = int(part_sum / 2)
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]
                        else:
                            angle = np.arctan(y_0 / x_0) / np.pi * 180

                            part_number = round(angle / float(partial_angle_step.get()))
                            if part_number < 0:
                                part_number = part_number + part_sum
                            if part_number >= part_sum:
                                part_number = part_number - part_sum
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]

            for iii in range(part_sum):
                partial_ave_intensity[iii] = partial_sum_intensity[iii] / partial_sum_number[iii]

            for iii in range(part_sum):
                partial_ave_intensity[iii + part_sum] = partial_ave_intensity[iii]

            partial_curve.append(partial_ave_intensity)

            # partial_ave_intensity=np.flipud(partial_ave_intensity)     #???

            # calculate partial direciton
            partial_prominence = float(partial_min_prominence.get())
            partial_width = float(partial_min_width.get()) / float(partial_angle_step.get())
            part_sum = int(180 / float(partial_angle_step.get()))

            COP = 0  # undefine

            partial_prominence_real = (np.amax(partial_ave_intensity) - np.amin(
                partial_ave_intensity)) * partial_prominence

            partial_peaks = find_peaks(partial_ave_intensity, prominence=partial_prominence_real, width=partial_width)
            # print(partial_peaks)
            partial_peak_number_sum[i, j] = np.shape(partial_peaks[0])[0]

            # show fit result
            if np.shape(partial_peaks[0])[0] == 1:
                # print("Peak Number:{}".format(np.shape(partial_peaks[0])[0]))

                partial_peak_max = float(partial_peaks[0]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][0] / part_sum * 180 + partial_peaks[1]['right_ips'][
                    0] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'] / 2) * float(partial_angle_step.get())

                # save direction
                partial_direciton[i, j] = partial_peak_center

                # show distribution
                ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                    i * y_part_number + j + 1)
                ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                ax_partial_direction_list[i][j].set_xticks([])
                ax_partial_direction_list[i][j].set_yticks([])

                plt.plot(x1, partial_ave_intensity)

                plt.scatter(x1[int(partial_peaks[0])], partial_ave_intensity[int(partial_peaks[0])], c="g", marker="x")
                plt.scatter(partial_peaks[1]['left_ips'] / part_sum * 180, partial_peaks[1]['width_heights'], c="g",
                            marker="v")
                plt.scatter(partial_peaks[1]['right_ips'] / part_sum * 180, partial_peaks[1]['width_heights'], c="g",
                            marker="v")

                if partial_peak_center > 180:
                    plt.text(0, 0.75 * (np.amax(partial_ave_intensity) - np.amin(partial_ave_intensity)) + np.amin(
                        partial_ave_intensity), "{:.2f}".format(partial_peak_center - 180), fontsize=10)
                else:
                    plt.text(0, 0.75 * (np.amax(partial_ave_intensity) - np.amin(partial_ave_intensity)) + np.amin(
                        partial_ave_intensity), "{:.2f}".format(partial_peak_center), fontsize=10)

                # COP=cal_COP(array=ave_intensity,peak_max=float(peaks[0]),step=float(angle_step.get()))
            elif np.shape(partial_peaks[0])[0] == 2:  # if fits two peaks, choose the higher one
                if float(partial_peaks[1]['prominences'][0]) > float(partial_peaks[1]['prominences'][1]):
                    partial_peak_choose = 0
                else:
                    partial_peak_choose = 1
                partial_peak_max = float(partial_peaks[0][partial_peak_choose]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][partial_peak_choose] / part_sum * 180 +
                                       partial_peaks[1]['right_ips'][partial_peak_choose] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'][partial_peak_choose] / 2) * float(
                    partial_angle_step.get())
                if partial_peak_center > 180:
                    partial_direciton[i, j] = partial_peak_center - 180
                else:
                    partial_direciton[i, j] = partial_peak_center

                # show distribution
                ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                    i * y_part_number + j + 1)
                ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                ax_partial_direction_list[i][j].set_xticks([])
                ax_partial_direction_list[i][j].set_yticks([])

                plt.plot(x1, partial_ave_intensity)

                plt.scatter(x1[int(partial_peaks[0][partial_peak_choose])],
                            partial_ave_intensity[int(partial_peaks[0][partial_peak_choose])], c="g", marker="x")
                plt.scatter(partial_peaks[1]['left_ips'][partial_peak_choose] / part_sum * 180,
                            partial_peaks[1]['width_heights'][partial_peak_choose], c="g", marker="v")
                plt.scatter(partial_peaks[1]['right_ips'][partial_peak_choose] / part_sum * 180,
                            partial_peaks[1]['width_heights'][partial_peak_choose], c="g", marker="v")

                if partial_peak_center > 180:
                    plt.text(0, 0.75 * (np.amax(partial_ave_intensity) - np.amin(partial_ave_intensity)) + np.amin(
                        partial_ave_intensity), "{:.2f}".format(partial_peak_center - 180), fontsize=10)
                else:
                    plt.text(0, 0.75 * (np.amax(partial_ave_intensity) - np.amin(partial_ave_intensity)) + np.amin(
                        partial_ave_intensity), "{:.2f}".format(partial_peak_center), fontsize=10)

                # COP=cal_COP(array=ave_intensity,peak_max=float(peaks[0][peak_choose]),step=float(angle_step.get()))
            else:
                # print("Peak Number:{}".format(np.shape(partial_peaks[0])[0]))

                # show distribution
                ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                    i * y_part_number + j + 1)
                ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                ax_partial_direction_list[i][j].set_xticks([])
                ax_partial_direction_list[i][j].set_yticks([])

                plt.plot(x1, partial_ave_intensity)

                for ix in range(np.shape(partial_peaks[0])[0]):
                    plt.scatter(x1[int(partial_peaks[0][ix])], partial_ave_intensity[int(partial_peaks[0][ix])], c="g",
                                marker="x")
                    plt.scatter(partial_peaks[1]['left_ips'][ix] / part_sum * 180,
                                partial_peaks[1]['width_heights'][ix], c="g", marker="v")
                    plt.scatter(partial_peaks[1]['right_ips'][ix] / part_sum * 180,
                                partial_peaks[1]['width_heights'][ix], c="g", marker="v")

                plt.text(0, 0.75 * (np.amax(partial_ave_intensity) - np.amin(partial_ave_intensity)) + np.amin(
                    partial_ave_intensity), "#:{}".format(np.shape(partial_peaks[0])[0]), fontsize=10, c="r")

            partial_peak_sum.append(partial_peaks)
            progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

    fig_partial_direction.suptitle('Partial Curve ( {} * {} )'.format(x_part_number, y_part_number))
    fig_partial_direction.show()

    # output pixel number-angle relationship (test part)
    # plt.figure(figsize=(10,8))
    # x0=np.array(range(part_sum))/part_sum*180
    # plt.title("pixel number")
    # plt.plot(x0,partial_sum_number[0:int(np.shape(partial_sum_number)[0]/2)])
    # plt.show()
    # print("total pixel numer:{}".format(np.sum(partial_sum_number)))

    partial_sum_number_1 = partial_sum_number[0:int(np.shape(partial_sum_number)[0] / 2)]

    # send alart if the pixel number differ too much
    # if ((np.amax(partial_sum_number_1)-np.amin(partial_sum_number_1))/np.amin(partial_sum_number_1))>0.4:
    #    tk.messagebox.showwarning(title='Pixel number differ too much', message='Max pixel number = {}\nMin pixel number = {}'.format(np.amax(partial_sum_number_1),np.amin(partial_sum_number_1)))

    # print("Sum intensity:")
    # print(partial_intensity_sum)
    # plt.figure(figsize=(10,8))
    # plt.imshow(np.abs(partial_intensity_sum), cmap='gray')
    # plt.show()

    print("\n-------------------------Partial Direction Fitted----------------------------\n")


def fit_partial_direction_1():
    global partial_direciton
    global partial_peak_number_sum
    global partial_peak_sum
    global partial_curve

    partial_peak_number_sum = np.zeros([x_part_number, y_part_number])
    partial_direciton = np.zeros([x_part_number, y_part_number])
    partial_intensity_sum = np.zeros([x_part_number, y_part_number])

    part_sum = int(180 / float(partial_angle_step.get()))

    partial_direction = []
    partial_peak_sum = []
    partial_curve = []

    start_time = time.time()
    for i in range(x_part_number):
        for j in range(y_part_number):
            center = [float((np.shape(partial_FFT[i * y_part_number + j])[0] - 1) / 2),
                      float((np.shape(partial_FFT[i * y_part_number + j])[1] - 1) / 2)]

            partial_sum_intensity = np.zeros(part_sum * 2)
            partial_sum_number = np.zeros(part_sum * 2)
            partial_ave_intensity = np.zeros(part_sum * 2)

            # main part to sum intensity
            for ii in range(np.shape(partial_FFT[i * y_part_number + j])[0]):
                for jj in range(np.shape(partial_FFT[i * y_part_number + j])[1]):

                    y_0 = ii - center[0]
                    x_0 = jj - center[1]

                    if partial_mask[i * y_part_number + j][ii][jj] == 1:
                        if jj - center[1] == 0:
                            part_number = int(part_sum / 2)
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]
                        else:
                            angle = np.arctan(y_0 / x_0) / np.pi * 180

                            part_number = round(angle / float(partial_angle_step.get()))
                            if part_number < 0:
                                part_number = part_number + part_sum
                            if part_number >= part_sum:
                                part_number = part_number - part_sum
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]

            for iii in range(part_sum):
                partial_ave_intensity[iii] = partial_sum_intensity[iii] / partial_sum_number[iii]

            for iii in range(part_sum):
                partial_ave_intensity[iii + part_sum] = partial_ave_intensity[iii]

            partial_curve.append(partial_ave_intensity)

            # partial_ave_intensity=np.flipud(partial_ave_intensity)     #???

            # calculate partial direciton
            partial_prominence = float(partial_min_prominence.get())
            partial_width = float(partial_min_width.get()) / float(partial_angle_step.get())
            part_sum = int(180 / float(partial_angle_step.get()))

            COP = 0  # undefine

            partial_prominence_real = (np.amax(partial_ave_intensity) - np.amin(
                partial_ave_intensity)) * partial_prominence

            partial_peaks = find_peaks(partial_ave_intensity, prominence=partial_prominence_real, width=partial_width)
            # print(partial_peaks)
            partial_peak_number_sum[i, j] = np.shape(partial_peaks[0])[0]

            # show fit result
            if np.shape(partial_peaks[0])[0] == 1:
                # print("Peak Number:{}".format(np.shape(partial_peaks[0])[0]))

                partial_peak_max = float(partial_peaks[0]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][0] / part_sum * 180 + partial_peaks[1]['right_ips'][
                    0] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'] / 2) * float(partial_angle_step.get())

                # save direction
                partial_direciton[i, j] = partial_peak_center

            elif np.shape(partial_peaks[0])[0] == 2:  # if fits two peaks, choose the higher one
                if float(partial_peaks[1]['prominences'][0]) > float(partial_peaks[1]['prominences'][1]):
                    partial_peak_choose = 0
                else:
                    partial_peak_choose = 1

                partial_peak_max = float(partial_peaks[0][partial_peak_choose]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][partial_peak_choose] / part_sum * 180 +
                                       partial_peaks[1]['right_ips'][partial_peak_choose] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'][partial_peak_choose] / 2) * float(
                    partial_angle_step.get())

                # save direction
                if partial_peak_center > 180:
                    partial_direciton[i, j] = partial_peak_center - 180
                else:
                    partial_direciton[i, j] = partial_peak_center

            partial_peak_sum.append(partial_peaks)
            progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)
    print("\n-------------------------Partial Direction Fitted----------------------------\n")

    # show partial fit
    if (partial_image_output.get()) == 1:
        fig_partial_direction, ax_partial_direction = plt.subplots(figsize=(10, 8))
        ax_partial_direction.axis('off')
        x1 = np.array(range(part_sum * 2)) / part_sum * 180
        ax_partial_direction_list = [[None for _ in range(y_part_number)] for _ in range(x_part_number)]

        start_time = time.time()

        for i in range(x_part_number):
            for j in range(y_part_number):
                if np.shape(partial_peak_sum[i * y_part_number + j][0])[0] == 1:
                    # show distribution
                    ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                        i * y_part_number + j + 1)
                    ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                    ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                    ax_partial_direction_list[i][j].set_xticks([])
                    ax_partial_direction_list[i][j].set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0])],
                                partial_curve[i * y_part_number + j][int(partial_peak_sum[i * y_part_number + j][0])],
                                c="g", marker="x")
                    plt.scatter(partial_peak_sum[i * y_part_number + j][1]['left_ips'] / part_sum * 180,
                                partial_peak_sum[i * y_part_number + j][1]['width_heights'], c="g", marker="v")
                    plt.scatter(partial_peak_sum[i * y_part_number + j][1]['right_ips'] / part_sum * 180,
                                partial_peak_sum[i * y_part_number + j][1]['width_heights'], c="g", marker="v")

                    if partial_peak_center > 180:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j] - 180), fontsize=10)
                    else:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j]), fontsize=10)

                    # COP=cal_COP(array=ave_intensity,peak_max=float(peaks[0]),step=float(angle_step.get()))

                elif np.shape(partial_peak_sum[i * y_part_number + j][0])[0] == 2:
                    if float(partial_peak_sum[i * y_part_number + j][1]['prominences'][0]) > float(
                            partial_peak_sum[i * y_part_number + j][1]['prominences'][1]):
                        partial_peak_choose = 0
                    else:
                        partial_peak_choose = 1

                    # show distribution
                    ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                        i * y_part_number + j + 1)
                    ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                    ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                    ax_partial_direction_list[i][j].set_xticks([])
                    ax_partial_direction_list[i][j].set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0][partial_peak_choose])],
                                partial_curve[i * y_part_number + j][
                                    int(partial_peak_sum[i * y_part_number + j][0][partial_peak_choose])], c="g",
                                marker="x")
                    plt.scatter(
                        partial_peak_sum[i * y_part_number + j][1]['left_ips'][partial_peak_choose] / part_sum * 180,
                        partial_peak_sum[i * y_part_number + j][1]['width_heights'][partial_peak_choose], c="g",
                        marker="v")
                    plt.scatter(
                        partial_peak_sum[i * y_part_number + j][1]['right_ips'][partial_peak_choose] / part_sum * 180,
                        partial_peak_sum[i * y_part_number + j][1]['width_heights'][partial_peak_choose], c="g",
                        marker="v")

                    if partial_peak_center > 180:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j] - 180), fontsize=10)
                    else:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j]), fontsize=10)

                    # COP=cal_COP(array=ave_intensity,peak_max=float(peaks[0][peak_choose]),step=float(angle_step.get()))

                else:
                    # show distribution
                    ax_partial_direction_list[i][j] = fig_partial_direction.add_subplot(x_part_number, y_part_number,
                                                                                        i * y_part_number + j + 1)
                    ax_partial_direction_list[i][j].xaxis.set_tick_params(labelbottom=False)
                    ax_partial_direction_list[i][j].yaxis.set_tick_params(labelleft=False)

                    ax_partial_direction_list[i][j].set_xticks([])
                    ax_partial_direction_list[i][j].set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    for ix in range(np.shape(partial_peak_sum[i * y_part_number + j][0])[0]):
                        plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0][ix])],
                                    partial_curve[i * y_part_number + j][
                                        int(partial_peak_sum[i * y_part_number + j][0][ix])], c="g", marker="x")
                        plt.scatter(partial_peak_sum[i * y_part_number + j][1]['left_ips'][ix] / part_sum * 180,
                                    partial_peak_sum[i * y_part_number + j][1]['width_heights'][ix], c="g", marker="v")
                        plt.scatter(partial_peak_sum[i * y_part_number + j][1]['right_ips'][ix] / part_sum * 180,
                                    partial_peak_sum[i * y_part_number + j][1]['width_heights'][ix], c="g", marker="v")

                    plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                        partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                             "#:{}".format(np.shape(partial_peak_sum[i * y_part_number + j][0])[0]), fontsize=10, c="r")
                progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

        fig_partial_direction.suptitle('Partial Curve ( {} * {} )'.format(x_part_number, y_part_number))
        fig_partial_direction.show()

    # output pixel number-angle relationship (test part)
    # plt.figure(figsize=(10,8))
    # x0=np.array(range(part_sum))/part_sum*180
    # plt.title("pixel number")
    # plt.plot(x0,partial_sum_number[0:int(np.shape(partial_sum_number)[0]/2)])
    # plt.show()
    # print("total pixel numer:{}".format(np.sum(partial_sum_number)))

    partial_sum_number_1 = partial_sum_number[0:int(np.shape(partial_sum_number)[0] / 2)]

    # send alart if the pixel number differ too much
    # if ((np.amax(partial_sum_number_1)-np.amin(partial_sum_number_1))/np.amin(partial_sum_number_1))>0.4:
    #    tk.messagebox.showwarning(title='Pixel number differ too much', message='Max pixel number = {}\nMin pixel number = {}'.format(np.amax(partial_sum_number_1),np.amin(partial_sum_number_1)))

    # print("Sum intensity:")
    # print(partial_intensity_sum)
    # plt.figure(figsize=(10,8))
    # plt.imshow(np.abs(partial_intensity_sum), cmap='gray')
    # plt.show()


buttun_fit_partial_direction = tk.Button(window, text='Fit Partial direction', font=('Arial', 12), height=1, width=25,
                                         command=fit_partial_direction_1)
buttun_fit_partial_direction.place(x=640, y=340, anchor='w')


# show FFT direction
def show_FFT_direction():
    # show peak number for every pary in an array
    print("partial_peak_number_sum:\n{}\n".format(partial_peak_number_sum))
    print("partial_direciton:\n{}\n".format(partial_direciton))

    plt.figure(figsize=(10, 8))
    start_time = time.time()

    for i in range(x_part_number):
        for j in range(y_part_number):
            plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
            plt.axis('off')
            plt.imshow(np.abs(partial_FFT[i * y_part_number + j]), cmap='gray')
            # draw arrow only when the direction is recognized
            if (partial_peak_number_sum[i, j] != 1) & (partial_peak_number_sum[i, j] != 2) & (
                    partial_peak_number_sum[i, j] != (-1)):
                continue

            r_circle = min(np.shape(partial_FFT[i * y_part_number + j])) * 0.8 / 2
            arrow_begin = (np.shape(partial_FFT[i * y_part_number + j])[1] / 2 + np.cos(
                partial_direciton[i, j] / 180 * np.pi) * r_circle,
                           np.shape(partial_FFT[i * y_part_number + j])[0] / 2 + np.sin(
                               partial_direciton[i, j] / 180 * np.pi) * r_circle)
            arrow_end = (np.shape(partial_FFT[i * y_part_number + j])[1] / 2 - np.cos(
                partial_direciton[i, j] / 180 * np.pi) * r_circle,
                         np.shape(partial_FFT[i * y_part_number + j])[0] / 2 - np.sin(
                             partial_direciton[i, j] / 180 * np.pi) * r_circle)

            plt.annotate(text='', xy=arrow_begin, xytext=arrow_end,
                         arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

    plt.suptitle('Partial FFT Direction ( {} * {} )'.format(x_part_number, y_part_number))
    plt.show()
    print("\n-------------------------FFT direction shown----------------------------\n")


# buttun_show_FFT_direction = tk.Button(window, text='Show FFT Direction', font=('Arial', 12), height=1,width=25, command=show_FFT_direction)
# buttun_show_FFT_direction.place(x=640,y=340,anchor='w')

# show image direction
def show_partial_real_direction():
    fig_partial_real_direction, ax_partial_real_direction = plt.subplots(figsize=(10, 8))
    ax_partial_real_direction.axis('off')
    ax_partial_real_direction_list = [[None for _ in range(y_part_number)] for _ in range(x_part_number)]
    start_time = time.time()

    for i in range(x_part_number):
        for j in range(y_part_number):
            ax_partial_real_direction_list[i][j] = fig_partial_real_direction.add_subplot(x_part_number, y_part_number,
                                                                                          i * y_part_number + j + 1)
            ax_partial_real_direction_list[i][j].axis('off')
            plt.imshow(np.abs(partial_image[i * y_part_number + j]), cmap='gray')

            # draw arrow only when the direction is recognized
            if (partial_peak_number_sum[i, j] != 1) & (partial_peak_number_sum[i, j] != 2) & (
                    partial_peak_number_sum[i, j] != (-1)):
                continue

            r_circle = min(np.shape(partial_image[i * y_part_number + j])) * 0.8 / 2
            arrow_begin = (np.shape(partial_image[i * y_part_number + j])[1] / 2 + np.sin(
                partial_direciton[i, j] / 180 * np.pi) * r_circle,
                           np.shape(partial_image[i * y_part_number + j])[0] / 2 - np.cos(
                               partial_direciton[i, j] / 180 * np.pi) * r_circle)
            arrow_end = (np.shape(partial_image[i * y_part_number + j])[1] / 2 - np.sin(
                partial_direciton[i, j] / 180 * np.pi) * r_circle,
                         np.shape(partial_image[i * y_part_number + j])[0] / 2 + np.cos(
                             partial_direciton[i, j] / 180 * np.pi) * r_circle)

            plt.annotate(text='', xy=arrow_begin, xytext=arrow_end,
                         arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            progress_bar((i * y_part_number + j + 1), start_time=start_time, total=x_part_number * y_part_number)

    # show standard deviation
    global partial_mean
    global partial_std
    global effective_direction
    global effective_direction_number

    effective_direction_number = 0
    effective_direction = []

    for i in range(x_part_number):
        for j in range(y_part_number):
            if (partial_peak_number_sum[i][j] == 1) or (partial_peak_number_sum[i][j] == 2) or (
            (partial_peak_number_sum[i][j] == (-1))):
                effective_direction.append(partial_direciton[i][j])
                effective_direction_number = effective_direction_number + 1

    partial_mean = np.mean(effective_direction)
    partial_std = np.std(effective_direction)

    # add manual link to change the direction
    def partial_direction_press(event):
        # right click to edit the direction
        if event.button == 3:
            flag_find_subplot = 0
            for i in range(x_part_number):
                for j in range(y_part_number):
                    if event.inaxes == ax_partial_real_direction_list[i][j]:
                        global editing_subplot
                        editing_subplot = [i, j]
                        print("You are clicking the [{},{}] subplot".format(editing_subplot[0] + 1,
                                                                            editing_subplot[1] + 1))

                        flag_find_subplot = 1

                        fig_manual_change_direction, ax_manual_change_direction = plt.subplots(figsize=(10, 8))
                        plt.title("RIGHT click to define the direction")
                        plt.imshow(np.abs(partial_FFT[i * y_part_number + j]), cmap='gray')
                        plt.axis('off')

                        global click_counts
                        global point_save
                        global flag_drawn

                        click_counts = 0
                        point_save = []
                        flag_drawn = 0

                        # draw arrow to manual define the direction
                        def manual_change_direction(event):
                            global click_counts
                            global point_save
                            global point1
                            global manual_arrows
                            global flag_drawn
                            global editing_subplot
                            # right click to draw arrows.
                            if event.button == 3:
                                click_point = np.array([event.xdata, event.ydata])
                                point_save.append(click_point)

                                if click_counts % 2 == 0:
                                    point1 = plt.scatter(click_point[0], click_point[1], color='red')

                                    if click_counts > 0:
                                        manual_arrows.remove()
                                else:
                                    point1.remove()
                                    # draw the arrows
                                    manual_arrows = plt.annotate(text='', xy=point_save[0], xytext=point_save[1],
                                                                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
                                    # calculate the angle
                                    if point_save[0][0] == point_save[1][0]:
                                        manual_angle = 90
                                    else:
                                        manual_angle = np.arctan((point_save[0][1] - point_save[1][1]) / (
                                                    point_save[0][0] - point_save[1][0])) / np.pi * 180

                                    if manual_angle < 0:
                                        manual_angle = manual_angle + 180
                                    print("The angle of [{},{}] subplot have been set to {:.2f}".format(
                                        editing_subplot[0] + 1, editing_subplot[1] + 1, manual_angle))
                                    # save the angle, mark the editted subplot as -1
                                    partial_peak_number_sum[editing_subplot[0]][editing_subplot[1]] = -1
                                    partial_direciton[editing_subplot[0]][editing_subplot[1]] = manual_angle
                                    # once the mark is drawn clear the saved point
                                    point_save = []
                                click_counts = click_counts + 1

                                fig_manual_change_direction.show()

                        cid_manual_change_direction = fig_manual_change_direction.canvas.mpl_connect(
                            'button_press_event', manual_change_direction)

                        fig_manual_change_direction.show()

            if flag_find_subplot == 0:
                print("You are not clicking any subplot")
        # middle button to un-recognize the area
        if event.button == 2:
            flag_find_subplot = 0
            for i in range(x_part_number):
                for j in range(y_part_number):
                    if event.inaxes == ax_partial_real_direction_list[i][j]:
                        flag_give_up = tkinter.messagebox.askokcancel(title='Attention!',
                                                                      message='Are you sure you want to give up the [{},{}] subplot?'.format(
                                                                          i + 1, j + 1))
                        # click 'Yes' to give up
                        if flag_give_up == True:
                            partial_peak_number_sum[i][j] = 0
                            partial_direciton[i][j] = 0
                            print("The angle of [{},{}] subplot have been give up.".format(i + 1, j + 1))
                        # click 'No' to cancle
                        elif flag_give_up == False:
                            print("Nothing happen for [{},{}] subplot".format(i + 1, j + 1))
                        else:
                            print("What happened?")

    cid_partial_direction_press = fig_partial_real_direction.canvas.mpl_connect('button_press_event',
                                                                                partial_direction_press)

    plt.suptitle(
        'Partial Direction ( {} * {} )\n\nMean of {} direction: {:.2f}\nStandard deviation of direction: {:.2f}'.format(
            x_part_number, y_part_number, effective_direction_number, partial_mean, partial_std))
    plt.show()
    print("\n-------------------------Image direction shown----------------------------\n")


# show image of angle deviation
def show_angle_deviation():
    global angle_deviation
    if (np.shape(peaks[0])[0] == 1) or (np.shape(peaks[0])[0] == 2):
        global angle_deviation
        angle_deviation = np.zeros([x_part_number, y_part_number])
        for i in range(np.shape(angle_deviation)[0]):
            for j in range(np.shape(angle_deviation)[1]):
                temp_angle_deviation = partial_direciton[i][j] - peak_center
                if temp_angle_deviation > 90:
                    temp_angle_deviation = temp_angle_deviation - 180
                elif temp_angle_deviation < (-90):
                    temp_angle_deviation = temp_angle_deviation + 180

                angle_deviation[i][j] = np.abs(temp_angle_deviation)

        print("Angle deviation:\n{}".format(angle_deviation))

        plt.figure(figsize=(10, 8))
        img_angle_deviation = plt.imshow(angle_deviation)
        plt.title("Deviation of angle")
        plt.colorbar()
        if np.amax(angle_deviation) > 45:
            img_angle_deviation.set_clim(0, 90)
        elif np.amax(angle_deviation) > 20:
            img_angle_deviation.set_clim(0, 45)
        elif np.amax(angle_deviation) > 10:
            img_angle_deviation.set_clim(0, 20)
        else:
            img_angle_deviation.set_clim(0, 10)

        plt.show()


# show partial orientation
def show_partial_orientation():
    partial_direciton_1 = partial_direciton.copy()
    bad_points = []
    for i in range(np.shape(partial_direciton)[0]):
        for j in range(np.shape(partial_direciton)[1]):
            partial_direciton_1[i][j] = partial_direciton[i][j] - 90
            if partial_direciton_1[i][j] > 90:
                partial_direciton_1[i][j] = partial_direciton_1[i][j] - 180
            if partial_direciton_1[i][j] < -90:
                partial_direciton_1[i][j] = partial_direciton_1[i][j] + 180
            if partial_direciton_1[i][j] == (-90):
                partial_direciton_1[i][j] = -100

    mycolors = [(0, '#012ce7'), (0.25, '#48db25'), (0.5, '#fff21e'), (0.75, '#de22ba'), (1, '#e70111')]
    mycmap = LinearSegmentedColormap.from_list('custom_cmap', mycolors2)

    plt.figure(figsize=(14, 10))
    plt.imshow(np.abs(partial_direciton_1))
    plt.colorbar()

    plt.show()


# show partial direction(FFT+real)
def show_partial_direciton():
    if (partial_image_output.get()) == 1:
        show_FFT_direction()
    show_partial_real_direction()
    show_angle_deviation()


buttun_show_partial_direction = tk.Button(window, text='Show Partial Direction', font=('Arial', 12), height=1, width=25,
                                          command=show_partial_direciton)
buttun_show_partial_direction.place(x=640, y=380, anchor='w')


# save partial
def partial_save_it():
    window_partial_save = tk.Toplevel(window)
    window_partial_save.geometry('350x300')
    window_partial_save.title('Save It (Partial)!')

    canvas = tk.Canvas(window_partial_save, width=1850, height=700)
    canvas.pack(fill='both', expand=True)
    rect_partial_save = canvas.create_rectangle(20, 60, 320, 200)

    def select_partial_save_route():
        global partial_save_route
        partial_save_route = tk.filedialog.askdirectory()
        print("Partial Save Route : {}".format(partial_save_route))

    buttun_select_partial_save_route = tk.Button(window_partial_save, text='Select Partial Save Route',
                                                 font=('Arial', 12), height=1, width=30,
                                                 command=select_partial_save_route)
    buttun_select_partial_save_route.place(x=30, y=30, anchor='w')

    def save_partial_image_and_FFT():
        if partial_save_route == "":
            print("please choose save route!")
            return 0

        os.makedirs('{}/partial_image'.format(partial_save_route))

        # save partial image

        plt.figure(figsize=(10, 8))
        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_image[i * y_part_number + j]), cmap='gray')

        plt.suptitle('Partial Image ( {} * {} )'.format(x_part_number, y_part_number))

        plt.savefig(
            "{}/partial_image/{}_partial_image.png".format(partial_save_route, os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        # save partial FFT

        plt.figure(figsize=(10, 8))
        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_FFT[i * y_part_number + j]), cmap='gray')

        plt.suptitle('Partial FFT ( {} * {} )'.format(x_part_number, y_part_number))

        plt.savefig(
            "{}/partial_image/{}_partial_masked_FFT.png".format(partial_save_route, os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        print("Partial image and FFT saved successfully!")

    buttun_save_partial_image_and_FFT = tk.Button(window_partial_save, text='Save Partial Image and FFT',
                                                  font=('Arial', 12), height=1, width=30,
                                                  command=save_partial_image_and_FFT)
    buttun_save_partial_image_and_FFT.place(x=30, y=90, anchor='w')

    def save_partial_direction():
        if partial_save_route == "":
            print("please choose save route!")
            return 0

        os.makedirs('{}/partial_direction'.format(partial_save_route))

        part_sum = int(180 / float(partial_angle_step.get()))
        x1 = np.array(range(part_sum * 2)) / part_sum * 180

        # save partial curve
        plt.figure(figsize=(10, 8))
        for i in range(x_part_number):
            for j in range(y_part_number):
                if partial_peak_number_sum[i, j] == 1:
                    # show distribution
                    plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                    ax = plt.gca()
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0])],
                                partial_curve[i * y_part_number + j][int(partial_peak_sum[i * y_part_number + j][0])],
                                c="g", marker="x")
                    plt.scatter(partial_peak_sum[i * y_part_number + j][1]['left_ips'] / part_sum * 180,
                                partial_peak_sum[i * y_part_number + j][1]['width_heights'], c="g", marker="v")
                    plt.scatter(partial_peak_sum[i * y_part_number + j][1]['right_ips'] / part_sum * 180,
                                partial_peak_sum[i * y_part_number + j][1]['width_heights'], c="g", marker="v")

                    if partial_direciton[i, j] > 180:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j] - 180), fontsize=10)
                    else:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j]), fontsize=10)

                elif partial_peak_number_sum[i, j] == 2:  # if fits two peaks, choose the higher one
                    # show distribution
                    if float(partial_peak_sum[i * y_part_number + j][1]['prominences'][0]) > float(
                            partial_peak_sum[i * y_part_number + j][1]['prominences'][1]):
                        partial_peak_choose = 0
                    else:
                        partial_peak_choose = 1
                    plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                    ax = plt.gca()
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0][partial_peak_choose])],
                                partial_curve[i * y_part_number + j][
                                    int(partial_peak_sum[i * y_part_number + j][0][partial_peak_choose])], c="g",
                                marker="x")
                    plt.scatter(
                        partial_peak_sum[i * y_part_number + j][1]['left_ips'][partial_peak_choose] / part_sum * 180,
                        partial_peak_sum[i * y_part_number + j][1]['width_heights'][partial_peak_choose], c="g",
                        marker="v")
                    plt.scatter(
                        partial_peak_sum[i * y_part_number + j][1]['right_ips'][partial_peak_choose] / part_sum * 180,
                        partial_peak_sum[i * y_part_number + j][1]['width_heights'][partial_peak_choose], c="g",
                        marker="v")

                    if partial_direciton[i, j] > 180:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j] - 180), fontsize=10)
                    else:
                        plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                            partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                                 "{:.2f}".format(partial_direciton[i, j]), fontsize=10)

                    # COP=cal_COP(array=ave_intensity,peak_max=float(peaks[0][peak_choose]),step=float(angle_step.get()))
                else:
                    # show distribution
                    plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                    ax = plt.gca()
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    plt.plot(x1, partial_curve[i * y_part_number + j])

                    for ix in range(int(partial_peak_number_sum[i, j])):
                        plt.scatter(x1[int(partial_peak_sum[i * y_part_number + j][0][ix])],
                                    partial_curve[i * y_part_number + j][
                                        int(partial_peak_sum[i * y_part_number + j][0][ix])], c="g", marker="x")
                        plt.scatter(partial_peak_sum[i * y_part_number + j][1]['left_ips'][ix] / part_sum * 180,
                                    partial_peak_sum[i * y_part_number + j][1]['width_heights'][ix], c="g", marker="v")
                        plt.scatter(partial_peak_sum[i * y_part_number + j][1]['right_ips'][ix] / part_sum * 180,
                                    partial_peak_sum[i * y_part_number + j][1]['width_heights'][ix], c="g", marker="v")

                    plt.text(0, 0.75 * (np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                        partial_curve[i * y_part_number + j])) + np.amin(partial_curve[i * y_part_number + j]),
                             "#:{}".format(partial_peak_number_sum[i, j]), fontsize=10, c="r")

        plt.suptitle('Partial Curve ( {} * {} )'.format(x_part_number, y_part_number))
        plt.savefig(
            "{}/partial_direction/{}_partial_curve.png".format(partial_save_route, os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        # save partial image direction
        plt.figure(figsize=(10, 8))
        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_image[i * y_part_number + j]), cmap='gray')

                # draw arrow only when the direction is recognized
                if (partial_peak_number_sum[i, j] != 1) & (partial_peak_number_sum[i, j] != 2):
                    continue

                r_circle = min(np.shape(partial_image[i * y_part_number + j])) * 0.8 / 2
                arrow_begin = (np.shape(partial_image[i * y_part_number + j])[1] / 2 + np.sin(
                    partial_direciton[i, j] / 180 * np.pi) * r_circle,
                               np.shape(partial_image[i * y_part_number + j])[0] / 2 - np.cos(
                                   partial_direciton[i, j] / 180 * np.pi) * r_circle)
                arrow_end = (np.shape(partial_image[i * y_part_number + j])[1] / 2 - np.sin(
                    partial_direciton[i, j] / 180 * np.pi) * r_circle,
                             np.shape(partial_image[i * y_part_number + j])[0] / 2 + np.cos(
                                 partial_direciton[i, j] / 180 * np.pi) * r_circle)

                plt.annotate(text='', xy=arrow_begin, xytext=arrow_end,
                             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        plt.suptitle(
            'Partial Direction ( {} * {} )\n\nMean of {} direction: {:.2f}\nStandard deviation of direction: {:.2f}'.format(
                x_part_number, y_part_number, effective_direction_number, partial_mean, partial_std))
        plt.savefig("{}/partial_direction/{}_partial_image_direction.png".format(partial_save_route,
                                                                                 os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        # save partial FFT direction
        plt.figure(figsize=(10, 8))
        for i in range(x_part_number):
            for j in range(y_part_number):
                plt.subplot(x_part_number, y_part_number, i * y_part_number + j + 1)
                plt.axis('off')
                plt.imshow(np.abs(partial_FFT[i * y_part_number + j]), cmap='gray')

                # draw arrow only when the direction is recognized
                if (partial_peak_number_sum[i, j] != 1) & (partial_peak_number_sum[i, j] != 2):
                    continue

                r_circle = min(np.shape(partial_FFT[i * y_part_number + j])) * 0.8 / 2
                arrow_begin = (np.shape(partial_FFT[i * y_part_number + j])[1] / 2 + np.cos(
                    partial_direciton[i, j] / 180 * np.pi) * r_circle,
                               np.shape(partial_FFT[i * y_part_number + j])[0] / 2 + np.sin(
                                   partial_direciton[i, j] / 180 * np.pi) * r_circle)
                arrow_end = (np.shape(partial_FFT[i * y_part_number + j])[1] / 2 - np.cos(
                    partial_direciton[i, j] / 180 * np.pi) * r_circle,
                             np.shape(partial_FFT[i * y_part_number + j])[0] / 2 - np.sin(
                                 partial_direciton[i, j] / 180 * np.pi) * r_circle)

                plt.annotate(text='', xy=arrow_begin, xytext=arrow_end,
                             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        plt.suptitle('Partial FFT Direction ( {} * {} )'.format(x_part_number, y_part_number))
        plt.savefig("{}/partial_direction/{}_partial_FFT_direction.png".format(partial_save_route,
                                                                               os.path.basename(filename)[0:-4]))
        plt.clf()
        plt.close()

        print("Partial direction saved successfully!")

    buttun_save_partial_direction = tk.Button(window_partial_save, text='Save Partial Direction', font=('Arial', 12),
                                              height=1, width=30, command=save_partial_direction)
    buttun_save_partial_direction.place(x=30, y=130, anchor='w')

    def save_partial_data():
        if partial_save_route == "":
            print("please choose save route!")
            return 0

        part_sum = int(180 / float(partial_angle_step.get()))
        x1 = np.array(range(part_sum * 2)) / part_sum * 180

        fo = file_angle_distribution = open(
            "{}/{}_partial_angle_distribution_data.txt".format(partial_save_route, os.path.basename(filename)[0:-4]),
            "w+")
        file_angle_distribution.write("Source image name : {}\n\n".format(filename))

        file_angle_distribution.write("Processing information:\n")
        file_angle_distribution.write(
            "Image size(%): x:{}*{} ; y:{}*{}\n".format(real_size_x_min, real_size_x_max, real_size_y_min,
                                                        real_size_y_max))
        file_angle_distribution.write("Mask inside radius (%): {}\n".format(mask_inside_radius.get()))
        file_angle_distribution.write("Mask outside radius (%): {}\n\n".format(mask_outside_radius.get()))

        file_angle_distribution.write("Cut part: {} * {}\n".format(x_part_number, y_part_number))
        file_angle_distribution.write("Angle step: {}\n".format(partial_angle_step.get()))
        file_angle_distribution.write("Min prominence: {}\n".format(partial_min_prominence.get()))
        file_angle_distribution.write("Min width: {}\n\n".format(partial_min_width.get()))

        file_angle_distribution.write("Fit result:\n")

        file_angle_distribution.write("Effective direction: {}\n".format(effective_direction_number))
        file_angle_distribution.write("Direction mean: {}\n".format(partial_mean))
        file_angle_distribution.write("Direction std: {}\n\n".format(partial_std))
        file_angle_distribution.write("Peak number:\n{}\n\n".format(partial_peak_number_sum))
        file_angle_distribution.write("Peak direction:\n{}\n\n".format(partial_direciton))

        file_angle_distribution.write("Peak fitting data:\n\n")
        for i in range(x_part_number):
            for j in range(y_part_number):
                file_angle_distribution.write(
                    "[{} * {}]:\n{}\n\n".format(i + 1, j + 1, partial_peak_sum[i * y_part_number + j]))
                file_angle_distribution.write("Angle Intensity\n")
                for k in range(np.shape(x1)[0]):
                    file_angle_distribution.write("{:.0f} {}\n".format(x1[k], partial_curve[i * y_part_number + j][k]))
                file_angle_distribution.write("\n")

        fo.close()
        print("Partial data file saved successfully!")

    buttun_save_partial_data = tk.Button(window_partial_save, text='Save Partial Data', font=('Arial', 12), height=1,
                                         width=30, command=save_partial_data)
    buttun_save_partial_data.place(x=30, y=170, anchor='w')

    def partial_save_all():
        if partial_save_route == "":
            print("please choose save route!")
            return 0
        save_partial_image_and_FFT()
        save_partial_direction()
        save_partial_data()
        print("---------saved all-----------")

    buttun_partial_save_all = tk.Button(window_partial_save, text='Save All', font=('Arial', 12), height=1, width=30,
                                        command=partial_save_all)
    buttun_partial_save_all.place(x=30, y=230, anchor='w')

    # exit save window
    buttun_partial_exit_save = tk.Button(window_partial_save, text='Exit', font=('Arial', 12), height=1, width=30,
                                         command=window_partial_save.destroy)
    buttun_partial_exit_save.place(x=30, y=270, anchor='w')


buttun_save_partial = tk.Button(window, text='Save Partialy', font=('Arial', 12), height=1, width=25,
                                command=partial_save_it)
buttun_save_partial.place(x=640, y=440, anchor='w')


# show max of FFT
def index_of_FFT():
    # global amax_FFT
    # amax_FFT=np.zeros([x_part_number,y_part_number])
    # for i in range(x_part_number):
    #    for j in range(y_part_number):
    #        temp=np.amax(partial_FFT[i*y_part_number+j])
    #        amax_FFT[i][j]=temp
    #
    ##print("amax of FFT:\n{}".format(amax_FFT))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(amax_FFT)
    # plt.colorbar()
    # plt.title("amax of FFT")
    # plt.show()
    #
    # std_FFT=np.zeros([x_part_number,y_part_number])
    # sum_FFT=np.zeros([x_part_number,y_part_number])
    # ptp_FFT=np.zeros([x_part_number,y_part_number])
    #
    # for i in range(x_part_number):
    #    for j in range(y_part_number):
    #        temp=partial_FFT[i*y_part_number+j][partial_FFT[i*y_part_number+j]!=0]
    #        temp_std=np.std(temp)
    #        temp_sum=np.sum(temp)
    #        temp_ptp=np.ptp(temp)
    #        std_FFT[i][j]=temp_std
    #        sum_FFT[i][j]=temp_sum
    #        ptp_FFT[i][j]=temp_ptp
    #
    ##print("std of FFT:\n{}".format(std_FFT))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(std_FFT)
    # plt.colorbar()
    # plt.title("std of FFT")
    # plt.show()
    #
    ##print("sum of FFT:\n{}".format(sum_FFT))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(sum_FFT)
    # plt.colorbar()
    # plt.title("sum of FFT")
    # plt.show()
    #
    ##print("ptp of FFT:\n{}".format(ptp_FFT))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(ptp_FFT)
    # plt.colorbar()
    # plt.title("ptp of FFT")
    # plt.show()
    #
    #
    # std_ODF=np.zeros([x_part_number,y_part_number])
    # normalized_std_ODF=np.zeros([x_part_number,y_part_number])
    # for i in range(x_part_number):
    #    for j in range(y_part_number):
    #        temp_std_ODF=np.std(partial_curve[i*y_part_number+j])
    #        temp_normalized_std_ODF=np.std((partial_curve[i*y_part_number+j]-np.amin(partial_curve[i*y_part_number+j]))/(np.amax(partial_curve[i*y_part_number+j])-np.amin(partial_curve[i*y_part_number+j])))
    #        std_ODF[i][j]=temp_std_ODF
    #        normalized_std_ODF[i][j]=temp_normalized_std_ODF
    #
    #
    ##print("std of ODF:\n{}".format(std_ODF))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(std_ODF)
    # plt.colorbar()
    # plt.title("std of ODF")
    # plt.show()
    #
    ##print("normalized std of ODF:\n{}".format(normalized_std_ODF))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(normalized_std_ODF)
    # plt.colorbar()
    # plt.title("normalized std of ODF")
    # plt.show()
    #
    partial_COP = np.zeros([x_part_number, y_part_number])
    normalized_partial_COP = np.zeros([x_part_number, y_part_number])
    for i in range(x_part_number):
        for j in range(y_part_number):
            temp_partial_COP = cal_COP(partial_curve[i * y_part_number + j], peak_max=(
                        np.argmax(partial_curve[i * y_part_number + j]) * int(partial_angle_step.get())),
                                       step=int(partial_angle_step.get()))
            temp_normalized_partial_COP = cal_COP(
                (partial_curve[i * y_part_number + j] - np.amin(partial_curve[i * y_part_number + j])) / (
                            np.amax(partial_curve[i * y_part_number + j]) - np.amin(
                        partial_curve[i * y_part_number + j])),
                peak_max=(np.argmax(partial_curve[i * y_part_number + j]) * int(partial_angle_step.get())),
                step=int(partial_angle_step.get()))
            partial_COP[i][j] = temp_partial_COP
            normalized_partial_COP[i][j] = temp_normalized_partial_COP

    # print("partial COP:\n{}".format(partial_COP))

    plt.figure(figsize=(10, 8))
    plt.imshow(partial_COP)
    plt.colorbar()
    plt.title("partial COP")
    plt.show()

    # print("normalized partial COP:\n{}".format(normalized_partial_COP))

    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_partial_COP)
    plt.colorbar()
    plt.title("normalized partial COP")
    plt.show()

    partial_ODF_parameter = np.zeros([x_part_number, y_part_number])
    for i in range(x_part_number):
        for j in range(y_part_number):
            temp_partial_ODF_parameter = cal_ODF_parameter(partial_curve[i * y_part_number + j], peak_max=(
                        np.argmax(partial_curve[i * y_part_number + j]) * int(partial_angle_step.get())),
                                                           step=int(partial_angle_step.get()))
            partial_ODF_parameter[i][j] = temp_partial_ODF_parameter

    # print("partial ODF parameter:\n{}".format(partial_ODF_parameter))

    plt.figure(figsize=(10, 8))
    plt.imshow(partial_ODF_parameter)
    plt.colorbar()
    plt.title("partial ODF parameter")
    plt.show()

    # partial_ODF_parameter_2=np.zeros([x_part_number,y_part_number])
    # for i in range(x_part_number):
    #    for j in range(y_part_number):
    #        temp_partial_ODF_parameter_2=cal_ODF_parameter(partial_curve[i*y_part_number+j],peak_max=(np.argmax(partial_curve[i*y_part_number+j])*int(partial_angle_step.get())),step=int(partial_angle_step.get()),type_para=2)
    #        partial_ODF_parameter_2[i][j]=temp_partial_ODF_parameter_2
    #
    ##print("partial ODF parameter_2:\n{}".format(partial_ODF_parameter_2))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(partial_ODF_parameter_2)
    # plt.colorbar()
    # plt.title("partial ODF parameter_2")
    # plt.show()

    # nn_cc=fft_cut_range_2.get()
    # partial_ODF_parameter_cc=np.zeros([x_part_number,y_part_number])
    # for i in range(x_part_number):
    #    for j in range(y_part_number):
    #        temp_partial_ODF_parameter_cc=cal_ODF_parameter(partial_curve[i*y_part_number+j],peak_max=(np.argmax(partial_curve[i*y_part_number+j])*int(partial_angle_step.get())),step=int(partial_angle_step.get()),type_para=nn_cc)
    #        partial_ODF_parameter_cc[i][j]=temp_partial_ODF_parameter_cc
    #
    ##print("partial ODF parameter_cc:\n{}".format(partial_ODF_parameter_cc))
    #
    # plt.figure(figsize=(10,8))
    # plt.imshow(partial_ODF_parameter_cc)
    # plt.colorbar()
    # plt.title("partial ODF parameter_cc(nn={})".format(nn_cc))
    # plt.show()


buttun_save_partial = tk.Button(window, text='Index of FFT', font=('Arial', 12), height=1, width=25,
                                command=index_of_FFT)
buttun_save_partial.place(x=640, y=480, anchor='w')

# batch recognize
lab_batch_recognize = tk.Label(window, text='Batch Recognize:', width=22, font=('Arial', 12))
lab_batch_recognize.place(x=40, y=390, anchor='w')


def batch_source_folder():
    global source_folder
    global num_image_file
    global image_load
    global image_list
    source_folder = tk.filedialog.askdirectory()

    if source_folder != "":
        print("Source folder of batch recognize : {}".format(source_folder))

        image_list = np.array(os.listdir(source_folder))

        # count the number of image
        num_image_file = 0

        for i, filename in enumerate(image_list):
            # count the number of image file
            if filename[-4:] == '.tif':
                num_image_file = num_image_file + 1

        print("This folder contains {} image file.".format(num_image_file))

        # show the first image
        for i, filename in enumerate(image_list):
            # only read image file
            if filename[-4:] != '.tif':
                continue

            # load image
            image_load = (plt.imread(source_folder + '/' + filename))
            if np.ndim(image_load) == 3:
                image_load = image_load[:, :, 0]

            plt.figure(figsize=(10, 8))
            plt.title("First image in folder")
            plt.imshow(np.abs(image_load), cmap='gray')
            plt.show()
            break


buttun_select_source_folder = tk.Button(window, text='Select Source Folder', font=('Arial', 12), height=1, width=21,
                                        command=batch_source_folder)
buttun_select_source_folder.place(x=40, y=420, anchor='w')


def test_first_image():
    global image_backup
    print("\n-----------------↓First Image Test↓-----------------\n")
    image_backup = image_load.copy()
    resize_img()
    draw_all()

    show_partial_image()
    show_partial_FFT()
    fit_partial_direction()
    show_FFT_direction()
    show_partial_real_direction()
    print("\n-----------------↑First Image Test↑-----------------\n")


buttun_test_first_image = tk.Button(window, text='Test First Image', font=('Arial', 12), height=1, width=21,
                                    command=test_first_image)
buttun_test_first_image.place(x=40, y=460, anchor='w')


def draw_all_without_show_img(img):
    ###resize img
    raw_x = np.shape(img)[0]
    raw_y = np.shape(img)[1]

    image_size_type = var_image_size.get()

    if image_size_type == 0:
        reshaped_img = img[int(0.01 * raw_x * float(y_size_min.get())):int(0.01 * raw_x * float(y_size_max.get())),
                       int(0.01 * raw_y * float(x_size_min.get())):int(0.01 * raw_y * float(x_size_max.get()))]
    elif image_size_type == 1:
        reshaped_img = img[int(y_size_min.get()):int(y_size_max.get()), int(x_size_min.get()):int(x_size_max.get())]

    ###masked FFT
    fftImage = np.fft.fft2(reshaped_img)
    fftShiftImage = np.fft.fftshift(fftImage)
    fftMagImage = np.abs(fftShiftImage)

    inside_r_precentage = float(mask_inside_radius.get())
    outside_r_precentage = float(mask_outside_radius.get())

    inside_r = np.amin(np.shape(fftMagImage)) * inside_r_precentage / 100 / 2
    outside_r = np.amin(np.shape(fftMagImage)) * outside_r_precentage / 100 / 2

    # mask of center masked FFT
    center_mask = np.ones(np.shape(fftImage))

    # specify circle parameters: centre ij
    ci, cj = np.shape(center_mask)[0] / 2, np.shape(center_mask)[1] / 2

    # Create index arrays to z
    I, J = np.meshgrid(np.arange(center_mask.shape[1]), np.arange(center_mask.shape[0]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

    # Assign value of 0 to those points where dist<cr:
    center_mask[np.where(dist < inside_r)] = 0

    # mask of masked FFT
    whole_mask = center_mask.copy()
    whole_mask[np.where(dist > outside_r)] = 0

    # masked FFT and center masked FFT
    center_masked_FFT = np.multiply(center_mask, fftMagImage)
    masked_FFT = np.multiply(whole_mask, fftMagImage)

    ###calculate angle distribution
    part_sum = int(180 / float(angle_step.get()))

    center = [float((np.shape(fftMagImage)[0] - 1) / 2), float((np.shape(fftMagImage)[1] - 1) / 2)]

    sum_intensity = np.zeros(part_sum * 2)
    sum_number = np.zeros(part_sum * 2)
    ave_intensity = np.zeros(part_sum * 2)

    for i in range(np.shape(fftMagImage)[0]):
        for j in range(np.shape(fftMagImage)[1]):
            y_0 = i - center[0]
            x_0 = j - center[1]
            if whole_mask[i][j] == 1:
                if j - center[1] == 0:
                    part_number = int(part_sum / 2)
                    sum_intensity[part_number] = sum_intensity[part_number] + fftMagImage[i][j]
                    sum_number[part_number] = sum_number[part_number] + 1
                else:
                    angle = np.arctan(y_0 / x_0) / np.pi * 180

                    part_number = round(angle / float(angle_step.get()))
                    if part_number < 0:
                        part_number = part_number + part_sum
                    if part_number >= part_sum:
                        part_number = part_number - part_sum
                    sum_intensity[part_number] = sum_intensity[part_number] + fftMagImage[i][j]
                    sum_number[part_number] = sum_number[part_number] + 1

    for i in range(part_sum):
        ave_intensity[i] = sum_intensity[i] / sum_number[i]

    for i in range(part_sum):
        ave_intensity[i + part_sum] = ave_intensity[i]

    ###fit curve
    prominence = float(min_prominence.get())
    width = float(min_width.get()) / float(angle_step.get())
    part_sum = int(180 / float(angle_step.get()))

    prominence_real = (np.amax(ave_intensity) - np.amin(ave_intensity)) * prominence

    peaks = find_peaks(ave_intensity, prominence=prominence_real, width=width)

    if np.shape(peaks[0])[0] == 1:
        peak_max = float(peaks[0]) * float(angle_step.get())
        peak_center = (peaks[1]['left_ips'][0] / part_sum * 180 + peaks[1]['right_ips'][0] / part_sum * 180) / 2
        peak_error = float(peaks[1]['widths'] / 2) * float(angle_step.get())

        COP = cal_COP(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()))
        OP1 = cal_ODF_parameter(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()), type_para=1)
        normalized_OP1 = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_center,
                                           step=float(angle_step.get()), type_para=1)

    elif np.shape(peaks[0])[0] == 2:  # if fits two peaks, choose the higher one
        if float(peaks[1]['prominences'][0]) > float(peaks[1]['prominences'][1]):
            peak_choose = 0
        else:
            peak_choose = 1
        peak_max = float(peaks[0][peak_choose]) * float(angle_step.get())
        peak_center = (peaks[1]['left_ips'][peak_choose] / part_sum * 180 + peaks[1]['right_ips'][
            peak_choose] / part_sum * 180) / 2
        peak_error = float(peaks[1]['widths'][peak_choose] / 2) * float(angle_step.get())

        COP = cal_COP(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()))
        OP1 = cal_ODF_parameter(array=ave_intensity, peak_max=peak_center, step=float(angle_step.get()), type_para=1)
        normalized_OP1 = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_center,
                                           step=float(angle_step.get()), type_para=1)

    else:
        peak_center = None

        peak_max = 0
        for i in range(np.shape(ave_intensity)[0]):
            if ave_intensity[i] > ave_intensity[peak_max]:
                peak_max = i

        peak_error = None
        COP = cal_COP(array=ave_intensity, peak_max=peak_max,
                      step=float(angle_step.get()))  # 对于找不出峰的图像，直接采用全图最高点作为峰值计算COP
        OP1 = cal_ODF_parameter(array=ave_intensity, peak_max=peak_max, step=float(angle_step.get()), type_para=1)
        normalized_OP1 = cal_ODF_parameter(array=(ave_intensity - np.amin(ave_intensity)), peak_max=peak_max,
                                           step=float(angle_step.get()), type_para=1)

    return peaks, peak_center, peak_error, COP, OP1, normalized_OP1


def draw_all_partially_without_show_img(img):
    ###resize img
    raw_x = np.shape(img)[0]
    raw_y = np.shape(img)[1]

    image_size_type = var_image_size.get()

    if image_size_type == 0:
        reshaped_img = img[int(0.01 * raw_x * float(y_size_min.get())):int(0.01 * raw_x * float(y_size_max.get())),
                       int(0.01 * raw_y * float(x_size_min.get())):int(0.01 * raw_y * float(x_size_max.get()))]
    elif image_size_type == 1:
        reshaped_img = img[int(y_size_min.get()):int(y_size_max.get()), int(x_size_min.get()):int(x_size_max.get())]

        ###partial image
    x_part_number, y_part_number = calculate_most_part_1()

    partial_image = []

    for i in range(x_part_number):
        for j in range(y_part_number):
            partial_image.append(reshaped_img[int(np.shape(reshaped_img)[0] * i / x_part_number):int(
                np.shape(reshaped_img)[0] * (i + 1) / x_part_number),
                                 int(np.shape(reshaped_img)[1] * j / y_part_number):int(
                                     np.shape(reshaped_img)[1] * (j + 1) / y_part_number)])

    ###partial FFT
    partial_FFT = []
    partial_mask = []

    # round_shrink=np.shape(img)[0]/np.amin([np.shape(img)[0]/x_part_number,np.shape(img)[1]/y_part_number])
    inside_r_precentage = float(mask_inside_radius.get())
    outside_r_precentage = float(mask_outside_radius.get())

    for i in range(x_part_number):
        for j in range(y_part_number):
            partial_fftImage = np.fft.fft2(partial_image[i * y_part_number + j])
            partial_fftShiftImage = np.fft.fftshift(partial_fftImage)
            partial_fftMagImage = np.abs(partial_fftShiftImage)

            inside_r = np.amin(np.shape(partial_fftMagImage)) * inside_r_precentage / 100 / 2
            outside_r = np.amin(np.shape(partial_fftMagImage)) * outside_r_precentage / 100 / 2

            # mask of center masked FFT
            partial_center_mask = np.ones(np.shape(partial_fftMagImage))

            # specify circle parameters: centre ij
            ci, cj = np.shape(partial_center_mask)[0] / 2, np.shape(partial_center_mask)[1] / 2

            # Create index arrays to z
            I, J = np.meshgrid(np.arange(partial_center_mask.shape[1]), np.arange(partial_center_mask.shape[0]))

            # calculate distance of all points to centre
            dist = np.sqrt((I - cj) ** 2 + (J - ci) ** 2)

            # Assign value of 0 to those points where dist<cr:
            partial_center_mask[np.where(dist < inside_r)] = 0

            # mask of masked FFT
            partial_whole_mask = partial_center_mask.copy()
            partial_whole_mask[np.where(dist > outside_r)] = 0

            masked_partial_FFT = np.multiply(partial_whole_mask, partial_fftMagImage)

            partial_mask.append(partial_whole_mask)
            partial_FFT.append(masked_partial_FFT)

    ###fit partial direction

    partial_peak_number_sum = np.zeros([x_part_number, y_part_number])
    partial_direciton = np.zeros([x_part_number, y_part_number])
    partial_error = np.zeros([x_part_number, y_part_number])
    partial_intensity_sum = np.zeros([x_part_number, y_part_number])

    part_sum = int(180 / float(partial_angle_step.get()))

    partial_peak_sum = []
    partial_curve = []

    x1 = np.array(range(part_sum * 2)) / part_sum * 180

    for i in range(x_part_number):
        for j in range(y_part_number):
            center = [float((np.shape(partial_FFT[i * y_part_number + j])[0] - 1) / 2),
                      float((np.shape(partial_FFT[i * y_part_number + j])[1] - 1) / 2)]

            partial_sum_intensity = np.zeros(part_sum * 2)
            partial_sum_number = np.zeros(part_sum * 2)
            partial_ave_intensity = np.zeros(part_sum * 2)

            # main part to sum intensity
            for ii in range(np.shape(partial_FFT[i * y_part_number + j])[0]):
                for jj in range(np.shape(partial_FFT[i * y_part_number + j])[1]):

                    y_0 = ii - center[0]
                    x_0 = jj - center[1]

                    if partial_mask[i * y_part_number + j][ii][jj] == 1:
                        if jj - center[1] == 0:
                            part_number = int(part_sum / 2)
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]
                        else:
                            angle = np.arctan(y_0 / x_0) / np.pi * 180

                            part_number = round(angle / float(partial_angle_step.get()))
                            if part_number < 0:
                                part_number = part_number + part_sum
                            if part_number >= part_sum:
                                part_number = part_number - part_sum
                            partial_sum_intensity[part_number] = partial_sum_intensity[part_number] + \
                                                                 partial_FFT[i * y_part_number + j][ii][jj]
                            partial_sum_number[part_number] = partial_sum_number[part_number] + 1
                            partial_intensity_sum[i, j] = partial_intensity_sum[i, j] + \
                                                          partial_FFT[i * y_part_number + j][ii][jj]

            for iii in range(part_sum):
                partial_ave_intensity[iii] = partial_sum_intensity[iii] / partial_sum_number[iii]

            for iii in range(part_sum):
                partial_ave_intensity[iii + part_sum] = partial_ave_intensity[iii]

            partial_curve.append(partial_ave_intensity)

            # calculate partial direciton
            partial_prominence = float(partial_min_prominence.get())
            partial_width = float(partial_min_width.get()) / float(partial_angle_step.get())
            part_sum = int(180 / float(partial_angle_step.get()))

            COP = 0  # undefine

            partial_prominence_real = (np.amax(partial_ave_intensity) - np.amin(
                partial_ave_intensity)) * partial_prominence

            partial_peaks = find_peaks(partial_ave_intensity, prominence=partial_prominence_real, width=partial_width)

            ###show fit result
            # to calculate STD correctly, range of [partial_direction] is set to -90 to 90
            if np.shape(partial_peaks[0])[0] == 1:
                partial_peak_max = float(partial_peaks[0]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][0] / part_sum * 180 + partial_peaks[1]['right_ips'][
                    0] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'] / 2) * float(partial_angle_step.get())

                # save direction
                while partial_peak_center < -90:
                    partial_peak_center = partial_peak_center + 180
                while partial_peak_center > 90:
                    partial_peak_center = partial_peak_center - 180
                partial_direciton[i, j] = partial_peak_center
                partial_error[i, j] = partial_peak_error

            elif np.shape(partial_peaks[0])[0] == 2:  # if fits two peaks, choose the higher one
                if float(partial_peaks[1]['prominences'][0]) > float(partial_peaks[1]['prominences'][1]):
                    partial_peak_choose = 0
                else:
                    partial_peak_choose = 1
                partial_peak_max = float(partial_peaks[0][partial_peak_choose]) * float(partial_angle_step.get())
                partial_peak_center = (partial_peaks[1]['left_ips'][partial_peak_choose] / part_sum * 180 +
                                       partial_peaks[1]['right_ips'][partial_peak_choose] / part_sum * 180) / 2
                partial_peak_error = float(partial_peaks[1]['widths'][partial_peak_choose] / 2) * float(
                    partial_angle_step.get())

                # save direction
                # HERE can change the range
                while partial_peak_center < 0:
                    partial_peak_center = partial_peak_center + 180
                while partial_peak_center > 180:
                    partial_peak_center = partial_peak_center - 180
                partial_direciton[i, j] = partial_peak_center
                partial_error[i, j] = partial_peak_error
            else:
                partial_direciton[i, j] = None
                partial_error[i, j] = None

            partial_peak_number_sum[i, j] = np.shape(partial_peaks[0])[0]

            effective_direction_number = 0
            effective_direction = []

            for ii in range(x_part_number):
                for jj in range(y_part_number):
                    if (partial_peak_number_sum[ii][jj] == 1) or (partial_peak_number_sum[ii][jj] == 2) or (
                    (partial_peak_number_sum[ii][jj] == (-1))):
                        effective_direction.append(partial_direciton[ii][jj])
                        effective_direction_number = effective_direction_number + 1

            partial_mean = np.mean(effective_direction)
            partial_STD = np.std(effective_direction)

    return partial_peak_number_sum, partial_direciton, partial_error, partial_mean, partial_STD


def progress_bar(iteration, total, start_time, length=10):
    """
    from chat老师
    显示进度条的函数。

    :param iteration: 当前迭代次数
    :param total: 总迭代次数
    :param length: 进度条的长度
    """
    current_time = time.time()
    elapsed_time = current_time - start_time
    if iteration == 0:
        remaining_time = 0
    else:
        remaining_time = elapsed_time / iteration * (total - iteration)

    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # sys.stdout.write('\r%s |%s| %s%% 完成' % ('进度', bar, percent))
    sys.stdout.write(
        '\rProcess rate : |{}| {}/{} ({}%), elapsed time : {:.1f}s, estimated remaining time : {:.1f}s'.format(bar,
                                                                                                               iteration,
                                                                                                               total,
                                                                                                               percent,
                                                                                                               elapsed_time,
                                                                                                               remaining_time))
    sys.stdout.flush()


def recognize_all():
    global processing_image_file

    global batch_peaks
    global batch_direction
    global batch_error
    global batch_COP
    global batch_OP1
    global batch_partial_peak_number
    global batch_partial_direction
    global batch_partial_error
    global batch_partial_ave
    global batch_partial_STD
    global batch_normalized_OP1

    print("\n------------------Batch Recognize Start------------------")

    batch_peaks = []
    batch_direction = []
    batch_error = []
    batch_COP = []
    batch_OP1 = []
    batch_partial_peak_number = []
    batch_partial_direction = []
    batch_partial_error = []
    batch_partial_ave = []
    batch_partial_STD = []
    batch_normalized_OP1 = []

    # close all img
    close_all_plt()

    processing_image_file = 0
    start_time = time.time()
    for i, filename in enumerate(image_list):
        # only read image file
        if filename[-4:] != '.tif':
            continue

        # count processed image file
        processing_image_file = processing_image_file + 1

        # load ALL image
        image_load = (plt.imread(source_folder + '/' + filename))
        if np.ndim(image_load) == 3:
            image_load = image_load[:, :, 0]

        temp_peaks, temp_direction, temp_error, temp_COP, temp_OP1, temp_normalized_OP1 = draw_all_without_show_img(
            img=image_load)
        batch_peaks.append(temp_peaks)
        # Here to change the range
        if (temp_direction != None):
            while temp_direction < 0:
                temp_direction = temp_direction + 180
            while temp_direction > 180:
                temp_direction = temp_direction - 180
        batch_direction.append(temp_direction)
        batch_error.append(temp_error)
        batch_COP.append(temp_COP)
        batch_OP1.append(temp_OP1)
        batch_normalized_OP1.append(temp_normalized_OP1)

        temp_partial_peak_number, temp_partial_direction, temp_partial_error, temp_partial_mean, temp_partial_STD = draw_all_partially_without_show_img(
            img=image_load)
        batch_partial_peak_number.append(temp_partial_peak_number)
        batch_partial_direction.append(temp_partial_direction)
        batch_partial_error.append(temp_partial_error)
        batch_partial_ave.append(temp_partial_mean)
        batch_partial_STD.append(temp_partial_STD)

        progress_bar(processing_image_file, start_time=start_time, total=num_image_file)

    print("\n------------------Batch Recognize FINISHED------------------")
    # print("Batch peaks:{}".format(batch_peaks))
    # print("Batch direction:{}".format(batch_direction))
    # print("Batch COP:{}".format(batch_COP))
    # print("Batch OP1:{}".format(batch_OP1))
    # print("Batch partial peak_number:{}".format(batch_partial_peak_number))
    # print("Batch partial direciton:{}".format(batch_partial_direciton))
    # print("Batch partial ave : {}".format(batch_partial_ave))
    # print("Batch partial STD : {}".format(batch_partial_STD))


def save_batch_recognized_data():
    # count image with effective direction
    effective_image = 0
    failed_image = []
    effective_direction = []
    effective_COP = []
    effective_OP1 = []
    for i in range(np.shape(batch_direction)[0]):
        if (batch_direction[i] != None):
            effective_image = effective_image + 1
            effective_direction.append(batch_direction[i])
            effective_COP.append(batch_COP[i])
            effective_OP1.append(batch_OP1[i])
        else:
            failed_image.append(i + 1)

    ave_direction = np.mean(effective_direction)
    std_direction = np.std(effective_direction)
    ave_COP = np.mean(batch_COP)
    std_COP = np.std(batch_COP)
    ave_OP1 = np.mean(batch_OP1)
    std_OP1 = np.std(batch_OP1)
    ave_partial_ave = np.mean(batch_partial_ave)
    std_partial_ave = np.std(batch_partial_ave)
    ave_partial_STD = np.mean(batch_partial_STD)
    std_partial_STD = np.std(batch_partial_STD)

    if effective_image != num_image_file:
        ave_effective_COP = np.mean(effective_COP)
        std_effective_COP = np.std(effective_COP)
        ave_effective_OP1 = np.mean(effective_OP1)
        std_effective_OP1 = np.std(effective_OP1)

    fo = file_batch_recognize = open("{}/batch_recognize_output.txt".format(source_folder), "w+")

    file_batch_recognize.write("------------------folder info------------------\n\n")
    file_batch_recognize.write("Source folder : {}\n".format(source_folder))
    file_batch_recognize.write("Recognize time : {}\n".format(datetime.datetime.now()))
    file_batch_recognize.write("Amount of image : {}\n\n".format(num_image_file))

    file_batch_recognize.write("------------------Statistical data------------------\n\n")
    file_batch_recognize.write("Amount of successfully recognized image : {}\n\n".format(effective_image))
    if effective_image != num_image_file:
        file_batch_recognize.write("Failed image : {}\n\n".format(failed_image))

    file_batch_recognize.write("Direction list: {}\n".format(batch_direction))
    file_batch_recognize.write("Direction error list: {}\n".format(batch_error))
    file_batch_recognize.write("Average direction : {}\n".format(ave_direction))
    file_batch_recognize.write("STD of direction : {}\n\n".format(std_direction))

    file_batch_recognize.write("COP list: {}\n".format(batch_COP))
    file_batch_recognize.write("Average COP : {}\n".format(ave_COP))
    file_batch_recognize.write("STD of COP : {}\n\n".format(std_COP))
    file_batch_recognize.write("OP1 list: {}\n".format(batch_OP1))
    file_batch_recognize.write("Average OP1 : {}\n".format(ave_OP1))
    file_batch_recognize.write("STD of OP1 : {}\n\n".format(std_OP1))

    file_batch_recognize.write("Normalized OP1 list : {}\n\n".format(batch_normalized_OP1))

    if effective_image != num_image_file:
        file_batch_recognize.write("Average COP of effective direction : {}\n".format(ave_effective_COP))
        file_batch_recognize.write("STD of COP of effective direction : {}\n".format(std_effective_COP))
        file_batch_recognize.write("Average OP1 of effective direction : {}\n".format(ave_effective_OP1))
        file_batch_recognize.write("STD of OP1 of effective direction : {}\n\n".format(std_effective_OP1))

    file_batch_recognize.write("Partial average direction list: {}\n".format(batch_partial_ave))
    file_batch_recognize.write("Average of partial average direction : {}\n".format(ave_partial_ave))
    file_batch_recognize.write("STD of partial average direction : {}\n\n".format(std_partial_ave))

    file_batch_recognize.write("Partial STD list: {}\n".format(batch_partial_STD))
    file_batch_recognize.write("Average of partial STD : {}\n".format(ave_partial_STD))
    file_batch_recognize.write("STD of partial STD : {}\n\n".format(std_partial_STD))

    file_batch_recognize.write("------------------Image recognize data------------------\n\n")

    No_image = 0
    for i, filename in enumerate(image_list):
        # only read image file
        if filename[-4:] != '.tif':
            continue

        file_batch_recognize.write("File number : {}/{}\n".format(No_image + 1, num_image_file))
        file_batch_recognize.write("File name : {}\n".format(filename))
        file_batch_recognize.write("Recognized peak : {}\n".format(batch_peaks[No_image]))
        file_batch_recognize.write("Recognized direction : {}\n\n".format(batch_direction[No_image]))
        file_batch_recognize.write("COP : {}\n".format(batch_COP[No_image]))
        file_batch_recognize.write("OP1 : {}\n\n".format(batch_OP1[No_image]))
        file_batch_recognize.write("Partical peak number :\n{}\n\n".format(batch_partial_peak_number[No_image]))
        file_batch_recognize.write("Partial direction :\n{}\n\n".format(batch_partial_direction[No_image]))
        file_batch_recognize.write("Partial direction error :\n{}\n\n".format(batch_partial_error[No_image]))
        file_batch_recognize.write("Partial average direction :{}\n".format(batch_partial_ave[No_image]))
        file_batch_recognize.write("Partial direction STD :{}\n\n".format(batch_partial_STD[No_image]))
        file_batch_recognize.write(
            "----------------------------------------------------------------------------------------\n\n")

        No_image = No_image + 1
    fo.close()

    print("\n\n------------------------File saved------------------------")


def batch_recognize():
    recognize_all()
    save_batch_recognized_data()


buttun_recognize_all = tk.Button(window, text='Recognize ALL', font=('Arial', 12), height=1, width=21,
                                 command=batch_recognize)
buttun_recognize_all.place(x=40, y=500, anchor='w')


# close plt
def close_all_plt():
    plt.close('all')


buttun_close_plt = tk.Button(window, text='Close All Plots', font=('Arial', 12), height=1, width=13,
                             command=close_all_plt)
buttun_close_plt.place(x=300, y=400, anchor='w')


# exit
def exit():
    plt.close('all')
    window.destroy()


buttun_exit = tk.Button(window, text='Exit', font=('Arial', 12), height=1, width=13, command=exit)
buttun_exit.place(x=452, y=400, anchor='w')

# start
window.mainloop()