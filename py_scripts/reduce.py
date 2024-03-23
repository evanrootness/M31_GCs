# Importing necessary packages and modules
import numpy as np
import pandas as pd
import math
import csv
from glob import glob
import matplotlib.pyplot as plt
%matplotlib inline
from astropy.io import ascii, fits
import scipy.optimize as opt
from scipy.interpolate import make_interp_spline
from matplotlib.colors import LogNorm



# Polynomial Models
def const(x, b):
    return b

def linear_model(x, m, b):
    return m * x + b

def poly2_model(x, a, b, c):
    return a * x ** 2 + b * x + c

def poly3_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def poly4_model(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def poly5_model(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

# Gaussian Models
def gaussian_model(x, a, sigma, mean, b):
    return (a * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))) + b

def neg_gauss_model(x, a, sigma, mean, b):
    return -((a * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2)))) + b

def neg_gauss_mx_model(x, a, sigma, mean, b, c):
    return -((a * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2)))) + b * x + c

def neg_gauss_poly2_model(x, a, sigma, mean, b, c, d):
    return (-((a * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))))
            + b * x ** 2 + c * x + d)

# Lorentzian Models
def Lorentzian_const(x, A, sigma, center, c):
    return A * (sigma / ((x - center) ** 2 + sigma ** 2)) + c

def Lorentzian_lin(x, A, sigma, center, c, d):
    return -(A * (sigma / ((x - center) ** 2 + sigma ** 2))) + c * x + d

def Lorentzian_quad(x, A, sigma, center, b, c, d):
    return (-(A * (sigma / ((x - center) ** 2 + sigma ** 2))) 
            + b * x ** 2 + c * x + d)


# Function Definitions

## Cosmic Zapping Functions
def cosmic_zap_median(input_dir, input_folder, 
                      file_number_list, yearmonth=2308):
    fits_data = []
    
    for i in file_number_list:
        fits_data.append(fits.getdata(input_dir + str(yearmonth) 
                                    + str(input_folder) + '/' + i + '.fits', 
                                    ext = 0).astype(np.float64))

    data_cube = np.dstack(fits_data)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)


def zap_median(input_dir, input_folder, file_number_list, 
               yearmonth=2308):
    # reading in files
    fits_data = []
    for i in file_number_list:
        fits_data.append(fits.getdata(input_dir + str(yearmonth) 
                                    + str(input_folder) + '/' 
                                    + i + '.fits', 
                                    ext = 0).astype(np.float64))
        
    data_cube_before_overscan = np.dstack(fits_data)

    # bias correction
    fits_data_2 = []
    for i in range(len(file_number_list)):
        input_array = data_cube_before_overscan[:, :, i]

        overscan_topleft = np.median(input_array[1:516, 1:31])
        overscan_bottomleft = np.median(input_array[517:1013, 1:31])
        overscan_topright = np.median(input_array[1:516, 4097:4128])
        overscan_bottomright = np.median(input_array[517:1013, 4097:4128])

        input_array_dup = np.copy(input_array)
        overscan_corrected = np.copy(input_array)

        overscan_corrected[:508, :2064] = (input_array_dup[:508, :2064] 
                                        - overscan_topleft)
        overscan_corrected[508:, :2064] = (input_array_dup[508:, :2064] 
                                        - overscan_bottomleft)
        overscan_corrected[:508, 2064:] = (input_array_dup[:508, 2064:] 
                                        - overscan_topright)
        overscan_corrected[508:, 2064:] = (input_array_dup[508:, 2064:] 
                                        - overscan_bottomright)
        
        fits_data_2.append(overscan_corrected)
    data_cube = np.dstack(fits_data_2)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)


def zap_two(input_dir, input_folder, file_number_list, 
            max_diff=10, yearmonth=2308):
    # reading in files
    fits_data = []
    for i in file_number_list:
        fits_data.append(fits.getdata(input_dir + str(yearmonth) 
                                    + str(input_folder) + '/' + i + '.fits', 
                                    ext = 0).astype(np.float64))
    data_cube_before_overscan = np.dstack(fits_data)

    # bias correction
    data_2 = []
    for i in range(len(file_number_list)):
        input_array = data_cube_before_overscan[:, :, i]

        overscan_topleft = np.median(input_array[1:516, 1:31])
        overscan_bottomleft = np.median(input_array[517:1013, 1:31])
        overscan_topright = np.median(input_array[1:516, 4097:4128])
        overscan_bottomright = np.median(input_array[517:1013, 4097:4128])

        input_array_dup = np.copy(input_array)
        overscan_corrected = np.copy(input_array)

        overscan_corrected[:508, :2064] = (input_array_dup[:508, :2064] 
                                        - overscan_topleft)
        overscan_corrected[508:, :2064] = (input_array_dup[508:, :2064] 
                                        - overscan_bottomleft)
        overscan_corrected[:508, 2064:] = (input_array_dup[:508, 2064:] 
                                        - overscan_topright)
        overscan_corrected[508:, 2064:] = (input_array_dup[508:, 2064:] 
                                        - overscan_bottomright)
        
        data_2.append(overscan_corrected)
    data_cube = np.dstack(data_2)

    # cosmic ray zapping
    tuple = np.zeros((1016, 4128))
    for i in range(np.shape(data_cube)[0]):
        for j in range(np.shape(data_cube)[1]):
        # Now run through the length of l to see if any have large difference (max_diff)
        for k in range(np.shape(data_cube)[2]):
            if (abs(data_cube[i][j][k] - data_cube[i][j][0]) > max_diff):
            tuple[i][j] = np.min(data_cube[i][j])
            else:
            tuple[i][j] = np.mean(data_cube[i][j])
    return(tuple)


def zap_two_std(input_dir, input_folder, file_number_list, 
                std = 5, yearmonth=2308):

    # reading in files
    fits_data = []
    for i in file_number_list:
        fits_data.append(fits.getdata(input_dir + str(yearmonth) 
                                    + str(input_folder) + '/' + i + '.fits', 
                                    ext = 0).astype(np.float64))
    data_cube_before_overscan = np.dstack(fits_data)

    # bias correction
    fits_data_2 = []
    for i in range(len(file_number_list)):

        input_array = data_cube_before_overscan[:, :, i]

        overscan_topleft = np.median(input_array[1:516, 1:31])
        overscan_bottomleft = np.median(input_array[517:1013, 1:31])
        overscan_topright = np.median(input_array[1:516, 4097:4128])
        overscan_bottomright = np.median(input_array[517:1013, 4097:4128])

        input_array_dup = np.copy(input_array)
        overscan_corrected = np.copy(input_array)

        overscan_corrected[:508, :2064] = (input_array_dup[:508, :2064] 
                                        - overscan_topleft)
        overscan_corrected[508:, :2064] = (input_array_dup[508:, :2064] 
                                        - overscan_bottomleft)
        overscan_corrected[:508, 2064:] = (input_array_dup[:508, 2064:] 
                                        - overscan_topright)
        overscan_corrected[508:, 2064:] = (input_array_dup[508:, 2064:] 
                                        - overscan_bottomright)
        
        fits_data_2.append(overscan_corrected)
    data_cube = np.dstack(fits_data_2)

    # cosmic ray zapping
    tuple = np.zeros((1016, 4128))
    for i in range(np.shape(data_cube)[0]):
        for j in range(np.shape(data_cube)[1]):
        if (np.std(data_cube[i][j]) > std):
            tuple[i][j] = np.min(data_cube[i][j])
        else:
            tuple[i][j] = np.mean(data_cube[i][j])
    return(tuple)


def min_zapping(input_dir, input_folder, file_number_list, 
                max_value_allowed = 200, max_diff = 10, yearmonth = 2308):

# reading in files
    fits_data = []
    for i in file_number_list:
        fits_data.append(fits.getdata(input_dir + str(yearmonth) 
                                  + str(input_folder) + '/' + i + '.fits', 
                                  ext = 0).astype(np.float64))
    data_cube_before_overscan = np.dstack(fits_data)

# bias correction
    fits_data_2 = []
    for i in range(len(file_number_list)):

    input_array = data_cube_before_overscan[:, :, i]

    overscan_topleft = np.median(input_array[1:516,1:31])
    overscan_bottomleft = np.median(input_array[517:1013,1:31])
    overscan_topright = np.median(input_array[1:516,4097:4128])
    overscan_bottomright = np.median(input_array[517:1013,4097:4128])

    input_array_dup = np.copy(input_array)
    overscan_corrected = np.copy(input_array)

    overscan_corrected[:508, :2064] = (input_array_dup[:508, :2064] 
                                       - overscan_topleft)
    overscan_corrected[508:, :2064] = (input_array_dup[508:, :2064] 
                                       - overscan_bottomleft)
    overscan_corrected[:508, 2064:] = (input_array_dup[:508, 2064:] 
                                       - overscan_topright)
    overscan_corrected[508:, 2064:] = (input_array_dup[508:, 2064:] 
                                       - overscan_bottomright)

    fits_data_2.append(overscan_corrected)
    data_cube = np.dstack(fits_data_2)
    min_data = np.min(data_cube, axis = 2)
    return(min_data)


## Spectrum Functions
### Lines
def lines(xmin=0, xmax=10000, labely=-0.2, legend=False):
    h_beta = 4861.363
    h_gamma = 4340.472
    h_delta = 4101.734
    h_epsilon = 3970.075
    magnesium = 5183.604
    O_II = 3911.957
    He_II = 5411.2
    He_II2 = 4685.7
    He_II3 = 4540
    S_II = 4070
    O_III = 4363
    O_I = 5577.3387
  
    lines = []
    lines.append([5889.95, "Na", 'g'])
    lines.append([5895.92, "Na", 'g'])
    # lines.append([5006.843, "O III", 'r'])
    lines.append([magnesium, "Magnesium", 'r'])
    lines.append([h_beta, "H-beta", 'b'])
    lines.append([He_II3, "He II3", 'c'])
    lines.append([h_gamma, "H-gamma", 'b'])
    lines.append([h_delta, "H-delta", 'b'])
    lines.append([h_epsilon, "H-epsilon", 'b'])
    # lines.append([O_II, "O II", 'r'])
    # lines.append([5448, "TiO", 'y'])
    lines.append([He_II, "He II", 'c'])
    lines.append([He_II2, "He II2", 'c'])

    for i in range(len(lines)):
        if (lines[i][0] < xmax and lines[i][0] > xmin):
            plt.axvline(lines[i][0], label = lines[i][1], 
                        color = lines[i][2], linewidth = 0.6)
            plt.text(lines[i][0], labely, lines[i][1], 
                     rotation=90, fontsize = 10)
    if legend is True:
        plt.legend()


# def cal_spec_plot(zap, flat, min, max, norm_min=600, 
#                   norm_max=2000, title='', color='b', linewidth=0.4):
#     a = zap / (flat)
#     spectrum_ff = grab_spectrum(a, min, max)
#     x = cal_wave(spectrum_ff, 1.69, 5015)
#     plt.figure(figsize = (14, 4))
#     plt.plot(x, spectrum_ff / spectrum_ff[norm_min: norm_max].max(), color = color, linewidth = linewidth)
#     l = len(x)
#     plt.xlim(x[0], x[0] + 3/4 * l)
#     plt.ylim(0, 1.2)
#     plt.title(title)
#     lines()

  
# def plot_pix(spectrum, xmin = 0, xmax = 4000, ymin = -0.1, ymax = 1.1, figwidth = 12, figheight = 3):
#   plt.figure(figsize = (figwidth, figheight))
#   plt.plot(spectrum, linewidth = 0.5, color = 'k')
#   plt.ylim(ymin, ymax)
#   plt.xlim(xmin, xmax)
#   plt.xticks(np.arange(xmin, xmax, 250))
#   plt.grid()
  
## Fitting

def fit_lorentzian_quad(spectrum, x, left, right, params, 
                        line, plot=True, xmin=3500, xmax=6000, 
                        ymin=-0.1, ymax=1.1, figwidth=10, figheight=3):
    wavelength_range = cal_wave_4th_order_range(np.linspace(left, right, 
                                                          right - left + 1), 
                                              params[0], params[1], 
                                              params[2], params[3], 
                                              params[4])
    a = np.where(x == wavelength_range[0])[0][0]
    b = np.where(x == wavelength_range[-1])[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a: b]))
    popt, covar = opt.curve_fit(Lorentzian_quad, 
                                x[a:b], spectrum[a:b], 
                                p0=[10 * (spectrum[a:b].max()
                                          - spectrum[a:b].max()), 
                                    sigma, mean, 0, 1, 0])
    if (plot == True):
        plt.figure(figsize = (figwidth, figheight))
        plt.plot(x, spectrum, color = 'k', linewidth = 0.6, alpha = 0.5)
        plt.plot(x, Lorentzian_quad(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), color = 'b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    velo = calc_velo(popt[2], line)
    velo_err = calc_velo_err(covar, line)
    return velo, velo_err


def fit_lorentzian_quad_blackhawk(spectrum, x, left, right, 
                                  params, params_shift_night, 
                                  line, plot=True, 
                                  xmin=3500, xmax=6000, 
                                  ymin=-0.1, ymax=1.1, 
                                  figwidth=10, figheight=3):
    # input range to wavelength_range is in pixels
    init_pix_range = np.linspace(left, right, right - left + 1)
    # calibrate that range in pixels to the other night
    new_pix_range = cal_wave_4th_order_range(
        init_pix_range, params_shift_night[0], params_shift_night[1], 
        params_shift_night[2], params_shift_night[3], params_shift_night[4])

    # calibrate the new range in the arc night's pixels, to a new range of wavelengths (to arc night)
    wavelength_new = cal_wave_4th_order_range(
        new_pix_range, params[0], params[1], params[2], params[3], params[4])

    # find the start and end indices of the range you want to fit, in terms of the new x ()
    a = np.where(x == wavelength_new[0])[0][0]
    b = np.where(x == wavelength_new[-1])[0][0]

    mean = sum(x[a: b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(Lorentzian_quad, 
                                x[a:b], spectrum[a:b], 
                                p0 = [10 * (spectrum[a:b].max() 
                                            - spectrum[a:b].max()), 
                                      sigma, mean, 0, 1, 0])
    if (plot == True):
        plt.figure(figsize = (figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_quad(x, popt[0], popt[1], 
                                    popt[2], popt[3], popt[4], 
                                    popt[5]), color='b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    velo = calc_velo(popt[2], line)
    velo_err = calc_velo_err(covar, line)
    return velo, velo_err


def fit_lorentzian_lin(spectrum, x, left, right, 
                       params, line, plot=True, 
                       xmin=3500, xmax=6000, 
                       ymin=-0.1, ymax=1.1, 
                       figwidth=10, figheight=3):
    wavelength_range = cal_wave_4th_order_range(np.linspace(left, right, 
                                                            right - left + 1), 
                                                params[0], params[1], params[2], 
                                                params[3], params[4])
    a = np.where(x == wavelength_range[0])[0][0]
    b = np.where(x == wavelength_range[-1])[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(Lorentzian_lin, x[a:b], 
                                spectrum[a:b], p0 = [spectrum[a:b].max(), 
                                                    sigma, mean, 0, 1])
    if (plot == True):
        plt.figure(figsize = (figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_lin(x, popt[0], popt[1], popt[2], 
                                popt[3], popt[4]), color = 'b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    velo = calc_velo(popt[2], line)
    velo_err = calc_velo_err(covar, line)
    return velo, velo_err


def fit_lorentzian_lin_blackhawk(spectrum, x, left, right, 
                                 params, params_shift_night, 
                                 line, plot=True, 
                                 xmin=3500, xmax=6000, 
                                 ymin=-0.1, ymax=1.1, 
                                 figwidth=10, figheight=3):
    # input range to wavelength_range is in pixels
    init_pix_range = np.linspace(left, right, right - left + 1)
    # calibrate that range in pixels to the other night
    new_pix_range = cal_wave_4th_order_range(
        init_pix_range, params_shift_night[0], 
        params_shift_night[1], params_shift_night[2], 
        params_shift_night[3], params_shift_night[4])
    # calibrate the new range in the arc night's pixels, 
    # to a new range of wavelengths (to arc night)
    wavelength_new = cal_wave_4th_order_range(
        new_pix_range, params[0], params[1], 
        params[2], params[3], params[4])

    # find the start and end indices of the range you want to fit, 
    # in terms of the new x ()
    a = np.where(x == wavelength_new[0])[0][0]
    b = np.where(x == wavelength_new[-1])[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(
        Lorentzian_lin, x[a:b], spectrum[a:b], 
        p0 = [spectrum[a:b].max(), sigma, mean, 0, 1])
    if (plot == True):
        plt.figure(figsize=(figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_lin(x, popt[0], popt[1], popt[2], 
                                   popt[3], popt[4]), color = 'b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    velo = calc_velo(popt[2], line)
    velo_err = calc_velo_err(covar, line)
    return velo, velo_err


def fit_lorentzian_const(spectrum, x, left, right, line, plot=True, 
                         xmin=3500, xmax=6000, ymin=-0.1, ymax=1.1, 
                         figwidth=10, figheight=3):
    # wavelength_range = cal_wave_4th_order_range(np.linspace(left, right, right - left + 1), params[0], params[1], params[2], params[3], params[4])
    x = np.array(x)
    a = np.where(x == left)[0][0]
    b = np.where(x == right)[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(Lorentzian_const, x[a:b], 
                                spectrum[a:b], p0 = [max(spectrum[a:b]), 
                                                     sigma, mean, 0])
    if (plot == True):
        plt.figure(figsize=(figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_const(x, popt[0], popt[1], 
                                     popt[2], popt[3]), color='b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    velo = calc_velo(popt[2], line)
    velo_err = calc_velo_err(covar, line)
    return velo, velo_err


def fit_lorentzian_lin_wave(spectrum, x, left, right, 
                            params, line, plot = True, 
                            xmin = 3500, xmax = 6000, 
                            ymin = -0.1, ymax = 1.1, 
                            figwidth = 10, figheight = 3):
    wavelength_range = cal_wave_4th_order_range(
        np.linspace(left, right, right - left + 1), 
        params[0], params[1], params[2], params[3], params[4])
    a = np.where(x == wavelength_range[0])[0][0]
    b = np.where(x == wavelength_range[-1])[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(
        Lorentzian_lin, x[a:b], spectrum[a:b], 
        p0 = [spectrum[a:b].max(), sigma, mean, 0, 1])
    if (plot == True):
        plt.figure(figsize=(figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_lin(x, popt[0], popt[1], popt[2], 
                                   popt[3], popt[4]), color='b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    return popt[2]


def fit_lorentzian_quad_wave(spectrum, x, left, right, 
                             params, line, plot = True, 
                             xmin = 3500, xmax = 6000, 
                             ymin = -0.1, ymax = 1.1, 
                             figwidth = 10, figheight = 3):
    wavelength_range = cal_wave_4th_order_range(
        np.linspace(left, right, right - left + 1), 
        params[0], params[1], params[2], params[3], params[4])
    a = np.where(x == wavelength_range[0])[0][0]
    b = np.where(x == wavelength_range[-1])[0][0]
    mean = sum(x[a:b] * spectrum[a:b]) / sum(spectrum[a:b])
    sigma = np.sqrt(sum(spectrum[a:b] * (x[a:b] - mean) ** 2) 
                    / sum(spectrum[a:b]))
    popt, covar = opt.curve_fit(
        Lorentzian_quad, x[a:b], spectrum[a:b], 
        p0 = [10 * (spectrum[a:b].max() - spectrum[a:b].max()), 
        sigma, mean, 0, 1, 0])
    if (plot == True):
        plt.figure(figsize=(figwidth, figheight))
        plt.plot(x, spectrum, color='k', linewidth=0.6, alpha=0.5)
        plt.plot(x, Lorentzian_quad(x, popt[0], popt[1], popt[2], 
                                    popt[3], popt[4], popt[5]), color='b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    return popt[2]


def fit_lorentzian_quad_pixel(spectrum, left, right, line, plot=True, xmin=3500, xmax=6000, ymin = -0.1, ymax = 1.1, figwidth = 10, figheight = 3):
    x = np.arange(0, len(spectrum))
    a = left
    b = right
    mean = sum(x[a: b] * spectrum[a: b]) / sum(spectrum[a: b])
    sigma = np.sqrt(sum(spectrum[a: b] * (x[a: b] - mean) ** 2) / sum(spectrum[a: b]))
    popt, covar = opt.curve_fit(Lorentzian_quad, x[a: b], spectrum[a: b], p0 = [10 * (spectrum[a: b].max() - spectrum[a: b].max()), sigma, mean, 0, 1, 0])
    if (plot == True):
        plt.figure(figsize = (figwidth, figheight))
        plt.plot(x, spectrum, color = 'k', linewidth = 0.6, alpha = 0.5)
        plt.plot(x, Lorentzian_quad(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), color = 'b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    return popt[2]


def fit_lorentzian_lin_pixel(spectrum, left, right, line, plot = True, xmin = 3500, xmax = 6000, ymin = -0.1, ymax = 1.1, figwidth = 10, figheight = 3):
    a = left
    b = right
    x = np.arange(0, len(spectrum))
    mean = sum(x[a: b] * spectrum[a: b]) / sum(spectrum[a: b])
    sigma = np.sqrt(sum(spectrum[a: b] * (x[a: b] - mean) ** 2) / sum(spectrum[a: b]))
    popt, covar = opt.curve_fit(Lorentzian_lin, x[a: b], spectrum[a: b], p0 = [spectrum[a: b].max(), sigma, mean, 0, 1])
    if (plot == True):
        plt.figure(figsize = (figwidth, figheight))
        plt.plot(x, spectrum, color = 'k', linewidth = 0.6, alpha = 0.5)
        plt.plot(x, Lorentzian_lin(x, popt[0], popt[1], popt[2], popt[3], popt[4]), color = 'b')
        plt.ylim(ymin, ymax)
        lines(xmin, xmax)
        plt.xlim(xmin, xmax)
    return popt[2]


### Get Spectra Functions
def file_to_spectrum(input_dir, input_folder, file_number_list, yearmonth = 2308):
    data = []
    for i in file_number_list:
        data.append(fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64))

    data_cube = np.dstack(data)
    median_data = np.median(data_cube, axis = 2)

    input_array = median_data

    overscan_topleft = np.median(input_array[1:516,1:31])
    overscan_bottomleft = np.median(input_array[517:1013,1:31])
    overscan_topright = np.median(input_array[1:516,4097:4128])
    overscan_bottomright = np.median(input_array[517:1013,4097:4128])

    input_array_dup = np.copy(input_array)
    overscan_corrected = np.copy(input_array)

    overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
    overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
    overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
    overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

    spectrum = np.median(overscan_corrected[:, 35:], axis = 0)

    return(spectrum)


def grab_spectrum(input, top_row, bottom_row):
    spectrum = np.mean(input[top_row: bottom_row, :], axis = 0)
    return(spectrum)


### Calibration Functions
def cal_wave(input, alpha, delta):
    x = np.arange(len(input))
    lambda_ang = alpha * x + delta
    return(lambda_ang)

def cal_wave_2nd_order(input, alpha, beta, delta):
    x = np.arange(len(input))
    lambda_ang = alpha * x ** 2 + beta * x + delta
    return(lambda_ang)

def cal_wave_3rd_order(input, a, b, c, d):
    x = np.arange(len(input))
    lambda_ang = a * x ** 3 + b * x ** 2 + c * x + d
    return(lambda_ang)

def cal_wave_4th_order(input, a, b, c, d, e):
    x = np.arange(len(input))
    lambda_ang = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    return(lambda_ang)

def cal_wave_4auto(input, params):
    x = np.arange(len(input))
    lambda_ang = params[0] * x ** 4 + params[1] * x ** 3 + params[2] * x ** 2 + params[3] * x + params[4]
    return(lambda_ang)

def cal_wave_4th_order_range(input, a, b, c, d, e):
    x = input
    lambda_ang = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    return(lambda_ang)

def cal_wave_4auto_range(input, params):
    x = input
    lambda_ang = params[0] * x ** 4 + params[1] * x ** 3 + params[2] * x ** 2 + params[3] * x + params[4]
    return(lambda_ang)

def cal_wave_5th_order(input, a, b, c, d, e, f):
    x = np.arange(len(input))
    lambda_ang = a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f
    return(lambda_ang)

def cal_wave_5th_order_range(input, a, b, c, d, e, f):
    x = input
    lambda_ang = a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f
    return(lambda_ang)


def find_lorentzian_maxes_old(data, range = 10):
    l = len(data)
    # array that will contain rough estimates of maxes
    array1 = np.array([])
    i = 1000
    while (i < (l - 20) and i < 4040):
        median = np.median(data[i-20:i+20])
        m8 = data[i - 8]
        m3 = data[i - 3]
        m2 = data[i - 2]
        m1 = data[i - 1]
        temp = data[i]
        p1 = data[i + 1]
        p2 = data[i + 2]
        p3 = data[i + 3]
        if (temp > m1 and temp > p1 and temp >= m2 and temp >= p2 and temp >= m3 and temp >= p3 and temp >= m8 and temp > 1.3 * median):
        array1 = np.append(array1, i)
        i += 1
    array1 = array1.astype(int)

    l_maxes = np.array([])
    param_array = np.empty([0, 4])
    covar_array = np.empty([0, 4])
    for wv in array1:
        if (wv != 3922):
        a = int(wv - range / 2)
        b = int(wv + range / 2)
        xlin = np.arange(a, b, 1)
        mean = sum(xlin * data[a: b]) / sum(data[a: b])
        sigma = np.sqrt(sum(data[a: b] * (xlin - mean) ** 2) / sum(data[a: b]))
        max = np.max(data[a: b])
        mean_out = np.mean([data[a-10: a], data[b: b+10]])
        far_mean = np.mean(data[wv - 100: wv +100])
        popt, covar = opt.curve_fit(Lorentzian_const, xlin, data[a: b], p0 = [mean * 5, sigma / 1, wv, 0])
        param_array = np.append(param_array, [popt], axis = 0)
        temp1 = np.sqrt(covar[0][0])
        temp2 = np.sqrt(covar[1][1])
        temp3 = np.sqrt(covar[2][2])
        temp4 = np.sqrt(covar[3][3])
        covar_array = np.append(covar_array, np.array([np.array([temp1, temp2, temp3, temp4])]), axis = 0)
        l_maxes = np.append(l_maxes, popt[2])
    return l_maxes, param_array, covar_array


def find_lorentzian_maxes(data, range = 10):
    l = len(data)
    # array that will contain rough estimates of maxes
    array1 = np.array([])
    i = 1000
    while (i < (l - 20)):# and i < 4040):
        median = np.median(data[i-20:i+20])
        m8 = data[i - 8]
        m3 = data[i - 3]
        m2 = data[i - 2]
        m1 = data[i - 1]
        temp = data[i]
        p1 = data[i + 1]
        p2 = data[i + 2]
        p3 = data[i + 3]
        if (temp > m1 and temp > p1 and temp >= m2 and temp >= p2 and temp >= m3 and temp >= p3 and temp >= m8 and temp > 1.1 * median):
        array1 = np.append(array1, i)
        i += 1
    array1 = array1.astype(int)

    l_maxes = np.array([])
    param_array = np.empty([0, 4])
    covar_array = np.empty([0, 4])
    fit_array = np.empty([0, 1])
    for wv in array1:
        a = int(wv - range / 2)
        b = int(wv + range / 2)
        xlin = np.arange(a, b, 1)
        mean = sum(xlin * data[a: b]) / sum(data[a: b])
        sigma = np.sqrt(sum(data[a: b] * (xlin - mean) ** 2) / sum(data[a: b]))
        max = np.max(data[a: b])
        mean_out = np.mean([data[a-10: a], data[b: b+10]])
        far_mean = np.mean(data[wv - 100: wv +100])
        # print(wv)
        if (max > 2 * far_mean):
        popt, covar = opt.curve_fit(Lorentzian_const, xlin, data[a: b], p0 = [mean * 2, sigma / 1, wv, 0])
        param_array = np.append(param_array, [popt], axis = 0)
        temp1 = np.sqrt(covar[0][0])
        temp2 = np.sqrt(covar[1][1])
        temp3 = np.sqrt(covar[2][2])
        temp4 = np.sqrt(covar[3][3])
        covar_array = np.append(covar_array, np.array([np.array([temp1, temp2, temp3, temp4])]), axis = 0)
        l_maxes = np.append(l_maxes, popt[2])
        fit_array = np.append(fit_array, 'Lorentzian')
        else:
        popt, covar = opt.curve_fit(gaussian_model, xlin, data[a: b], p0 = [mean * 2, sigma / 10, wv, 0])
        param_array = np.append(param_array, [popt], axis = 0)
        temp1 = np.sqrt(covar[0][0])
        temp2 = np.sqrt(covar[1][1])
        temp3 = np.sqrt(covar[2][2])
        temp4 = np.sqrt(covar[3][3])
        covar_array = np.append(covar_array, np.array([np.array([temp1, temp2, temp3, temp4])]), axis = 0)
        l_maxes = np.append(l_maxes, popt[2])
        fit_array = np.append(fit_array, 'Gaussian')
    return l_maxes, param_array, covar_array, fit_array


def plot_maxes(spectrum, maxes, minx = 1000, figwidth = 16 * 4, figheight = 3):
    plt.figure(figsize = (figwidth, figheight))
    x = np.arange(len(spectrum))
    plt.plot(x, spectrum, color = 'orange', linewidth = 1.9)
    for i in range(len(maxes)):
        if (maxes[i] > minx):
        plt.axvline(maxes[i], linewidth = 0.3)
        plt.text(maxes[i], 3000, i, rotation = 0, fontsize = 10)
    plt.xlim(left = minx)
  
  
def plot_maxes_lorntz(spectrum, maxes, params, minx = 1000, figwidth = 16 * 4, figheight = 3, ymin = 0, ymax = 1):
    plt.figure(figsize = (figwidth, figheight))
    x = np.arange(len(spectrum))
    plt.plot(x, spectrum, color = 'k', linewidth = 1.5, alpha = 0.3)
    for i in range(len(maxes)):
        if (maxes[i] > minx):
        # plt.axvline(maxes[i], linewidth = 0.3)
        # plt.text(maxes[i], 3000, maxes[i], rotation=90, fontsize = 10)
        xtemp = np.arange(maxes[i] - 8, maxes[i] + 8)
        plt.plot(xtemp, Lorentzian_const(xtemp, params[i][0], params[i][1], params[i][2], params[i][3]), color = 'b', alpha = 0.5)
        plt.text(params[i][2], 2000, i, rotation = 0, fontsize = 13)
        # plt.axvline(params[i][2])
    plt.xlim(left = minx)
    plt.ylim(ymin, ymax)
  
  
def plot_maxes_lorntz_gauss(spectrum, maxes, params, fit_array, minx = 1000, figwidth = 16 * 4, figheight = 3, ymin = 0, ymax = 1):
    plt.figure(figsize = (figwidth, figheight))
    x = np.arange(len(spectrum))
    plt.plot(x, spectrum, color = 'k', linewidth = 1.5, alpha = 0.3)
    for i in range(len(maxes)):
        if (maxes[i] > minx):
        if (fit_array[i] == 'Lorentzian'):
            # plt.axvline(maxes[i], linewidth = 0.3)
            # plt.text(maxes[i], 3000, maxes[i], rotation=90, fontsize = 10)
            xtemp = np.arange(maxes[i] - 8, maxes[i] + 8)
            plt.plot(xtemp, Lorentzian_const(xtemp, params[i][0], params[i][1], params[i][2], params[i][3]), color = 'b', alpha = 0.5)
            plt.text(params[i][2], 2000, i, rotation = 0, fontsize = 13)
            # plt.axvline(params[i][2])
        else:
            # plt.axvline(maxes[i], linewidth = 0.3)
            # plt.text(maxes[i], 3000, maxes[i], rotation=90, fontsize = 10)
            xtemp = np.arange(maxes[i] - 8, maxes[i] + 8)
            plt.plot(xtemp, gaussian_model(xtemp, params[i][0], params[i][1], params[i][2], params[i][3]), color = 'b', alpha = 0.5)
            plt.text(params[i][2], 2000, i, rotation = 0, fontsize = 13)
            # plt.axvline(params[i][2])
    plt.xlim(left = minx)
    plt.ylim(ymin, ymax)
  
  
def plot_mins(spectrum, x, mins, minx = 1, maxx = 10000, figwidth = 15, figheight = 3):
    plt.figure(figsize = (figwidth, figheight))
    plt.plot(x, spectrum, color = 'k', linewidth = 1)
    for i in range(len(mins)):
        if (x[mins][i] > minx and x[mins][i] < maxx):
        # plt.axvline(mins[i], linewidth = 0.3)
        # plt.text(mins[i], 3000, mins[i], rotation=90, fontsize = 10) #
        plt.axvline(x[mins][i], linewidth = 0.3)
        plt.text(x[mins][i], 0.3, x[mins][i], rotation=90, fontsize = 10)
    if (minx != 1):
        plt.xlim(left = minx)
    if (maxx != 10000):
        plt.xlim(right = maxx)
    
    
def plot_lines(lines, col, line_height = 10, max_intensity = 0, figwidth = 16 * 5, figheight = 2, xmin = 0, xmax = 6000):
    max = np.max(lines[:, 2])
    plt.figure(figsize = (figwidth, figheight))
    for i in range(len(lines)):
        if (int(lines[i, 2]) > max_intensity and lines[i, 0] < xmax and lines[i, 0] > xmin):# and lines[i, 0] < 5000 and Ar_lines[i, 0] > 4650):
        plt.axvline(lines[i, 0], ymax = line_height * (lines[i, 2] / max), color = col, linewidth = 0.5, alpha = 1)
        plt.text(lines[i, 0], 0.6, i, rotation = 0, fontsize = 10)
    plt.xlim(xmin, xmax)
  
  
def make_pixels_array(maxes, list):
    array = np.empty([0, 1])
    for i in range(len(list)):
        array = np.append(array, maxes[0][list[i]])
    return array

def make_pixels_err_array(maxes, list):
    array = np.empty([0, 1])
    for i in range(len(list)):
        array = np.append(array, maxes[2][list[i]][2])
    return array

def make_wavelength_array(list):
    array = np.empty([0, 1])
    for i in range(len(list)):
        if (list[i][0] == 1):
        array = np.append(array, Ar_lines[list[i][1]][0])
        else:
        array = np.append(array, Xe_lines[list[i][1]][0])
    return array


def calc_velo_err(covar, wave_0):
    return c * (np.sqrt(covar[2, 2]) * 10 ** (-10)) / (wave_0 * 10 ** (-10)) / 1000

def calc_velo_err_lmfit(stderr, wave_0):
    return c * (stderr * 10 ** (-10)) / (wave_0 * 10 ** (-10)) / 1000

def calc_velo(wave, wave_0):
    return c * (wave * 10 ** (-10) - wave_0 * 10 ** (-10)) / (wave_0 * 10 ** (-10)) / 1000


## Plotting Functions
def basic_plot(input_array, min = 75, max = 120, title = ''):
    plt.figure(figsize = (12 * 1.4, 3 * 1.4))
    plt.imshow(input_array, aspect = 'auto', vmin=min, vmax=max)#, norm = LogNorm(vmin = 1, vmax = 1000000))
    plt.title(title)
    plt.colorbar()
    plt.show()
  
  
def plot_limits_size(input_array, min = 75, max = 120,
                     xleft = 0, xright = 4128, ybottom = 1016, ytop = 0,
                     figsizewidth = 12 * 1.4, figsizeheight = 3 * 1.4, title = '',
                     marker = False, markerx = 0, markery = 0, markeralpha = 1, markersize = 10, markercolor = 'r'):
    plt.figure(figsize = (figsizewidth, figsizeheight))
    if (marker == True):
        plt.plot(markerx, markery, marker = 'o', alpha = markeralpha, markersize = markersize, color = markercolor, markeredgecolor = 'w')
    plt.imshow(input_array, aspect = 'auto', vmin=min, vmax=max)#, norm = LogNorm(vmin = 1, vmax = 1000000))
    plt.title(title)
    plt.xlim(xleft, xright)
    plt.ylim(ybottom, ytop)
    plt.colorbar()
    plt.show()
  
  
def plot_limits_size_line(input_array, min = 75, max = 120, xleft = 0, xright = 4128, ybottom = 1016, ytop = 0,
                     figsizewidth = 12 * 1.4, figsizeheight = 3 * 1.4, title = '', marker = False, markerx = 0, markery = 0, markeralpha = 1, markersize = 10, markercolor = 'r',
                     slope1 = 0, b1 = 0, slope2 = 0, b2 = 0):
    plt.figure(figsize = (figsizewidth, figsizeheight))
    if (marker == True):
        plt.plot(markerx, markery, marker = 'o', alpha = markeralpha, markersize = markersize, color = markercolor, markeredgecolor = 'w')
    plt.imshow(input_array, aspect = 'auto', vmin=min, vmax=max)#, norm = LogNorm(vmin = 1, vmax = 1000000))
    plt.title(title)
    plt.xlim(xleft, xright)
    plt.ylim(ybottom, ytop)
    plt.colorbar()
    x = np.arange(0, 4128)
    y1 = slope1 * x + b1
    y2 = slope2 * x + b2
    plt.plot(x, y1, color = 'w')
    plt.plot(x, y2, color = 'w')
    plt.show()
  
  
## Fits/overscan/median/normalize
def fits_overscan_median(input_dir, input_folder, file_number_list, yearmonth = 2308):
    data = []
    for i in file_number_list:

        temp = fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64)

        overscan_topleft = np.median(temp[1:516,1:31])
        overscan_bottomleft = np.median(temp[517:1013,1:31])
        overscan_topright = np.median(temp[1:516,4097:4128])
        overscan_bottomright = np.median(temp[517:1013,4097:4128])

        input_array_dup = np.copy(temp)
        overscan_corrected = np.copy(temp)

        overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
        overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
        overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
        overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

        data.append(overscan_corrected)

    data_cube = np.dstack(data)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)


def fits_overscan(input_dir, input_folder, file_number_list, yearmonth = 2308):
    data = []
    for i in file_number_list:

        temp = fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64)

        overscan_topleft = np.median(temp[1:516,1:31])
        overscan_bottomleft = np.median(temp[517:1013,1:31])
        overscan_topright = np.median(temp[1:516,4097:4128])
        overscan_bottomright = np.median(temp[517:1013,4097:4128])

        input_array_dup = np.copy(temp)
        overscan_corrected = np.copy(temp)

        overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
        overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
        overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
        overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

        data.append(overscan_corrected)

    data_cube = np.dstack(data)
    # median_data = np.median(data_cube, axis = 2)

    return(data_cube)


def fit_to_array(input_dir, input_folder, file_number, yearmonth = 2308):
    data = fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + file_number + '.fits', ext = 0).astype(np.float64)
    return(data)
    # return(data.astype(np.float64))
  
  
def median_fits(input_dir, input_folder, file_number_list, yearmonth = 2308):
    data = []
    for i in file_number_list:
        data.append(fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64))

    data_cube = np.dstack(data)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)
    # return(median_data.astype(np.float64))
  
  
def fits_overscan_normalize_median(input_dir, input_folder, file_number_list, yearmonth = 2308):
    data = []
    for i in file_number_list:

        temp = fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64)

        overscan_topleft = np.median(temp[1:516,1:31])
        overscan_bottomleft = np.median(temp[517:1013,1:31])
        overscan_topright = np.median(temp[1:516,4097:4128])
        overscan_bottomright = np.median(temp[517:1013,4097:4128])

        input_array_dup = np.copy(temp)
        overscan_corrected = np.copy(temp)

        overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
        overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
        overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
        overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

        data.append(overscan_corrected / np.median(overscan_corrected[:, 33: 4096]))

    data_cube = np.dstack(data)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)


def fits_overscan_rangenormalize_median(input_dir, input_folder, file_number_list, min = 0, max = 1, yearmonth = 2308):
    data = []
    for i in file_number_list:

        temp = fits.getdata(input_dir + str(yearmonth) + str(input_folder) + '/' + i + '.fits', ext = 0).astype(np.float64)

        overscan_topleft = np.median(temp[1:516,1:31])
        overscan_bottomleft = np.median(temp[517:1013,1:31])
        overscan_topright = np.median(temp[1:516,4097:4128])
        overscan_bottomright = np.median(temp[517:1013,4097:4128])

        input_array_dup = np.copy(temp)
        overscan_corrected = np.copy(temp)

        overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
        overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
        overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
        overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

        data.append(overscan_corrected / np.median(overscan_corrected[:, 33: 4096]))

    data_cube = np.dstack(data)
    median_data = np.median(data_cube, axis = 2)

    return(median_data)


def subtract_overscan(input_array):

    overscan_topleft = np.median(input_array[1:516,1:31])
    overscan_bottomleft = np.median(input_array[517:1013,1:31])
    overscan_topright = np.median(input_array[1:516,4097:4128])
    overscan_bottomright = np.median(input_array[517:1013,4097:4128])

    input_array_dup = np.copy(input_array)
    overscan_corrected = np.copy(input_array)

    overscan_corrected[:508, :2064] = input_array_dup[:508, :2064] - overscan_topleft
    overscan_corrected[508:, :2064] = input_array_dup[508:, :2064] - overscan_bottomleft
    overscan_corrected[:508, 2064:] = input_array_dup[:508, 2064:] - overscan_topright
    overscan_corrected[508:, 2064:] = input_array_dup[508:, 2064:] - overscan_bottomright

    return(overscan_corrected)


