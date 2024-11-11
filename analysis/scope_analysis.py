import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Leela :
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def gaussian(self, x, *p): #Where p is a list of parameters in order of a, x0, sigma
        y = np.zeros_like(x)
        for i in range(0, len(p), 3):
            a = p[i]
            x0 = p[i+1]
            sigma = p[i+2]
            y = y + a*np.exp(-(x-x0)**2/(2*sigma**2))
        return y
    
    def lorentzian(self, x, *p): #Where p is a list of parameters in order of a, x0, gamma
        y = np.zeros_like(x)
        for i in range(0, len(p), 3):
            a = p[i]
            x0 = p[i+1]
            gamma = p[i+2]
            y = y + a*gamma**2/((x-x0)**2+gamma**2)
        return y
    
    def fano(self, x, *p): #Where p is a list of parameters in order of F, q, x0
        y = np.zeros_like(x)
        for i in range(0, len(p), 3):
            F = p[i]
            q = p[i+1]
            x0 = p[i+2]
            y = y + F*(q+x)**2/(q**2+1)
        return y
    
    def fit_gaussian(self, x, y):
        popt, pcov = curve_fit(self.gaussian, x, y)
        plt.plot(x, self.gaussian(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.legend(['gaussian_fit', 'data'])
        plt.xlabel('Wavelenght (nm)')
        plt.ylabel('Intensity (V)')
        plt.show()
        return popt, pcov
    
    def fit_lorentzian(self, x, y):  
        popt, pcov = curve_fit(self.lorentzian, x, y)
        plt.plot(x, self.lorentzian(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.xlabel('Wavelenght (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['lorentzian fit', 'data'])
        plt.show()
        return popt, pcov
    
    def fit_fano(self, x, y):
        popt, pcov = curve_fit(self.fano, x, y)
        plt.plot(x, self.fano(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.xlabel('Wavelenght (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['fano lineshape fit', 'data'])
        plt.show()
        return popt, pcov
    
    def fhwm(self, x, y):
        max_y = max(y)
        half_max_y = max_y/2
        idx_min = (np.abs(y - half_max_y)).argmin()
        idx_max = (np.abs(y - half_max_y)).argmax()
        return np.absolute(x[idx_min] - x[idx_max])
    
    def fwhm_gaussian(self, x, y):
        popt, pcov = self.fit_gaussian(x, y)
        return 2*np.sqrt(2*np.log(2))*popt[2]
    
    def fwhm_lorentzian(self, x, y):
        popt, pcov = self.fit_lorentzian(x, y)
        return 2*popt[2]

    def gaussian_residuals(self, x, y):
        return y - self.fit_gaussian(x, y)
    
    def lorentzian_residuals(self, x, y):
        return y - self.fit_lorentzian(x, y)
    
    def fano_residuals(self, x, y):
        return y - self.fit_fano(x, y)
    
    def load_data(self, filename):
        data = np.loadtxt(filename)
        return data
    
    def window_averaging(self, array, window_size):
        window_average = np.convolve(array, np.ones(window_size)/window_size, mode='valid')
        plt.plot(window_average)
        return window_average
    
    
    

        
    