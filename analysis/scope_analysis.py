import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.signal import find_peaks

class Leela():
    def __init__(self):
        print("Initializing Leela")
        self.celeretite = 299792458 #m/s
        self.channel1 = self.load_data("output/channel1_trace.npy")
        self.channel2 = self.load_data("output/channel2_trace.npy")
        self.data1 = self.window_averaging(self.channel1, 10000)
        self.data2 = self.window_averaging(self.channel2, 10000)
        self.split1 = self.split_peaks(self.data1)
        self.split2 = self.split_peaks(self.data2)
        self.fit_fano(self.split1[1])

    def __enter__(self):
        return self
   
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def gaussian(self, x, a, x0, sigma, C): #Where p is a list of parameters in order of a, x0, sigma
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + C
    
    def lorentzian(self, x, a, x0, gamma, C): #Where p is a list of parameters in order of a, x0, gamma
        return a*gamma**2/((1/x - 1/x0)**2+gamma**2) + C
    
    def fano(self, x, A, q, x0, gamma, C): #Where p is a list of parameters in order of F, q
        numerator = (q + self.celeretite*(1/(x - x0 + 0.00001))/gamma)**2
        denominator = 1 + (self.celeretite*(1/(x - x0 + 0.00001))/gamma)**2
        return A*numerator/denominator + C
      
    
    def fit_gaussian(self, y, x_start=0, x_end=None):
        
        if x_end is None:
            x_end = len(y)  
        x = np.linspace(x_start, x_end, len(y))
        
        guess_sigma = self.fwhm(y, x)/2
        guess = [max(y), np.mean(x), guess_sigma, min(y)]
        popt, pcov = curve_fit(self.gaussian, x, y, p0=guess, maxfev=1000)

        plt.figure(figsize=(8, 6))
        plt.plot(x, self.gaussian(x, *popt), 'r-', label='Gaussian Fit')
        plt.plot(x, y, 'b-', label='Data')
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (V)')
        plt.title('Gaussian Fit')
        plt.grid(True)
        plt.show()
        
        return popt, pcov
    
    def fit_lorentzian(self, y, x_start = 0, x_end = None): 
        if x_end is None:
            x_end = len(y)
        x = np.linspace(x_start, x_end, len(y)) 
        popt, pcov = curve_fit(self.lorentzian, x, y, maxfev=10000)
        plt.plot(x, self.lorentzian(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.xlabel('Wavelenght (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['lorentzian lineshape fit', 'data'])
        plt.show()

        return popt, pcov
    
    def fit_fano(self, y, x_start = 0, x_end = None):
        if x_end is None:
            x_end = len(y)
        x = np.linspace(x_start, x_end, len(y))
        x0_guess = x[np.argmax(y)]
        A_guess = max(y) * 10
        C_guess = min(y)
        gamma_guess = self.fwhm(y, x)*0.1
        q_guess = 0.1
        guess = [A_guess, q_guess, x0_guess, gamma_guess, C_guess]
        popt, pcov = curve_fit(self.fano, x, y, p0=guess, maxfev=1000)
        print(popt)
        plt.plot(x, self.fano(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['Fano lineshape fit', 'data'])
        plt.show()
        
        return popt, pcov

    def fwhm(self, y, x):
        max_y = max(y)
        half_max_y = max_y/2
        idx_min = (np.abs(y - half_max_y)).argmin()
        idx_max = (np.abs(y - half_max_y)).argmax()
        return np.absolute(x[idx_min] - x[idx_max])
    '''
    #functions to be further modified for x because ill defined in the current state
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
    '''
    
    def load_data(self, filename):
        data = np.load(filename)
        return data
    
    
    def window_averaging(self, array, window_size):
        window_average = np.convolve(array, np.ones(window_size)/window_size, mode='valid')
        plt.plot(window_average)
        plt.show()
        return window_average
    ''' #split peaks function way too fancy and doesnt work
    def split_peaks(self, y:np.array, x_start=0, x_end=None):
        if x_end is None:
            x_end = len(y)
        x = np.linspace(x_start, x_end, len(y))
    
        peaks, _ = find_peaks(y)
        x_peaks = x[peaks]
        y_peaks = y[peaks]
        split_peaks = []

        if len(peaks) == 0:
            split_peaks.append(y)
        else:
            boundaries = [0] + [(peaks[i - 1] + peaks[i]) // 2 for i in range(1, len(peaks))] + [len(y)]
            for i in range(len(boundaries) - 1):
                split_peaks.append(y[boundaries[i]:boundaries[i + 1]])

        return split_peaks
        '''
    def split_peaks(self, y:np.array, x_start=0, x_end=None):
        if x_end is None:
            x_end = len(y)
        x = np.linspace(x_start, x_end, len(y))
    
        peaks, _ = find_peaks(y, height = 10, width = 1000)
        x_peaks = x[peaks]
        y_peaks = y[peaks]
        split_peaks = []

        for peak in peaks:
            right = peak + 100000
            left = peak - 100000
            split_peaks.append(y[left:right])
        
        return split_peaks
               
if __name__ == "__main__":
    with Leela() as lc:
        import code 
    code.interact(local=locals())