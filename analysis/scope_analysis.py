import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.signal import find_peaks

class Leela():
    def __init__(self):
        print("Initializing Leela")

    def __enter__(self):
        return self
   
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def gaussian(self, x, a, x0, sigma): #Where p is a list of parameters in order of a, x0, sigma
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    def lorentzian(self, x, a, x0, gamma): #Where p is a list of parameters in order of a, x0, gamma
        return a*gamma**2/((x-x0)**2+gamma**2)
    
    def fano(self, x, F, q): #Where p is a list of parameters in order of F, q
        return F*(q + x)**2/(q**2 + 1)
      
    
    def fit_gaussian(self, y, x_start=0, x_end=None):
        # Generate x values
        if x_end is None:
            x_end = len(y)  # Use the length of y if x_end is not provided
        x = np.linspace(x_start, x_end, len(y))

        # Fit the Gaussian model to the data
        popt, pcov = curve_fit(self.gaussian, x, y)

        # Plot the data and the Gaussian fit
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
        popt, pcov = curve_fit(self.lorentzian, x, y)
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
        popt, pcov = curve_fit(self.fano, x, y)
        plt.plot(x, self.fano(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['Fano lineshape fit', 'data'])
        plt.show()
        
        return popt, pcov

    def fhwm(self, y, x_start = 0, x_end = None):
        if x_end is None:
            x_end = len(y)
        x = np.linspace(x_start, x_end, len(y))
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

    

        
    