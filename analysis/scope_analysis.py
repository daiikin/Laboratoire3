import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.signal import find_peaks

class Leela():
    def __init__(self):
        print("Initializing Leela")
        self.celeretite = 299792458 #m/s
        self.a_channel1 = 1.18 #s/nm value of coefficient for cavity 1 (longest cavity)
        self.a_channel2 = 0.112 #s/nm value of coefficient for cavity 2 (shortest cavity)
        self.time_bin = 1e-7 #time bin in seconds
        self.lambda_middle = 1568
        self.lambda_delta = 0.038
        self.lambda_div = self.lambda_delta/1e6

        self.channel1 = self.load_data("output/channel1_trace_20241125.npy")
        self.channel2 = self.load_data("output/channel2_trace_20241125.npy")
        print(self.channel1)
        
        self.data1 = self.window_averaging(self.channel1, 10000)
        self.data2 = self.window_averaging(self.channel2, 10000)
        self.split1 = self.split_peaks(self.data1)
        self.split2 = self.split_peaks(self.data2)
        self.fsr_cavity1 = 0.038 #in nm
        self.fsr_cavity2 = 0.4 #in nm
        self.fsr_cavity1_freq = self.celeretite/self.lambda_middle**2 * self.fsr_cavity1 * 1e3
        self.fsr_cavity2_freq = self.celeretite/self.lambda_middle**2 * self.fsr_cavity2 * 1e3
        self.gammas1 = []
        self.gammas2 = []
    
        """
        self.fit_fano(self.split1[5])
        """
        for peaks in self.split1:
            if peaks.size > 0:
                popt = self.fit_fano(peaks, 1)[0]
                self.gammas1.append(popt[3])
            else:
                print("No peaks found")
        for peaks in self.split2:
            if peaks.size > 0:
                popt = self.fit_fano(peaks, 2)[0]
                self.gammas2.append(popt[3])
            else:
                print("No peaks found")

        self.gamma1_avg = np.mean(self.gammas1)
        self.gamma2_avg = np.mean(self.gammas2)
        self.finesse1 = self.fsr_cavity1_freq/self.gamma1_avg
        self.finesse2 = self.fsr_cavity2_freq/self.gamma2_avg
        print(f"Finesse cavity 1: {self.finesse1}")
        print(f"Finesse cavity 2: {self.finesse2}")
        



        

    def __enter__(self):
        return self
   
    def __exit__(self, exc_type, exc_value, traceback):
        return

    ### wvl in nm, freq out in MHz
    def wvl2freq(self, wvl):
        return self.celeretite / (wvl * 1e-9) / 1e6
    
    def gaussian(self, x, a, x0, sigma, C): #Where p is a list of parameters in order of a, x0, sigma
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + C
    
    def lorentzian(self, x, a, x0, gamma, C): #Where p is a list of parameters in order of a, x0, gamma
        return a*gamma**2/((x - x0)**2+gamma**2) + C
    
    def fano(self, x, A, q, x0, gamma, C): #Where p is a list of parameters in order of F, q
        numerator = (q + (x - x0 )/gamma)**2
        denominator = 1 + ((x - x0)/gamma)**2
        return A*numerator/denominator + C
      
    
    def fit_gaussian(self, y):
         
        x = np.linspace(self.lambda_middle - self.lambda_delta, self.lambda_middle + self.lambda_middle , len(y))
        
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
    
    def fit_lorentzian(self, y): 
        
        x = np.linspace(self.lambda_middle - self.lambda_delta, self.lambda_middle + self.lambda_middle , len(y))
        w = self.a_channel1 * self.wvl2freq(x) / self.time_bin

        popt, pcov = curve_fit(self.lorentzian, w, y, maxfev=1000)
        plt.plot(w, self.lorentzian(w, *popt), 'r-')
        plt.plot(w, y, 'b-')
        plt.xlabel('Wavelenght (nm)')
        plt.ylabel('Intensity (V)')
        plt.legend(['lorentzian lineshape fit', 'data'])
        plt.show()

        return popt, pcov
    
    def fit_fano(self, y, channel):
        x = np.arange(len(y))
        peak_index = np.argmax(y)
        print(peak_index, "peak index")
        x_centered = x - peak_index
        if channel == 1:
            wvls = x_centered * self.time_bin / self.a_channel1 + self.lambda_middle
        elif channel == 2:
            wvls = x_centered * self.time_bin / self.a_channel2 + self.lambda_middle
        w = self.wvl2freq(wvls) ## in MHz

        if channel == 1:
            w0_guess = w[np.argmax(y)]
            A_guess = -max(y) * 15
            C_guess = -min(y)/10
            gamma_guess = self.fwhm(y, w)*0.1
            q_guess = 0.1
            
        elif channel == 2:
            w0_guess = w[np.argmax(y)]
            A_guess = -max(y)
            C_guess = 0
            gamma_guess = self.fwhm(y, w)
            q_guess = 0.1
        guess = [A_guess, q_guess, w0_guess, gamma_guess, C_guess]
        plt.plot(w, self.fano(w, *guess), 'g-')
        popt, pcov = curve_fit(self.fano, w, y, p0=guess, maxfev=10000)
        print(popt)
        plt.plot(w, self.fano(w, *popt), 'r-')
        plt.plot(w, y, 'b-')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Intensity (V)')
        plt.legend([f"Guess",f'Fano lineshape fit, gamma = {popt[3]:.4f}$\pm${pcov[3][3]:.4f}', 'data'])
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
        # plt.plot(window_average)
        # plt.show()
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
        
        x = np.linspace(self.lambda_middle - self.lambda_delta, self.lambda_middle + self.lambda_middle , len(y))
    
        datamin = np.min(y)
        peaks, _ = find_peaks(y, height = 10 + datamin, width = 1000)
        x_peaks = x[peaks]
        y_peaks = y[peaks]
        split_peaks = []

        for peak in peaks:
            right = peak + 50000 
            left = peak - 50000
            split_peaks.append(y[left:right])
            # plt.plot(y[left:right])
            # plt.show()
        
        return split_peaks
               
if __name__ == "__main__":
    Leela()
    # with Leela() as lc:
    #     import code 
    # code.interact(local=locals())