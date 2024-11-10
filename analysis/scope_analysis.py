import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Leela :
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def gaussian(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def lorentzian(self, x, a, x0, gamma):
        return a*gamma**2/((x-x0)**2+gamma**2)
    
    def fano(self, x, q, F):
        return F*(q+x)**2/(q**2+1)
    
    def fit_gaussian(self, x, y):
        popt, pcov = curve_fit(self.gaussian, x, y)
        plt.plot(x, self.gaussian(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        return popt, pcov
    
    def fit_lorentzian(self, x, y):  
        popt, pcov = curve_fit(self.lorentzian, x, y)
        plt.plot(x, self.lorentzian(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        return popt, pcov
    
    def fit_fano(self, x, y):
        popt, pcov = curve_fit(self.fano, x, y)
        plt.plot(x, self.fano(x, *popt), 'r-')
        plt.plot(x, y, 'b-')
        return popt, pcov
    

    def gaussian_residuals(self, x, y):
        return y - self.fit_gaussian(x, y)
    
    def lorentzian_residuals(self, x, y):
        return y - self.fit_lorentzian(x, y)
    
    def fano_residuals(self, x, y):
        return y - self.fit_fano(x, y)
    
    
        
    