
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

def moving_average(ydata,point=5):
    """
    moving average
    
    ydata: original np.array data
    point: average points
    
    return: moving average data
    data length of calculated data is as same as original.
    
    ref:畳み込み積分や移動平均を求めるnumpy.convolve関数の使い方
    https://deepage.net/features/numpy-convolve.html

    numpyで移動平均
    https://qiita.com/wrblue_mica34/items/51adf0059b61887075d9
    """
    
    b =np.ones(point)/point
    y_ma = np.convolve(ydata,b, mode='same')
    #  Mode ‘same’ returns output of length max(M, N). Boundary effects are still visible.
    
    return y_ma

def cent_of_gravity(x,y,bg=0,printf=False):
    """
    center of gravity method
    fwhm is calculated from variance.
    This method is greatly affected by the background,
    so it is necessary to remove the background appropriately. 
    Especially the variance is affected.

    """
    y_rembg = y-bg
    cg = np.sum(x*(y_rembg))/np.sum(y_rembg)

    vpd = y_rembg*(x-cg)*(x-cg)
    # vp2: variance sigma^2
    vp2= np.sum(vpd)/np.sum(y_rembg)
    # vp: standard deviation
    vp = np.sqrt(vp2)

    if printf==True:
        print(f'mu: {cg}, var: {vp2}, var1/2: {vp}')

    return cg, vp2

def fwhm(x,y,bg=0,printf=False):
    """
    full width at half maximum
    fwhm = 2.35*sigma (gauss distribution)
    The data needs to be smoothed.
    """
    # find max index
    ymax_ind = np.argmax(y)
    center_x= x[ymax_ind]
    ymax = y.max()

    yhalf = (ymax-bg)/2
    yhalf_ind = np.where( y-bg > yhalf)[0]
    yhalf_ind_min = yhalf_ind.min()
    yhalf_ind_max = yhalf_ind.max()

    fwhm_s = x[yhalf_ind_min]
    fwhm_l = x[yhalf_ind_max]
    fwhm = np.abs(fwhm_s-fwhm_l)
    # if the difference between the center and small edge is over 2, fwhm value set 'nan'.
    if np.abs(fwhm_l-center_x) > 2*np.abs(center_x-fwhm_s) :
         center_x, fwhm = np.nan, np.nan

    if printf==True:
        print(f'center_x: {center_x}, fwhm: {fwhm}, center_y: {ymax}, half_y: {yhalf}')

    return center_x, fwhm, ymax, yhalf


def r2_cal(ydata, fitdata):
    """
    Coefficient of determination
    """

    residuals =  ydata - fitdata
    rss = np.sum(residuals**2)
    tss = np.sum((ydata-np.mean(ydata))**2)
    r2 = 1 - (rss / tss)
    
    return r2


def gauss_fit(xdata,ydata, para4_list, printf=False):
    # para4_list = [a,mu,sigma,c]  
    try:
        popt, pcov = curve_fit(gauss, xdata,ydata, 
        p0=np.array(para4_list), maxfev=10000)  

    except RuntimeError as Rte:
#             print('RuntimeError')
        popt = (0, 0, 0, 0)

    if printf==True:
        print(f"initial parameter -> {para4_list}")
        print(f"estimated parameter -> {popt}" )
        
    return popt

# Gauss distribution
def gauss(x, a = 1, mu = 0, sigma = 1, bg = 0):
    return a *(1/(sigma*np.sqrt(2*np.pi))) *np.exp(-(x - mu)**2 / (2*sigma**2)) + bg

def gauss_pdf(x, a = 1, mu = 0, sigma = 1, bg = 0):
    """using scipy module function
    """
    return a*norm.pdf(x, mu, sigma) + bg

# make simulation data with gauss noise
def simu_data(original_data,sigma):
    data_with_noise=[]
    for iy in original_data:
        data_temp = np.random.normal(loc=iy, scale=sigma, size=1) 
        data_with_noise.append(float(data_temp))

    return np.array(data_with_noise)

if __name__ == '__main__':
    # make x data
    # -10～10まで0.05刻みの数値配列作成
    x = np.arange(-10, 10, 0.01)
    
    # N = 100
    # x = np.linspace(-10.0,10.0,N) 

    a = 1
    mu = 0
    sigma = 1
    bg = 0  

    y = gauss_pdf(x,a,mu,sigma,bg)
    # plt.plot(x,y)


    yn = simu_data(original_data=y,sigma=0.2)
    ys = moving_average(yn,point=10)

    plt.plot(x,yn,'g')
    plt.plot(x,ys,'b')
    plt.plot(x,y,'r')
    plt.show()


    cent_of_gravity(x,y,bg=0,printf=True)
    cent_of_gravity(x,yn,bg=0,printf=True)
    cent_of_gravity(x,ys,bg=0,printf=True)


    fwhm(x,y,bg=0,printf=True)
    fwhm(x,yn,bg=0,printf=True)
    fwhm(x,ys,bg=0,printf=True)


    para4_list=[1,0,1,0]
    gauss_fit(x,y, para4_list, printf=True)
    gauss_fit(x,yn, para4_list, printf=True)
    gauss_fit(x,ys, para4_list, printf=True)
