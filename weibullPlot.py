import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
import matplotlib.scale
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator, LogLocator
import pandas as pd
from scipy import stats
from scipy import special

class weibullCDFScale(mscale.ScaleBase):
    """
    Transforms data in range 0 to 1 (non-inclusive) by
    applying a double log reciprocol.

    The scale function:
      ``ln(ln(1/(1-y)))``

    The inverse scale function:
      ``1/ln(ln(1/(1-y)))``

    Since the Weibull scale is undefined <= 0 and >= 1,
    there is required threshold of 0 < y < 1, above and below 
    which nothing will be plotted.

    This class was constructed using a matplotlib docs example,
    MercatorLatitudeScale.

    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``gca().set_yscale("weibull")`` would
    # be used to select this scale.
    name = 'weibull'

    def __init__(self, axis, *, domainLower = 0, domainUpper = 1, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        domainUpper: Upper bound of the function domain, 1.
        domainLower: Lower bound of the function domain, 0.
        """
        super().__init__(axis)
        self.domainUpper = domainUpper
        self.domainLower = domainLower

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The ``WeibullCDFTransform`` class is defined below as a
        nested class of this one.
        """
        return self.WeibullCDFTransform(self.domainLower, self.domainUpper)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.

        In our case, the Weibull CDF must be constrained between 0 and
        1, with typical limits being at 0.1% and 99.8%. A custom formatter
        class is used to specifiy one decimal place, scale by 100%,
        and display the percent sign.
        """
        class ReliabilityFormatter(Formatter):
            def __call__(self, x, pos=None):
                return "%.1f%%" % (100*x)

        axis.set_major_locator(FixedLocator(np.array([0.001,0.005,0.01,0.05,0.1,0.50,0.99])))
        axis.set_minor_locator(FixedLocator(np.array([0.002,0.003,0.004,0.006,0.007,0.008,0.009,
                          0.02,0.03,0.04,0.06,0.07,0.08,0.09,0.2,0.3,0.4,0.6,0.7,0.8,0.9])))
        axis.set_major_formatter(ReliabilityFormatter())
        

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of a Weibull CDF, the
        bounds should be limited to 0 < y < 1.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, self.domainLower), min(vmax, self.domainUpper)

    class WeibullCDFTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # input_dims and output_dims specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, domainLower, domainUpper):
            mtransforms.Transform.__init__(self)
            self.domainLower = domainLower
            self.domainUpper = domainUpper

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the scale
            is limited to ``0 < y 1``, the input array must be masked to
            contain only valid values.  ``matplotlib`` will handle
            masked arrays and remove the out-of-range data from the plot.
            Importantly, the transform method *must* return an array that
            is the same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            masked = np.ma.masked_where((a <= self.domainLower) | (a >= self.domainUpper), a)
            if masked.mask.any():
                return np.ma.log(np.ma.log(1/(1-masked)))
            else:
                return np.log(np.log(1/(1-a)))
            return 

        def inverted(self):
            """
            Override this method so ``matplotlib`` knows how to get the
            inverse transform for this transform.
            """
            return weibullCDFScale.InvertedWeibullCDFTransform(
                self.domainLower, self.domainUpper)

    class InvertedWeibullCDFTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, domainLower, domainUpper):
            mtransforms.Transform.__init__(self)
            self.domainLower = domainLower
            self.domainUpper = domainUpper

        def transform_non_affine(self, a):
            return 1/np.log(np.log(1/(1-a)))

        def inverted(self):
            return weibullCDFScale.WeibullCDFTransform(self.domainLower,
                self.domainUpper)

def weibCDF(failureData,shapePara,scalePara):
    """
    Defines the CDF (reliability function) for a 2 parameter Weibull function where the location
    parameter, gamma = 0. In order to make a "standard" weibuill distribution, alpha = 1.

    ``beta`` = shape parameter
    ``eta`` = scale parameter
    ``gamma`` = location parameter
    """
    return (1 - np.exp(-(failureData / scalePara)**shapePara))

def weibCDFinv(qT,beta,eta):
    """
    Inverse of ''weibCDF''. For a given 2-parameter distribution, transforms from a given
    unreliability to a failure time.

    ``qT`` = Unreliability
    ``beta`` = shape parameter
    ``eta`` = scale parameter
    """
    return (np.exp((np.log(np.log(1/(1-qT))))/beta + np.log(eta)))

def bernardsEstimate(x):
    """
    Quick estimate for median ranks of the unreliability of sample data.
    *Same as solving the cumulative beta binomial equation for P = 0.50 , but 
    much less intensive*

    ``x`` = set of sample data
    """

    mr = []
    i = 0
    while i < len(x):
        mr.append(((i+1)-0.3)/(len(x)+0.4))
        i += 1
    return mr

def FMvariance(x,beta,eta):
    """
    An esitmation of parameter variance using the weibull Fisher
    Information matrix.

    Computationally demanding way of estimating the variance of each parameter.

    ``x`` = Sample data
    ``beta`` = shape parameter
    ``eta`` = scale parameter

    """

    i = 0
    N = len(x)
    sDet = 0.
    varBeta = 0.
    varEta = 0.

    while i < N :
            k = 0
            aai = 0.
            abi = 0.
            adi = 0.
            bbi = 0.
            while k < N :
                    aai += (x[i-k]*x[k]/(eta**2))**beta
                    abi += (x[i-k]*x[k]/(eta**2))**beta * np.log(x[k]/eta)
                    adi += (x[i-k]*x[k]/(eta**2))**beta * np.log(x[k]/eta)**2
                    bbi += (x[i-k]*x[k]/(eta**2))**beta * np.log(x[i-k]/eta) * np.log(x[k]/eta)
                    
                    k += 1
            sDet += eta**(-2) * (-((N/beta)+1) + (((2*N+1)*beta+N)/beta)*(x[i]/eta)**(beta) + beta*(N+1)*(x[i]/eta)**(beta)*np.log(x[i]/eta)
                    - N*beta * (x[i]/eta)**(beta) * (np.log(x[i]/eta))**(2) - aai - 2*beta*abi + beta*(beta+1)*adi - beta**(2)*bbi)
            varBeta += sDet**(-1) * (-beta/(eta**(2)) * (1 - (beta+1)*(x[i]/eta)**(beta)))
            varEta += sDet**(-1) * (1/((beta)**(2)) + (x[i]/eta)**(beta) * (np.log(x[i]/eta))**(2))
            
            i +=1
    return (varBeta, varEta)

def mplPlotData(fig,ax,x,y,**kwargs):
    
    return ()

# Register the above scale class weibullCDFScale so matplotlib can find it.
mscale.register_scale(weibullCDFScale)

###open and read data from .csv###
# 
# tbd 



#  #placeholder for reading data
myDiction = {'Sample': np.arange(8),
        'Measured Value': [4.230, 4.050, 4.370, 4.200, 4.510, 4.170, 4.250, 
        4.440],
        'Suspended?':list("NNNNNNNN")}
data = pd.DataFrame(myDiction)


#  #extract needed data from dataFrame into lists
# x = list(data['Measured Value'])
x = [16,34,53,75,93,120]
x.sort()  #rank this list -- very important that data values are ranked for plotting against median ranks



###fit data to weibull dist###

weibullParameters = stats.exponweib.fit(x,floc=0, f0=1)

alpha = weibullParameters[0]    #floc = alpha = 0
beta = weibullParameters[1]     #beta = shape parameter
gamma = weibullParameters[2]    #gamma = f0 = 1 = location parameter
eta = weibullParameters[3]      #eta = scale parameter


#  #calculate y-plotting positions (median ranks)
#    #bernards estimate for median ranks
mr = bernardsEstimate(x)

#    #OR use

#    #cumulative binomial solution


#  #calculate Weibull best fit line
x2 = np.linspace(weibCDFinv(0.001,beta,eta),weibCDFinv(0.998,beta,eta),20)
weibUnreliability = weibCDF(x2,beta,eta)


###confidence intervals###

# #calculate upper and lower bounds on the weibull parameters
k_alpha = stats.norm.ppf(0.75)  #constant calculated by inverse normal CDF of a value based on confidcence interval (0.75 = 50% CI)
                                #ppf = percent point function, similar to quartiles?
(varBeta, varEta) = FMvariance(x,beta,eta)

betaUpper = beta * np.exp(k_alpha/beta * np.sqrt(varBeta))
betaLower = beta * np.exp(-k_alpha/beta * np.sqrt(varBeta))
etaUpper = eta * np.exp(k_alpha/eta * np.sqrt(varEta))
etaLower = eta * np.exp(-k_alpha/eta * np.sqrt(varEta))


# #create confidence bounds best fit lines using the limit Weibull parameters
x3 = np.linspace(weibCDFinv(0.001,betaLower,etaLower),weibCDFinv(0.998,betaUpper,etaUpper),20)
upperBound = weibCDF(x3,betaUpper,etaUpper)

# x4 = np.linspace(weibCDFinv(0.001,betaLower,etaLower),weibCDFinv(0.998,betaLower,etaLower),20) #just made x3 slightly longer, lower to upper limit.
lowerBound = weibCDF(x3,betaLower,etaLower)

# #R99C50 calculation
r99c50 = weibCDFinv(0.01,betaLower,etaLower)

###plot configuration###

#  #size of figure
figscale = 4
figHeight = 3 * figscale
figWidth = 5 * figscale
fig = plt.figure(figsize=(figWidth,figHeight),facecolor=(225/255,225/255,225/255))


#  #add Text reporting weibull parameters
ax1 = fig.add_subplot(1,5,1)
ax1.axis("off")

results_preferences = dict(fontsize=int(5.75*figscale), ha = 'right', va = 'center')

ax1.text(0.1/4*figscale, 0.7/4*figscale, "Two-parameter \nWeibull Fit\n\n"+
                r"$Q(t)=1-e^{-\left(\frac{t}{\eta}\right)^\beta}$",     #r prefix for a string makes the string
                fontsize=int(np.floor(7*figscale)), ha = 'center')     #"raw" (ignores string literals \b & \a)
ax1.text(0, 0.63/4*figscale,r"$\beta =$",**results_preferences)
ax1.text(0, 0.565/4*figscale,r"$\eta =$",**results_preferences)
ax1.text(0.45/4*figscale, 0.63/4*figscale,str('%.2f' % np.round(beta,2)),**results_preferences)
ax1.text(0.45/4*figscale, 0.565/4*figscale,str('%.2f' % np.round(eta,2)),**results_preferences)
ax1.text(0.7/4*figscale, 0.6/4*figscale,"+"+str('%.2f' % np.round(betaUpper-beta,2))+"\n"+str('%.2f' % np.round(betaLower-beta,2))
        +"\n\n+"+str('%.2f' % np.round(etaUpper-eta,2))+"\n"+str('%.2f' % np.round(etaLower-eta,2)),
        ha = 'right', va = 'center', fontsize=int(np.floor(3*figscale)))
ax1.text(0, 0.4/4*figscale,r"R99C50 =",**results_preferences)
ax1.text(0.725/4*figscale, 0.4/4*figscale,str('%.3f' % np.round(r99c50,3)) + " $kNm$",fontsize=int(5.75*figscale),
        ha='right', va='center', color='tab:red')


#  #create weibull plot
ax2 = fig.add_subplot(1,5,(2,5))

ax2.set_title("Group B",**results_preferences)
ax2.set_ylim(bottom=0.002, top = 0.998)
ax2.set_yscale('weibull')   #custom mscaled defined in weibullCDFScale, requires data from 0 < y < 1
# ax2.set_xlim(left = 10**(np.floor(np.log10(r99c50))), right = 10**(np.floor(np.log10(x[-1])))) #set limits at log ticks
ax2.grid(True, which='both', **dict(color=(224/255,224/255,224/255)))
ax2.set_xlabel("Torque $[kNm]$", fontsize=int(np.floor(9/2*figscale)))
ax2.set_ylabel("Unreliability, $Q(x) [\\%]$", fontsize=int(np.floor(9/2*figscale)))
ax2.tick_params(axis='both', which='both', labelsize = int(np.floor(3*figscale)))
ax2.tick_params(axis='x', which='minor', bottom='flase', labelsize = int(np.floor(3*figscale)))

marker_style = dict(color='tab:blue',linestyle='none',marker='o',fillstyle='none',
         markersize=int(np.floor(7/3*figscale)),markeredgewidth=int(np.floor(3/4*figscale)))

ax2.semilogx([ax2.get_xlim()[0], r99c50],[0.01,0.01],color='tab:red',linestyle='--',marker='None', linewidth=1)
# ax2.semilogx([r99c50, r99c50],[0.001,0.01],color='tab:red',linestyle='--',marker='None', linewidth=1)
ax2.semilogx(x2,weibUnreliability, linestyle='-',color=(120/255,120/255,120/255),linewidth=1)
ax2.semilogx(x,mr, **marker_style)
ax2.semilogx(x3,lowerBound, linestyle='--',color=(160/255,160/255,160/255),linewidth=1)
ax2.semilogx(x3,upperBound, linestyle='--',color=(160/255,160/255,160/255),linewidth=1)
ax2.semilogx(r99c50,0.01, color='tab:red', linestyle='none', marker='.', fillstyle='none',
        markersize=int(np.floor(5/2*figscale)), markeredgewidth=int(np.floor(figscale)))


plt.show()


