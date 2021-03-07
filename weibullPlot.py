import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers
import matplotlib.scale
import probscale
import pandas as pd
from scipy import stats
from scipy import special


# weibCDF: defines the CDF (reliability function) for a 2 parameter Weibull function where the location
# parameter, gamma = 0. In order to make a "standard" weibuill distribution, alpha = 1.

# alpha = ?
# beta = shape parameter
# eta = scale parameter
# gamma = location parameter

def weibCDF(failureData,shapePara,scalePara):
        return (1 - np.exp(-(failureData / scalePara)**shapePara))

# unreliabilityTrans: transformation of a 

def unreliabilityTrans(qT,beta,eta): #y-axis transformation from linear sapce to Q(t)
        return (np.exp((np.log(np.log(1/(1-qT))))/beta + np.log(eta)))

def bernardsEstimate():         # more simple estimate for median ranks only of unreliability.
        mr = []                 # could replace betaBinoCDFsolver(x,p=0.50)
        i = 0
        while i < len(x):
                mr.append(((i+1)-0.3)/(len(x)+0.4)*100)
                i += 1
        return mr

def binomCoeff(a,b):            # simple function to call scipy nCr method.
        return special.comb(a,b, exact=True)

def betaBinoCDFsolver(x,p):     # returns an ordered list of unreliability estimates by solving the
        n = len(x)              # cumulative binomial equation
        rank = []
        coeff = list([0.])*(n+1)
        coeff[n] = -p
        k = n
        while k > 0 :
                i = n
                while i >= 0:
                        try: coeff[n-i] += (-1)**(i-k) * binomCoeff(n,k) * binomCoeff(n-k,i-k)
                        except: break
                        i -= 1
                
                polyRoots = np.roots(coeff) #calculates all roots (real & complex)
                realRoots = polyRoots[np.isreal(polyRoots)].real #returns only real roots
                
                ctr1 = 0
                while ctr1 < len(realRoots): #loop to locate the desired solution between 0.00 & 1.00
                    if (realRoots[ctr1] >= 0.0 and realRoots[ctr1] <= 1.0):
                        kRank = realRoots[ctr1] * 100
                    ctr1 += 1

                rank.insert(0,kRank)
                k -= 1
        
        #idea: use stirling's approximation instead of binomCoeff() for large values of N

        return rank


# def preferredAutoLogScale(whichAxis,selfData):     #will work on the current axes
#         tempAx = fig.gca()
#         if whichAxis == 'x':
#                 if np.log10(selfData[-1]/selfData[0]) <= 1 : #if the range of your data is <= 1 set of major log tics
#                         temporary1 = 10 #Need to figure out how to get current axes length *for the correct axes* (different than figure?)
#                         numOfAxisTicks = int(np.round(temporary1/3))
#                         tempAx.
#                 do stuff
#         elif whichAxis == 'y':
#                 do stuff
#         else : print("***Warning: neither x nor y axis was specified. AutoScale failed.")



###open and read data from .csv###
# 
# tbd 



#  #placeholder for reading data
myDiction = {'Sample': np.arange(8),                                                          #placeholder until
        'Measured Value': [4.230, 4.050, 4.370, 4.200, 4.510, 4.170, 4.250, 4.440],        #implement open / read
        'Suspended?':list("NNNNNNNN")}
data = pd.DataFrame(myDiction)


#  #extract needed data from dataFrame into lists
x = list(data['Measured Value'])
x.sort()  #rank this list -- very important that data values are ranked for plotting against median ranks



###fit data to weibull dist###

weibullParameters = stats.exponweib.fit(x,floc=0, f0=1)

alpha = weibullParameters[0]    #floc = alpha = 0
beta = weibullParameters[1]     #beta = shape parameter
gamma = weibullParameters[2]    #gamma = f0 = 1 = location parameter
eta = weibullParameters[3]      #eta = scale parameter


#  #calculate y-plotting positions (median ranks)
#    #bernards estimate for median ranks

#    #OR use

#    #cumulative binomial solution
mr = betaBinoCDFsolver(x, 0.5)


#  #calculate Weibull best fit line
x2 = np.linspace(unreliabilityTrans(0.002,beta,eta),unreliabilityTrans(0.998,beta,eta),20)
weibUnreliability = weibCDF(x2,beta,eta)


###confidence intervals###

# #calculate upper and lower bounds on the ranks / plotting positions
lowerRanks = betaBinoCDFsolver(x, 0.25)
upperRanks = betaBinoCDFsolver(x, 0.75)

# #create best fit lines using the calulated Weibull parameters
# lowerCI = 

###plot configuration###

#  #size of figure
figHeight = 12
figWidth = 20
fig = plt.figure(figsize=(figWidth,figHeight),facecolor=(225/255,225/255,225/255))


#  #add Text reporting weibull parameters
ax1 = fig.add_subplot(1,5,1)
ax1.axis("off")

results_preferences = dict(fontsize=24, ha = 'right', va = 'center')

ax1.text(0.1, 0.7, "Two-parameter \nWeibull Fit\n\n"+
                r"$Q(t)=1-e^{-\left(\frac{t}{\eta}\right)^\beta}$",     #r prefix for a string makes the string
                fontsize=28, ha = 'center')                              #"raw" (ignores string literals \b & \a)
ax1.text(0, 0.6,r"$\beta =$"+"\n"+r"$\eta =$",**results_preferences)
ax1.text(0.45, 0.6,str(np.round(beta,2))+"\n"+str(np.round(eta,2)),**results_preferences)


#  #create weibull plot
ax2 = fig.add_subplot(1,5,(2,5))

marker_style = dict(color='tab:blue',linestyle='none',marker='o',fillstyle='none',
                        markersize=7,markeredgewidth=3)

ax2.semilogx(x,lowerRanks, color='red',linestyle='none',marker='.',fillstyle='none',
                        markersize=7,markeredgewidth=3)
ax2.semilogx(x,upperRanks, color='black',linestyle='none',marker='.',fillstyle='none',
                        markersize=7,markeredgewidth=3)
ax2.semilogx(x2,weibUnreliability*100, linestyle='-',color=(120/255,120/255,120/255),linewidth=1)
ax2.semilogx(x,mr, **marker_style)
# ax2.semilogx(x,lowerRanks, linestyle='-',color=(80/255,80/255,80/255),linewidth=1)
# ax2.semilogx(x,upperRanks, linestyle='-',color=(120/255,120/255,120/255),linewidth=1)


ax2.set_yscale('prob',dist=stats.weibull_min(beta))
ax2.set_ylim(bottom=0.2, top = 99.8)
ax2.set_xlim(left = 10**(np.floor(np.log10(x[0]))), right = 10**(np.ceil(np.log10(x[-1]))) )#set limits at log ticks
ax2.grid(True, which='both', **dict(color=(224/255,224/255,224/255)))
ax2.set_xlabel("Torque $[kNm]$", fontsize=18)
ax2.set_ylabel("Unreliability, $Q(x) [\\%]$", fontsize=18)

plt.show()


