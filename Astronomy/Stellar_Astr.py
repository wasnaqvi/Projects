# Read .dat file from the internet and read it into a pandas dataframe
# http://www.sns.ias.edu/~jnb/SNdata/Export/BP2000/bp2000stdmodel.dat
# The Table begins after the 23rd line

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a pandas dataframe. The table begins after the 23rd line.The column labels are in the 22nd line

url = 'http://www.sns.ias.edu/~jnb/SNdata/Export/BP2000/bp2000stdmodel.dat'
column_names = [
    "M/Msun", "R/Rsun", "T", "Rho", "P", "L/Lsun",
    "X(^1H)", "X(^4He)", "X(^3He)", "X(^12C)", "X(^14N)", "X(^16O)"
]

df = pd.read_table(url, skiprows=23, delim_whitespace=True, header=None,skipfooter=2, engine='python')
df.columns = column_names

# Plotting Mass fractions of  X(^1H)", "X(^4He)", "X(^3He) again radius of the Sun.

'''

plt.plot(df["R/Rsun"], df["X(^1H)"], label="X(^1H)")
plt.plot(df["R/Rsun"], df["X(^4He)"], label="X(^4He)")
plt.plot(df["R/Rsun"], df["X(^3He)"], label="X(^3He)")
plt.xlabel(r"$\mathrm{R/R}_\odot$") 
plt.ylabel(r"$\mathrm{Mass Fraction}$") 
plt.title(r"$\mathrm{Mass\ fractions\ of\ X}({}^1\mathrm{H}),\ X({}^4\mathrm{He}),\ X({}^3\mathrm{He})\ \mathrm{against\ radius\ of\ the\ Sun}$")  # LaTeX-formatted title
plt.legend() 
plt.show()

'''
'''
# Plotting Temperature against radius of the Sun

# Create figure and axes
fig, ax1 = plt.subplots()

# Plot Temperature on the left y-axis
ax1.plot(df["R/Rsun"], df["T"], label="Temperature", color='tab:blue',linestyle='--')
ax1.set_xlabel(r"$\mathrm{R/R}_\odot$")
ax1.set_ylabel(r"$\mathrm{Temperature\ (K)}$", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title(r"$\mathrm{Temperature\ and\ Pressure\ against\ radius\ of\ the\ Sun}$")
ax1.legend(loc='upper left')

# Create a second y-axis for Pressure
ax2 = ax1.twinx()
ax2.plot(df["R/Rsun"], df["P"], label="Pressure", color='tab:red')
ax2.set_ylabel(r"$\mathrm{Pressure\ (\mathrm{dyn/cm}^3)}$", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.show()


#Plotting Mass vs Radius and Density vs Radius
fig, ax1 = plt.subplots()
ax1.plot(df["R/Rsun"], df["M/Msun"], label="Mass", color='tab:blue',linestyle='--')
ax1.set_xlabel(r"$\mathrm{R/R}_\odot$")
ax1.set_ylabel(r"$\mathrm{Mass\ (M}_\odot)$", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title(r"$\mathrm{Mass\ and\ Density\ against\ radius\ of\ the\ Sun}$")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df["R/Rsun"], df["Rho"], label="Density", color='tab:red')
ax2.set_ylabel(r"$\rho\ (\mathrm{g/cm}^3)$", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.show()
'''
# Plotting mean mass per Hydrogen atom against radius of the Sun
# mu = 1/(X(^1H) + Y/4 + Z/15.5)
Z= df["X(^12C)"] + df["X(^14N)"] + df["X(^16O)"]
Y= df["X(^4He)"] + df["X(^3He)"]
X=(df["X(^1H)"])
mu = 1/X + Y/4 + Z/15.5
plt.plot(df["R/Rsun"], mu, label="Mean mass per Hydrogen atom")
plt.xlabel(r"$\mathrm{R/R}_\odot$")
plt.ylabel(r"$\mu$")
plt.title(r"$\mathrm{Mean\ mass\ per\ Hydrogen\ atom\ against\ radius\ of\ the\ Sun}$")
plt.legend()

plt.show()

# Plotting Rosseland Mean Opacity against radius of the Sun. Average of the summ of the kbf, kes, kff opacities

kbf = 4.34e21*0.1*Z*(1+X)*df["Rho"]*(df["T"]**(-3.5))
kff = 3.68e18*(1-Z)*(1+X)*df["Rho"]*(df["T"]**(-3.5))
kes = 0.02*(1+X)
#Average of the summ of the kbf, kes, kff opacities
K= (kbf + kff + kes)/3


# Log K vs Log T
plt.plot(np.log(df["T"]), np.log10(K), label="Rosseland Mean Opacity")

plt.xlabel(r"$\log\ T$")
plt.ylabel(r"$\log\ K$")
plt.title(r"$\mathrm{Rosseland\ Mean\ Opacity\ against\ Temperature\ on a Log Scale}$")
plt.legend()

plt.show()

# Plotting K vs Radius of the Sun
plt.plot(df["R/Rsun"], K, label="Rosseland Mean Opacity")
plt.xlabel(r"$\mathrm{R/R}_\odot$")
plt.ylabel(r"$K$")
plt.title(r"$\mathrm{Rosseland\ Mean\ Opacity\ against\ radius\ of\ the\ Sun}$")
plt.legend()

plt.show()


# Plotting $\delta ln(\P)/\delta ln(\T))$ against R/Rsun
'''
np.diff(array). The resulting array will have a length one less than the array you started with, so if you’re
plotting np.diff(array) as a function of x and x has the same length as array, you’ll want to throw out
the last element of x: “x[:-1]”, so you’ll run something like plt.plot(x[:-1], np.diff(array))
'''
dPdT = np.diff(np.log(df["P"]))/np.diff(np.log(df["T"]))
plt.plot(df["R/Rsun"][:-1], dPdT, label=r"$\frac{\delta ln(P)}{\delta ln(T)}$")
plt.xlabel(r"$\mathrm{R/R}_\odot$")
plt.ylabel(r"$\frac{\delta ln(P)}{\delta ln(T)}$")
plt.title(r"$\mathrm{Rate\ of\ change\ of\ Pressure\ with\ Temperature\ against\ radius\ of\ the\ Sun}$")
plt.legend()

plt.show()

# The plot is not smooth.
'''
s. I find it works best to separately smooth ∆ ln P and ∆ ln T before
calculating the derivative. There are lots of smoothing algorithms; I find that the Savgol filter works pretty
well. In python, first run from scipy.signal import savgol filter. Then savgol filter(array, 5,
3) smooths array 5 cells at a time with a third-order polynomial; you’ll want to experiment to find how
many zones works well for smoothing.
REFERENCES
Bahcall, J. N., Pinsonneault, M. H., & Basu, S. 2001, ApJ, 555, 990

'''
from scipy.signal import savgol_filter
# Smooth the Pressure and Temperature
dPdT_smooth = savgol_filter(dPdT, 5, 3)
plt.plot(df["R/Rsun"][:-1], dPdT_smooth, label=r"$\frac{\delta ln(P)}{\delta ln(T)}$")
plt.xlabel(r"$\mathrm{R/R}_\odot$")
plt.ylabel(r"$\frac{\delta ln(P)}{\delta ln(T)}$")
plt.title(r"$\mathrm{Smoothed Rate\ of\ change\ of\ Pressure\ with\ Temperature\ against\ radius\ of\ the\ Sun}$")
plt.legend()

plt.show()