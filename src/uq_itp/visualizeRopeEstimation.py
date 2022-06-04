# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from nd2reader import ND2Reader
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats, signal

# +
mpl.style.use(['science', "bright"])

mpl.rcParams['figure.dpi'] = 300

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["xtick.top"] = False
mpl.rcParams["ytick.right"] = False

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['figure.titlesize'] = 9

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

mpl.use("pgf")

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all]{siunitx}",
        r'\usepackage{mathtools}',
        r'\DeclareSIUnit\pixel{px}'
        ,r"\usepackage{sansmathfonts}"
        ,r"\usepackage[scaled=0.95]{helvet}"
        ,r"\renewcommand{\rmdefault}{\sfdefault}"
        ])
    }

plt.rcParams.update(pgf_with_latex)


# -

# define gaussian shaped fit function
def skewed(x, a, c, w, alpha, off):
    return a*np.exp(-(c - x)**2/2/w**2) * (1 - erf((alpha*(c - x))/np.sqrt(2)/w)) + off


# +
path= r"/home/cb51neqa/projects/itp/exp_data/2021-12-20/5µA/AF647_10ng_l"
path = path.replace("\\","/")
file = "005"
name = path + "/" + file
inname = "{}.nd2".format(name)

with ND2Reader(inname) as rawimages:
    rawimages.bundle_axes = 'yx' # defines which axes will be present in a single frame
    rawimages.iter_axes = 't' # defines which axes will be the index axis; z-axis is the time axis
    
    # determine metadata of the images
    height = rawimages.metadata["height"]
    width = rawimages.metadata["width"]
    nframes = rawimages.metadata["num_frames"]
    print(height, width)
    # Y x X x N
    data = np.zeros((height, width, nframes))
    
    # load image data into data array
    for frame in np.arange(0, len(rawimages), 1):
        data[:,:,frame] = rawimages[frame]
        
print(data.shape)        

# +
fstrow = int(height / 2 - 27)
lstrow = int(height / 2 + 27)
data = data[fstrow:lstrow,:,:]

print (data.shape)

# +
back = data[:,:,-11:nframes-1]
back = np.average(back, axis=2)

# subtract background fluorescence
data2 = np.zeros(data.shape)
for i in np.arange(0, nframes, 1):
    data2[:,:,i] = data[:,:,i] - back
# -

avgdata = np.average(data2, axis=0)
avgdata = (avgdata-np.mean(avgdata))/np.std(avgdata)
print(avgdata.shape)

# +
x = np.linspace(1,avgdata.shape[0],avgdata.shape[0])#*1.6

first = 100
last = 200

pltdata1 = avgdata[:,first]

ast1 = np.max(pltdata1)
cst1 = np.argmax(pltdata1)#*1.6
wst1 = 1
alphast1 = 1
offst1 = np.average(pltdata1)
popt1, pcov1 = curve_fit(skewed,x,pltdata1,p0=[ast1,cst1,wst1,alphast1,offst1])
fit1 = skewed(x, *popt1)

pltdata2 = avgdata[:,last]

ast2 = np.max(pltdata2)
cst2 = np.argmax(pltdata2)#*1.6
wst2 = 1
alphast2 = 1
offst2 = np.average(pltdata2)
popt2, pcov1 = curve_fit(skewed,x,pltdata2,p0=[ast2,cst2,wst2,alphast2,offst2])
fit2 = skewed(x, *popt2)
# -

plt.plot(x,pltdata1-popt1[4],x,fit1-popt1[4])
print(popt1)

plt.plot(x,pltdata2-popt2[4],x,fit2-popt2[4])
print(popt2)

# +
step = 46
fps = 46

vel = np.zeros(last-step-first+1)

for i in np.arange(first,last-step+1,1):
    pltdata1 = avgdata[:,i]

    ast1 = np.max(pltdata1)
    cst1 = np.argmax(pltdata1)#*1.6
    wst1 = 1
    alphast1 = 1
    offst1 = np.average(pltdata1)
    popt1, pcov1 = curve_fit(skewed,x,pltdata1,p0=[ast1,cst1,wst1,alphast1,offst1])
    fit1 = skewed(x, *popt1)

    pltdata2 = avgdata[:,i+step]

    ast2 = np.max(pltdata2)
    cst2 = np.argmax(pltdata2)#*1.6
    wst2 = 1
    alphast2 = 1
    offst2 = np.average(pltdata2)
    popt2, pcov1 = curve_fit(skewed,x,pltdata2,p0=[ast2,cst2,wst2,alphast2,offst2])
    fit2 = skewed(x, *popt2)
    
    x1 = np.argmax(fit1)#*1.6
    x2 = np.argmax(fit2)#*1.6
    
    
    vel[i-first] = ((x2-x1)) / (step/fps)
    
vel_avg=np.average(vel)
vel_std=np.std(vel) 
# -

print(vel_avg, vel_std)

# +
width = np.zeros(last-first+1)

for i in np.arange(first,last+1,1):
    sampdata = avgdata[:,i]

    ast = np.max(sampdata)
    cst = np.argmax(sampdata) #*1.6
    wst = 1
    alphast = 1
    offst = np.average(sampdata)
    popt, pcov = curve_fit(skewed,x,sampdata,p0=[ast,cst,wst, alphast, offst])
    width[i-first] = popt[2]
    #print(i)
    
width = width[width < 6]
print(np.mean(width), np.std(width))    

# +
fr = np.arange(first,last+1,1) - first

fig = plt.figure(figsize=(7.5,5))
plt.plot(width)
plt.title(file)
plt.xlabel('$frame$')
plt.ylabel('$spread$')
#plt.savefig(file + "png")

# +
m,b = np.polyfit(fr,width,1)

fig = plt.figure(figsize=(7.5,5))
plt.plot(fr,width,fr,m*fr+b)
plt.title(file)
plt.xlabel('$frame$')
plt.ylabel('$spread$')
#plt.savefig(file + "_fit.png")

print(m)

# +
#bins1 = np.linspace(6,24,37)
bins1 = np.linspace(3,15,25)

fig = plt.figure(figsize=(7.5,5))
plt.hist(width, bins=bins1)
plt.title(file)
plt.xlabel('$spread$')
plt.ylabel('$probablity$')
#plt.savefig(file + "_hist.png")

print(bins1)

# +
pltdata1 = avgdata[:,168]

ast1 = np.max(pltdata1)
cst1 = np.argmax(pltdata1)#*1.6
wst1 = 1
alphast1 = 1
offst1 = np.average(pltdata1)
popt1, pcov1 = curve_fit(skewed,x,pltdata1,p0=[ast1,cst1,wst1,alphast1,offst1])
fit1 = skewed(x, *popt1)

pltdata2 = avgdata[:,214]

ast2 = np.max(pltdata2)
cst2 = np.argmax(pltdata2)#*1.6
wst2 = 1
alphast2 = 1
offst2 = np.average(pltdata2)
popt2, pcov1 = curve_fit(skewed,x,pltdata2,p0=[ast2,cst2,wst2,alphast2,offst2])
fit2 = skewed(x, *popt2)
# -

plt.plot(x,pltdata1-popt1[4],x,fit1-popt1[4])
print(popt1)

plt.plot(x,pltdata2-popt2[4],x,fit2-popt2[4])
print(popt2)

# +
c_data = "dimgray"#"#BBBBBB"#black"
c_hist = c_data

fig = plt.figure(figsize=(4.5, 3.5))

ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
ax2 = plt.subplot2grid((2,2), (1,0))
ax3 = plt.subplot2grid((2,2), (1,1))

#plt.subplots_adjust(hspace=0.4)
#plt.subplots_adjust(wspace=0.25)

ax1.set_title(r"A: Intensity at 10 \si{\nano\gram\per\liter} for two different frames", loc="left")

#ax1.plot([(np.argmax(fit1)+1)*1.6, (np.argmax(fit1)+1)*1.6], [np.min(pltdata2)-np.absolute(np.min(pltdata2)*0.1), np.max(pltdata2-popt2[4])+np.absolute(np.min(pltdata2-popt2[4])*0.1)],'--k')
#x1.plot([(np.argmax(fit2)+1)*1.6, (np.argmax(fit2)+1)*1.6], [np.min(pltdata2)-np.absolute(np.min(pltdata2)*0.1), np.max(pltdata2-popt2[4])+np.absolute(np.min(pltdata2-popt2[4])*0.1)],'--k')
ax1.plot([(np.argmax(fit1)+1), (np.argmax(fit1)+1)], 
            [np.min(pltdata2)-np.absolute(np.min(pltdata2)*0.1), np.max(pltdata2-popt2[4])+np.absolute(np.min(pltdata2-popt2[4])*0.1)],'--k')
ax1.plot([(np.argmax(fit2)+1), (np.argmax(fit2)+1)], [np.min(pltdata2)-np.absolute(np.min(pltdata2)*0.1), np.max(pltdata2-popt2[4])+np.absolute(np.min(pltdata2-popt2[4])*0.1)],'--k')

ax1.plot(x,fit1-popt1[4], c="#EE6677")
ax1.plot(x,fit2-popt2[4])
ax1.plot(x,pltdata1-popt1[4],alpha=0.6)
ax1.plot(x,pltdata2-popt2[4],alpha=0.6)


ax1.set_xlabel('Length (\si{\micro\meter})')
ax1.set_ylabel('Intensity (-)')

ax1.annotate("$t$", (200,2.5), (150,3), arrowprops=dict(arrowstyle="->"))
ax1.annotate("$t+\Delta t$", (325,1.5), (375,2), arrowprops=dict(arrowstyle="->"))
ax1.annotate(r"", (np.argmax(fit1)+1,5), (np.argmax(fit2)+1,5), arrowprops=dict(arrowstyle="<->"))
ax1.annotate(r"$\Delta x_\text{max}$", (np.argmax(fit1)+1,5.3), (18,0), textcoords="offset points")
#ax1.annotate(r"$v_{ITP} = \frac{x_{max,t+\Delta t} - x_{max,t}}{\Delta t}$", (340,4))
ax1.annotate(r"$v_{ITP} = \frac{\Delta x_\text{max}}{\Delta t}$", (400,4.5))

###

ax2.set_title("B: Sample spread along channel length", loc="left")
ax2.plot(fr,width,fr,m*fr+b)
ax2.set_xlabel('Frame no.')
ax2.set_ylabel('Spread (\si{\pixel})')

##


_, binedges, _ = ax3.hist(width, bins=50, density=True, alpha=0.7, color=c_data)
bins = (binedges[1:]+binedges[:-1])/2
p1, p2 = stats.norm.fit(width)
mean, var = stats.norm.stats(p1, p2)
norm = stats.norm.pdf(bins, p1, p2)
ax3.plot(bins, norm, color="#EE6677", label=r"Normal($\mu\approx{:.1f},\sigma\approx{:.1f}$".format(p1, p2))
ax3.set_xlabel('Spread (\si{\pixel})')
#ax3.set_ylabel('Probability')
ax3.tick_params(labelleft = False, left=False)
ax3.tick_params(axis='y', which='both', labelleft=False, left=False)
ax3.spines['left'].set_visible(False)
#ax3.legend()

fig.tight_layout()
fig.savefig("EstimationROPE.pdf")
# -






