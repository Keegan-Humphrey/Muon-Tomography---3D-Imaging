#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:30:41 2020

@author: keegan
"""
import numpy as np
import datetime
from scipy import special, misc
import math
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.optimize import curve_fit
import scipy.constants as spc

print('Hello Viewer!')

#---------------------------------- Global Variables -----------------------------------------


#---------------------
# Detector Information
#---------------------
TriggerSize = 41.6                                              #[cm]
BarWidth = 3.2                                                  #[cm]
BarHight = 1.7                                                  #[cm]
NumOfBars = 27
PlanesSeperation = 25                                           #[cm]
TriggerWidth = 1                                                #[cm]

Seperation = 25                                                 #[cm]


#---------------------
# Landau
#---------------------
LandauOrder = 15
m_mu = 105.7 # [MeV]


#---------------------
# Plotting
#---------------------
n = 500
 

#---------------------
# Error Tracking
#---------------------
NoSignal = 0
OneSignal = 0


# --------------------------------- Translation of Gilad's Code -----------------------------------------



def ReadRowDataFileFastest(FileName,DetectorPos): #must use 'path/Filename' (including quoatations) in function call 
    
    #--------------------------------------------------------------------
    # Reads RowData.out files and produces a Dictionary with the 
    # information contained in the file
    #--------------------------------------------------------------------
    
    begin_time = datetime.datetime.now()
    
    G = np.loadtxt(FileName, dtype = str, delimiter = '~') #Delimiter '~' is chosen in order to avoid default whitespace delimiter
    Gsize = len(G)
    
    EventWordInd = []
    n = 0
    
    for i in range(Gsize):    
        if G[i] == '*Event*':
            n += 1
            EventWordInd.append(i)
    
    RowData = {'FileName':FileName,
               'NumberOfEvents':n,
               'DateAndTime':[],
               'DetectorPos':DetectorPos, #Coordinates (in x,y plane) of the Detector
               'BarsReadout': [[],[]],
               'UpperTrigPos':[[],[]],
               'LowerTrigPos':[[],[]]} #Trig Positions are left empty, can be filled if needed
    
    for i in range(n):
        RowData['DateAndTime'].append(G[EventWordInd[i]+2])   
       
    for i in range(n-1): # -1 to avoid error from length of last event 
        Bar = []
        Length = []
        
        for l in range(12):
            if 12 <= len(G[EventWordInd[i]+l]) <= 13: #Bounds depend delicately on precision of path lengths
                Bar.append(G[EventWordInd[i]+l][0:3])
                Length.append(G[EventWordInd[i]+l][4:])
    
        BarFloat = np.float_(Bar).tolist() #Converts string elements to float
        LengthFloat = np.float_(Length).tolist()
        
        RowData['BarsReadout'][0].append(BarFloat)
        RowData['BarsReadout'][1].append(LengthFloat)
        
    print('There were ', n,' events in ', FileName,', the simulation ran between ', \
          RowData['DateAndTime'][0],' - ',RowData['DateAndTime'][n-1],'.')
    
    tictoc = datetime.datetime.now() - begin_time
    print('It took ', tictoc,' to read the file.') #',FileName)

    return RowData


def CalcLocalPos(Bar,Length): #Arguments are BarsReadout elements from one event
    # LocalPos == [LocalX or LocalY, LocalZ]
    
    #--------------------------------------------------------------------
    # Determines position of the muon through each layer of scintillators
    # as well as outputs list of which bars produced the signal for each
    # event
    #--------------------------------------------------------------------
    
    LengthLocal = [[],[],[],[]]
    BarLocal = [[],[],[],[]]
    LocalPos = [[],[],[],[]]
    
    for n in range(len(Bar)): #Sorts Bar and Length data into nested lists corresponding to each layer on the detector
         LengthLocal[math.floor(Bar[n]/100)-1].append(Length[n])
         BarLocal[math.floor(Bar[n]/100)-1].append(Bar[n])
         
    a = np.sqrt(BarHight**2+(BarWidth/2)**2)
    Alpha = np.arctan(2*BarHight/BarWidth)
    
    for i in range(len(LengthLocal)):
        NumOfBarsLocal = len(LengthLocal[i])
        
        if NumOfBarsLocal == 0: # No signal, return error
            X,Z = -9999,-9999
            global NoSignal 
            NoSignal += 1
        
        elif NumOfBarsLocal == 1: 
            
            #X,Z = -9999,-9999
            #X,Z = BarWidth/2, BarHight/2
            
            global OneSignal 
            OneSignal += 1
            
            if BarLocal[i][0]%2 == 0: #The first bar's vertex is facing down
                X,Z = 0, 0
            
            else:
                X,Z = 0, BarHight
            
            #Takes tip instead of middle of bar
            
            
        elif NumOfBarsLocal >= 2:
            Readout = []
            mxind = LengthLocal[i].index(max(LengthLocal[i]))
            
            if mxind == 0: #The first bar has the max readout so we take the first and second bars
                Readout = [LengthLocal[i][mxind],LengthLocal[i][mxind+1]]
            
            elif mxind == len(LengthLocal[i])-1: #The last bar has the max readout so we take the last bar and the one before
                Readout = [LengthLocal[i][mxind],LengthLocal[i][mxind-1]]
            
            else: #The max readout is somewhere at the middle, so we take this bar and it's highest neighbor
                Readout = np.amax([[LengthLocal[i][mxind],LengthLocal[i][mxind+1]],\
                                   [LengthLocal[i][mxind],LengthLocal[i][mxind-1]]],axis = 0) 

            if BarLocal[i][0]%2 == 0: #The first bar's vertex is facing down
                X = BarWidth/2 - (a*Readout[0]/(Readout[0]+Readout[1]))*math.cos(Alpha)
                Z = BarHight/2 - (a*Readout[0]/(Readout[0]+Readout[1]))*math.sin(Alpha)
            
            else: #The first bar's vertex is facing up
                X = -BarWidth/2 + (a*Readout[1]/(Readout[0]+Readout[1]))*math.cos(Alpha)
                Z = BarHight/2 - (a*Readout[1]/(Readout[0]+Readout[1]))*math.sin(Alpha)
            
        LocalPos[i] = [X,Z] #[X/2 + np.random.random()*X, Z/2 + np.random.random()*Z] #randomize 50% (increases computing time)
            
    return LocalPos, BarLocal 



def CalcAbsPos(LocalPos,BarLocal, Seperation):
    #AbsPos[i] == [AbsX or AbsY, AbsZ] 
    
    #--------------------------------------------------------------------
    # Deterimines position relative to centre of the detector layer, 
    # and height measured from the bottom of the detector
    #--------------------------------------------------------------------
    
    AbsPos = [[0,0],[0,0],[0,0],[0,0]]
    
    if [-9999,-9999] in LocalPos:
        return -9999
        
    else:    
        for l in range(len(LocalPos)):  
            FirstBarIndex = BarLocal[l][0]
            i = math.floor(FirstBarIndex/100)
            
            if i == 1: #XUp
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + Seperation + 3.5 * BarHight
                FirstBarIndex = FirstBarIndex - 100
           
            if i == 2: #YUp
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + Seperation + 2.5 * BarHight
                FirstBarIndex = FirstBarIndex - 200
            
            if i == 3: #XDown
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + 1.5 * BarHight
                FirstBarIndex = FirstBarIndex - 300
            
            if i == 4: #YDown
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + 0.5 * BarHight
                FirstBarIndex = FirstBarIndex - 400
            
            
            if FirstBarIndex%2 == 0 : #The first bar's vertex is facing down
                AbsPos[l][0] = LocalPos[l][0] - (NumOfBars / 4 - 0.25) * BarWidth + BarWidth / 2 * FirstBarIndex
                
            else: #The first bar's vertex is facing up
                AbsPos[l][0] = LocalPos[l][0] - (NumOfBars / 4 - 0.25) * BarWidth + BarWidth / 2 * (FirstBarIndex + 1) 
                
        return AbsPos 



def CalcAngle(Bar, Length, DetectorPos, Seperation): #( <RowData>['BarsReadout'][0][i], <RowData>['BarsReadout'][1][i], ...)
    
    #--------------------------------------------------------------------
    # Determines polar Angle of muon trajectory 
    #--------------------------------------------------------------------
    
    [LocalPos,BarLocal] = CalcLocalPos(Bar,Length)
    
    AbsPos = CalcAbsPos(LocalPos,BarLocal, Seperation)
    
    if AbsPos == -9999:
        Theta = None
        Thickness = None
    
    else:    
        X = AbsPos[0][0] - AbsPos[2][0]
        Y = AbsPos[1][0] - AbsPos[3][0]
        
        Z = 2 * TriggerWidth + 4 * BarHight + Seperation
        
        R = np.sqrt(X**2 + Y**2)
        
        Theta = np.arctan(R/Z)
        
        Thickness = BarHight / np.cos(Theta)
        
    return Theta, Thickness



# --------------------------------- Empirical Muon Distribution -----------------------------------------

# https://arxiv.org/abs/hep-ph/0604145v2
# Reyna 2008
# valid where muon energies ~ O(10 GeV) are important
# distribution of muon energies and angles at sea level in this domain

def I_v(p):
    
    c = [0.00253,0.2455,1.288,-0.2555,0.0209]
    
    I = c[0] * p**(-(c[1] + c[2] * np.log10(p) + c[3] * np.log10(p)**2 + c[4] * np.log10(p)**3))
    
    return I


def Intensity(Energ,theta):
    # Expects [Energ] = [MeV]
    
    global m_mu
    
    p = np.sqrt(Energ**2 - (m_mu * 10**(-3))**2)  # [J / c]
    
    I = np.cos(theta)**3 * I_v(p * np.cos(theta))

    return I 


def Intensity_List(Energ,theta):
    # Expects [Energ] = [MeV]
    
    global m_mu
    
    for en in Energ:
        for ang in theta:
            p = np.sqrt(en**2 - (m_mu * 10**(-3))**2)  # [J / c]
        
            I = np.cos(ang)**3 * I_v(p * np.cos(ang))
            
    return I 


# --------------------------------- Angular Analysis -----------------------------------------



def ReadSort():
    
    #RowData = ReadRowDataFileFastest('/Users/keegan/Downloads/Sky_my_run/RowData.out',[0,0])
    #RowData = ReadRowDataFileFastest('/Users/keegan/Downloads/More simulations/Detector at 20_0/Seperation 25/Sky/RowData.out',[0,0])
    #RowData = ReadRowDataFileFastest('/Users/keegan/Downloads/More simulations/Detector at 20_0/Seperation 25/Real/RowData.out',[0,0])
    
    RowData = ReadRowDataFileFastest(input('input Path/to/file_name: \t'),[0,0])
    
    Energies11 = []
    Energies12 = []
    
    Theta = []
    Thickness = []
    
    RejectedEvents = 0
    
    for i in range(RowData['NumberOfEvents']-1):
        
        Angle, Thick = CalcAngle(RowData['BarsReadout'][0][i], RowData['BarsReadout'][1][i], \
                                                  RowData['DetectorPos'], Seperation)
        
        if Thick != None:
            Theta.append(Angle)
            Thickness.append(Thick)
            
            Energies11.append(RowData['BarsReadout'][1][i][0])
            Energies12.append(RowData['BarsReadout'][1][i][1])
        
        else:
            RejectedEvents += 1
            
    Energies11 = np.array(Energies11)
    Energies12 = np.array(Energies12)
    
    Energies = Energies11 + Energies12
    
    Thickness = np.array(Thickness)
    
    print('There were %s rejected events'%RejectedEvents)
    
    return Energies, Theta, Thickness, RowData



# --------------------------------- Call Functions -----------------------------------------


'''
A = ACoefficients_Arxiv()
'''
Energies, Theta, Thickness, RowData = ReadSort()


# --------------------------------- Plot Histogram -----------------------------------------



   
EEdges = np.linspace(np.amin(Energies),np.max(Energies),num=n)
TEdges = np.linspace(np.amin(Theta),np.max(Theta),num=n)

H, EEdges, TEdges = np.histogram2d(Energies, Theta, bins=(EEdges,TEdges))

EHist, EBinEdges = np.histogram(Energies,EEdges)
THist, TBinEdges = np.histogram(Theta,TEdges)

Norm_H = np.sum(H)
Prob_H = H * 100 / Norm_H  # Normalize and convert to percent probability

fig = plt.figure(dpi = 180) #, figsize=(15,15))
ax = fig.add_subplot(111)
plt.imshow(Prob_H,origin='low') #,extent=(EEdges[0],EEdges[-1],TEdges[0],TEdges[-1]))
ax.set_title('Probability Distribution For Simulated Detector Data')
ax.set_ylabel('Energy Deposited in Top Layer [MeV]') #.set_visible(True)
ax.set_xlabel('Angle [rad]') #.set_visible(True)

y_label_list = [str(round(EEdges[0],3)), str(round(EEdges[100],3)), str(round(EEdges[200],3)), str(round(EEdges[300],3)), str(round(EEdges[400],3))]
x_label_list = [str(round(TEdges[0],3)), str(round(TEdges[100],3)), str(round(TEdges[200],3)), str(round(TEdges[300],3)), str(round(TEdges[400],3))]

ax.set_yticks([0,100,200,300,400])
ax.set_xticks([0,100,200,300,400])

ax.set_yticklabels(y_label_list)
ax.set_xticklabels(x_label_list)

plt.colorbar(orientation='vertical',label='% Probability')
plt.show()



# --------------------------------- Plot Simulation Distributions -----------------------------------------



EEdges2 = np.linspace(np.amin(Energies),np.max(Energies),num=int(n/20))
TEdges2 = np.linspace(np.amin(Theta),np.max(Theta),num=int(n/20))

H, EEdges2, TEdges2 = np.histogram2d(Energies, Theta, bins=(EEdges2,TEdges2))

EHist, EBinEdges2 = np.histogram(Energies,EEdges2)
THist, TBinEdges2 = np.histogram(Theta,TEdges2)

fig = plt.figure(dpi = 180) #, figsize=(15,15))
ax = fig.add_subplot(111)

plt.plot(EEdges2[:-1],EHist)
ax.set_title('Energy Distribution - Simulated')
ax.set_ylabel('Bin Counts') #.set_visible(True)
ax.set_xlabel('Energy Deposited [MeV]') #.set_visible(True)

X = np.linspace(np.amin(EBinEdges2),np.max(EBinEdges2),num=len(EBinEdges2))
X = np.array(X,dtype='float64')

up = np.where(EHist>np.max(EHist)/2)
FWAHM = (EBinEdges[up[0][-1]]-EBinEdges[up[0][0]])
        
'''
#popt, pcov = curve_fit(LandauDensity, EBinEdges[:-1], EHist, p0=(20,1,100000))
#ax.plot(X,LandauDensity(X,popt[0],popt[1],popt[2]),label='landau')

#popt, pcov = curve_fit(Convolved, EBinEdges[:-1], EHist, p0=(20,1,1000))
#ax.plot(X,Convolved(X,popt[0],popt[1],popt[2]),label='convolved')

'''
fig = plt.figure(dpi = 180) #,figsize=(15,15))
ax = fig.add_subplot(111)

plt.plot(TEdges2[:-1],THist)  
ax.set_title('Angle Distribution - Simulated')
ax.set_ylabel('Bin Counts') #.set_visible(True)
ax.set_xlabel('Angle [rad]') #.set_visible(True)

plt.show()
'''   

'''
# --------------------------------- Plot Empirical Distributions -----------------------------------------



EEdges =  10**np.arange(-1,2,0.01)+0.0058 # [GeV]
#TEdges = np.linspace(0,1.01721,num=n/5) # [rad]

TMesh, EMesh = np.meshgrid(TEdges, EEdges)

Emp_H = Intensity(EMesh,TMesh)

Norm = np.sum(Emp_H)
Emp_Norm = Emp_H * 100 / Norm # Normalize and convert to percent probability

#Emp_Norm *= 1000

fig = plt.figure(dpi = 180) #, figsize=(15,15))
ax = fig.add_subplot(111)
plt.imshow(Emp_Norm,origin='high') #,extent=(EEdges[0],EEdges[-1],TEdges[0],TEdges[-1]))
ax.set_title('Probability Distribution for CR Muons at Sea Level')
ax.set_ylabel('Energy of Muons [GeV]') #.set_visible(True)
ax.set_xlabel('Incident Zenith Angle [rad]') #.set_visible(True)

y_label_list = ['0.1','1','10','100','10^5',]
#y_label_list = [str(round(EEdges[0],3)), str(round(EEdges[100],3)), str(round(EEdges[200],3)), str(round(EEdges[299],3))] #, str(round(EEdges[400],3)), str(round(EEdges[499],3))]
x_label_list = [str(round(TEdges[0],3)), str(round(TEdges[100],3)), str(round(TEdges[200],3)), str(round(TEdges[299],3)), str(round(TEdges[400],3)), str(round(TEdges[499],3))]

ax.set_yticks([0,100,200,299]) #,400,499])
ax.set_xticks([0,100,200,300,400,499])

ax.set_yticklabels(y_label_list)
ax.set_xticklabels(x_label_list)

plt.colorbar(orientation='vertical',label = '% Probability')
plt.show()


##Change to a log plot of energies to include distribution of absorbed energies



# --------------------------------- Plot Both Log-Log -----------------------------------------

EEdges = 10**np.arange(-1,2,0.01)+0.0058 # [GeV]
TEdges = np.linspace(0,1.01721,num=n) # [rad] field of view of detector at 25 cm

TMesh, EMesh = np.meshgrid(TEdges[:-1], EEdges)

Emp_H = Intensity(EMesh,TMesh)

Norm = np.sum(Emp_H)
Emp_Norm = Emp_H * 100 / Norm # Normalize and convert to percent probability

Emp_Norm *= 100

LogBins = 10**np.arange(1,5.01,0.01)
TEdges = np.linspace(np.amin(Theta),np.max(Theta),num=n)

H_log,LogBins,TEdges = np.histogram2d(Energies, Theta, bins=(LogBins,TEdges))

Norm = np.sum(H_log)
log_Norm = H_log * 100 / Norm

log_Norm[100:] += Emp_Norm

fig = plt.figure(dpi = 180) #, figsize=(15,15))
ax = fig.add_subplot(111)
plt.imshow(log_Norm,origin='high') #,extent=(EEdges[0],EEdges[-1],TEdges[0],TEdges[-1]))
ax.set_title('Sea Level Muon Distribution vs Deposited Distribution')
ax.set_ylabel('Energy [MeV]') #.set_visible(True)
ax.set_xlabel('Incident Zenith Angle [rad]') #.set_visible(True)

y_label_list = ['10','10^2','10^3','10^4','10^5',]
#y_label_list = [str(round(LogBins[0],3)), str(round(LogBins[100],3)), str(round(LogBins[200],3)), str(round(LogBins[300],3)), str(round(LogBins[400],3))] #, str(round(LogBins[499],3))]
x_label_list = [str(round(TEdges[0],3)), str(round(TEdges[100],3)), str(round(TEdges[200],3)), str(round(TEdges[300],3)), str(round(TEdges[400],3)), str(round(TEdges[499],3))]

ax.set_yticks([0,100,200,300,400]) #,499])
ax.set_xticks([0,100,200,300,400,499])

ax.set_yticklabels(y_label_list)
ax.set_xticklabels(x_label_list)

plt.colorbar(orientation='vertical',label = '% Probability (lower disribution only)')
plt.show()



