#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:47:00 2020

@author: keegan
"""

#case 2, use 10 GeV muon energy to find Beta and fit a more accurate landau distribution

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
E = 10**4 * (spc.eV * 10**6) # [J] (10 GeV for case 2)
m_mu = 105.7 * (spc.eV * 10**6) # [J c^(-2)]
n = 100
start = 2 
end = 5

#---------------------
# Error tracking
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



# --------------------------------- Landau Density -----------------------------------------



def ACoefficients_Arxiv():
    
    #https://arxiv.org/pdf/hep-ph/0305310.pdf
    
    start = time()
    
    A = []
    
    for k in range(LandauOrder):
        
        A_k = [special.gamma(k+1)]
        
        for j in range(1,k+1):
            A_k_j = 0
            
            for p in range(j):
                A_k_j += special.comb(j-1,p) * A_k[p] * special.polygamma(j-1-p,k+1)
                
            A_k.append(A_k_j)
    
        A.append(A_k)
    
    print('it took %s s to calculate the A_rk coefficients to order k = %s'%(round(time()-start,3),LandauOrder))

    return A


def GammaIncomplete(n, z): 
    # see https://mathworld.wolfram.com/IncompleteGammaFunction.html for more info on the expansion and definition
    
    Gamma = 0
    
    for i in range(n):
        Gamma += z**i / special.factorial(i)
        
    Gamma *= special.factorial(n-1) * np.exp(-z)

    return Gamma



def LandauDensity(C, Shift, Squeeze, Scale): 
    
    Landau = 0 
    
    C = (C - Shift) / Squeeze
    
    for k in range(LandauOrder):
        Temp = 0
        
        for r in range(k+1):
            
            Temp += (-1)**r * special.comb(k,r) * A[k][k-r] * \
                    np.imag((np.log(C - np.complex(0,1)*np.pi))**r / (C - np.complex(0,1)*np.pi)**(k+1))
        
        Landau += (-1)**k * Temp / (np.pi * special.factorial(k))

    
    Landau *= Scale
    
    return Landau 


def LandauDensity_Normal(C, Shift, Scale): 
    
    Landau = 0 
    
    C = (C - Shift) / Scale
    
    for k in range(LandauOrder):
        Temp = 0
        
        for r in range(k+1):
            
            Temp += (-1)**r * special.comb(k,r) * A[k][k-r] * \
                    np.imag((np.log(C - np.complex(0,1)*np.pi))**r / (C - np.complex(0,1)*np.pi)**(k+1))
        
        Landau += (-1)**k * Temp / (np.pi * special.factorial(k))

    
    Landau /= Scale
    
    return Landau 



def Gauss(x, a, b, c):
    
    Dist = c*np.exp(-(x - a)**2 / b**2)
    
    return Dist



def Moyal(x, loc, squeeze, scale):
    
    y = (x - loc) / squeeze
    
    Dist = np.exp(-(y + np.exp(-y)) / 2) / (np.sqrt(np.pi*2) * squeeze)
    
    Dist *= scale
    
    return Dist


def LandauFunction(Del,MatParam,I,scale):
    
    global E,m_mu
    
    x = BarHight / 100 # [m]
    
    c = spc.c
    beta = np.sqrt(1 - m_mu**2 / E**2)
    v = beta * c
    
    xi = (2*np.pi * spc.Avogadro * spc.e**4 * MatParam * x) / (spc.m_e * v**2)
    
    Lambda = Del / xi - np.log((2*spc.m_e * c**2 * beta**2 * xi) / ((1-beta**2)*I**2)) - 1 + beta**2 + np.euler_gamma
    
    f_L = LandauDensity(Lambda,0,1,1) / xi
    
    f_L *= scale
    
    return f_L
    
    
    
# --------------------------------- Angular Analysis -----------------------------------------



def ReadCases():
    RowData = []
    
    for i in range(2,5,2):
        RowData.append(ReadRowDataFileFastest('/Users/keegan/Downloads/Energy deposition in the bars/Case%s/RowData.out'%(2),[0,0])['BarsReadout'])
    
    #RowData.append(ReadRowDataFileFastest(input('input Path/to/file_name: \t'),[0,0])['BarsReadout'])
    
    Energies11 = []
    Energies12 = []
    
    for readout in RowData:
        Temp11 = []
        Temp12 = []
        
        for i in range(len(readout[1])):
            Temp11.append(readout[1][i][0])
            Temp12.append(readout[1][i][1])
        
        Energies11.append(Temp11)
        Energies12.append(Temp12)
    
    Energies11 = np.array(Energies11)
    Energies12 = np.array(Energies12)
    
    return Energies11, Energies12



def PlotFit(Energies):
    
    def Chi(v,O,E):
       
        Out = np.sum((O-E)**2 / np.var(O)) 
        Out /= v
        
        return Out
    
    '''
    Psuedo:
        read row data
        sort energy readouts in each bar 
            put into a list / array
        use list / array to create a histogram
            
        fit landau to histogram 
            take each bin as a data point
    '''
    
    fig = plt.figure(figsize=(15,15))
    
    Histograms = []
    Bins = []
    
    Parameters = []
    Covariances = []
    ChiList = []
    FWAHM = []
    
    
    for i in range(len(Energies)):
        TempP = []
        TempC = []
        
        global n, start, end
        
        X = np.linspace(start,end,num=n)
        
        v = n - 3 # Degrees of freedom (for use with Chi())
        
        Hist, BinEdges = np.histogram(Energies[i],X)
        
        Hist = np.array(Hist,dtype='float64')
        BinEdges = np.array(BinEdges,dtype='float64')
        
        Histograms.append(Hist)
        Bins.append(BinEdges)
        
        BinEdges = BinEdges[:-1] + (end - start)/(2*n) # takes centre of bins as Energy Value
        
        ax = fig.add_subplot(2,1,i+1)
        ax.set_title('Case %s' % (2*(i+1)))
        ax.plot(BinEdges,Hist,marker='o',linewidth=0)
        ax.set_xlabel('Energy deposited in top layer of scintillators [MeV]')
        ax.set_ylabel('BinCounts')
        
        #X = np.linspace(np.amin(BinEdges),np.max(BinEdges),num=len(BinEdges))
        #X = np.array(X,dtype='float64')
    
        popt, pcov = curve_fit(LandauDensity, BinEdges, Hist, p0=(3,1,1000))
        
        TempP.append(popt)
        Covariances.append(pcov)
        Red = Chi(v,Hist,LandauDensity(X,popt[0],popt[1],popt[2])[:-1])
        TempC.append(Red)
        
        ax.plot(X,LandauDensity(X,popt[0],popt[1],popt[2]),label='Landau (%s)'%float('%.3g' % Red))
        
        
        popt, pcov = curve_fit(Moyal, BinEdges, Hist, p0=(3,1,100))
        
        TempP.append(popt)
        Covariances.append(pcov)
        Red = Chi(v,Hist,Moyal(X,popt[0],popt[1],popt[2])[:-1])
        TempC.append(Red)
        
        ax.plot(X,Moyal(X,popt[0],popt[1],popt[2]),label='Moyal (%s)'%float('%.3g' % Red))
        
        
        popt, pcov = curve_fit(Gauss, BinEdges, Hist, p0=(3,1,100))
        
        TempP.append(popt)
        Covariances.append(pcov)
        Red = Chi(v,Hist,Gauss(X,popt[0],popt[1],popt[2])[:-1])
        TempC.append(Red)
        
        ax.plot(X,Gauss(X,popt[0],popt[1],popt[2]),label='Gauss (%s)'%float('%.3g' % Red))
        
        
        '''
        popt, pcov = curve_fit(Convolved, BinEdges, Hist, p0=(4,1,10))
        
        TempP.append(popt)
        Covariances.append(pcov)
        
        Red = Chi(v,Hist,Convolved(X,popt[0],popt[1],popt[2])[:-1])
        TempC.append(Red)
        
        ax.plot(X,Convolved(X,popt[0],popt[1],popt[2]),label='Convolved (%s)'%float('%.3g' % Red))
        '''
        
        
        up = np.where(Hist>np.max(Hist)/2)
        FWAHM.append(BinEdges[up[0][-1]]-BinEdges[up[0][0]])
        
        ChiList.append(TempC)
        Parameters.append(TempP)
        
        ax.legend()
            
    plt.show()

    return Histograms, Bins, Parameters, Covariances, ChiList, FWAHM



A = ACoefficients_Arxiv()

Energies11, Energies12 = ReadCases()

#3.89677 - 4
#1.15149 - 1 
#0.848819 - 1 


#PlotFit(Energies11) # make p0 list for each of the 10 plots to improve fits in each 
#PlotFit(Energies12)
Hist, Bins, Param, Cov, Chi, FWAHM = PlotFit(Energies11+Energies12)
