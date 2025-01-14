# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:11:48 2024

@author: Gaelen
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


import matplotlib.pyplot as plt



#%%
def power_speed(v,cd,a,wc,wb,g,vhw=0,rho=1.1995,crr=0.004,lossdt=2):
    '''
    power given speed
    v = speed (m/s)
    cd = drag coefficient, cadex with disc is 0.7185 --> https://zwifterbikes.web.app/whatif
    a = frontal area
    rho = air density (1.1995 kg/m3 --> https://gribble.org/cycling/air_density.html)
    vhw = headwind velocity (m/s)
    wc = weight cyclist (kg)
    wb = weight bike (kg) --> cadex w disc is 9.221 kg --> https://zwifterbikes.web.app/whatif
    g = grade (rise/run*100)
    crr = rolling resistance coefficient (0.004 on pavement --> https://zwiftinsider.com/crr/)
    lossdt = drivertrain losses (/100)
    '''
    dtl_term = (1-lossdt/100)**-1
    w = wc+wb
    g_term = 9.8067 * w * (np.sin(np.arctan(g/100))+crr*np.cos(np.arctan(g/100)))
    print(g_term)
    a_term = 0.5*cd*a*rho*(v+vhw)**2
    print(a_term)
    
    return dtl_term * (g_term + a_term)*v


def frontal_area(height,weight):
    '''https://www.researchgate.net/publication/7906171_The_Science_of_Cycling_Factors_Affecting_Performance_Part_2
    height in m
    weight in kg
    '''
    return 0.0293*height**0.725*weight**0.425


def asymp_func(x,x0,y0,x1,y1):
    '''
    used to help model draft position base array
    x = array of positions to eval, then supply two values (0,1) and (7,0.55)
    '''
    f=1
    a = -1
    b = x1-x0
    c = (x1*x0)-(f*x0-f*x1)/(y0-y1)
    d = np.roots([a,b,c])[1]
    e = y0 - (1/(x0-d))
    res = 1/(x-d)+e
    return res

@dataclass
class Rider:
    name: str
    wt: float
    ht: float
    ftp: float
    wp: float
    
    def __post_init__(self):
        self.frontal_area = frontal_area(self.ht/100,self.wt)
    
    def calc_wbal(self):
        '''
        stolen from here --> https://forum.intervals.icu/t/using-w-balance/804/4
        '''
        df = self.wo_df.copy()
        df['Time'] = df['Time'].cumsum()
        tdf = pd.DataFrame(data={'Time':np.arange(0,df['Time'].max()+1,1)})
        df= pd.merge(df,tdf,how='right',on='Time')
        df['Power'] = df['Power'].bfill()
        wb = np.zeros(len(df))
        for i in range(len(wb)):
            if i == 0:
                wb[i] = self.wp
                continue
            t = df['Time'].iloc[i]
            deltaW = self.ftp - df['Power'].iloc[i]
            if deltaW > 0:
                w = wb[i-1] + deltaW * (self.wp - wb[i-1])/self.wp
            else:
                w = wb[i-1]+deltaW
            wb[i] = w
        df['Wbal'] = wb
        self.wbal = df

@dataclass
class Paceline:
    riders: list
    scaling_factor: float
    
    def __post_init__(self):
        assert all([isinstance(x,Rider) for x in self.riders])
        self.mean_ftp = np.mean([x.ftp for x in self.riders])
        self.target_power = self.mean_ftp*self.scaling_factor
        self.mean_wt = np.mean([x.wt for x in self.riders])
        self.mean_ht = np.mean([x.ht for x in self.riders])
        
    def calc_power_demands(self,base_array,height_factor,weight_factor,draft_factor):
        '''given certain scaling factors, calculates the estimated demands for each rider in each position
        scaling factors include:
            -base_array array of numbers [1.0,...0.6 ] etc. that provide a "base array" for estimating draft benefit as a function of position
                - asymp function can be helpful for playing with this
            -height/weight factors scale the power demands based on some combo of rider height/weight, aiming to roughly approximate the zwift voodo
            -draft factor scales the base_array even further, giving additional draft benefit to more slippery riders and penalizing the less aero riders.
            
            adds a pwr_df (dataframe) to the instance
        '''
        hf_array = (np.array([x.ht for x in self.riders])/self.mean_ht)**height_factor
        wf_array = (np.array([x.wt for x in self.riders])/self.mean_wt)**weight_factor
        f = hf_array * wf_array #represents "base scaling" based on height/weight --> i.e. best estimate of CdA
        for i,r in enumerate(self.riders):
            df = (f[i]*base_array)**draft_factor #additional scaling on each draft position based on CdA for that rider
            p = f[i]*self.target_power*df
            r.power_array = p
        df = pd.DataFrame(index=[x.name for x in pl.riders]
                          ,data=np.array([x.power_array for x in pl.riders])
                          ,columns=[i for i in range(len(pl.riders))]).round(0).reset_index().rename(columns={'index':'rider'})
        self.pwr_df = df       

    def build_wodf(self, dur, turn_times):
        '''with power demands calculated, given a total duration and an array of turn times (len = len(self.riders)
        calculates a "workout_df" for each rider in the paceline        
        '''
        for i,r in self.pwr_df.iterrows():
            #rearrange the power demands based on positioning
            c = r[list(range(0,len(self.pwr_df)))].to_list()
            pa = c[:i+1][::-1]+c[i+1:][::-1]          
            df = pd.DataFrame(data={'Time':turn_times,'Power':pa})
            nrnds = int(np.ceil(dur/df['Time'].sum()))
            df = pd.concat([df for x in range(nrnds)]).reset_index(drop=True)
            df['cs'] = df['Time'].cumsum()
            df = df[df.index <= df[df['cs']<dur].index.max() + 1].reset_index(drop=True)
            df['Time'].loc[-1] = dur-df['cs'].iloc[-2]
            df = df.drop(columns='cs')
            r = [x for x in self.riders if x.name == r['rider']][0]
            if hasattr(r,'wo_df'):
                r.wo_df = pd.concat([r.wo_df,df])
            else:
                r.wo_df = df

#step one --> define riders
r1 = Rider(name='AL',wt=60,ht=170,ftp=295,wp=20000)
r2 = Rider(name='GM',wt=64,ht=173,ftp=305,wp=20000)
r3 = Rider(name='DS',wt=83,ht=188,ftp=350,wp=20000)
# r3 = Rider(name='ER',wt=81,ht=184,ftp=330,wp=20000)
r4 = Rider(name='SG',wt=67,ht=179,ftp=290,wp=20000)
r5 = Rider(name='EK',wt=76,ht=183,ftp=315,wp=20000)
r6 = Rider(name='TR',wt=75,ht=180,ftp=335,wp=20000)

#%%
# r6 = Rider(name='DK',wt=76,ht=178,ftp=305,wp=20000)
# r7 = Rider(name='JW',wt=78,ht=185,ftp=305,wp=20000)
# r8 = Rider(name='LN',wt=77,ht=182,ftp=309,wp=20000)
# r9 = Rider(name='MB',wt=77.4,ht=180,ftp=301,wp=20000)

# r7 = Rider(name='JM',wt=74,ht=171,ftp=265,wp=20000)
# r8 = Rider(name='SP',wt=88.9,ht=179,ftp=287,wp=20000)
# r8 = Rider(name='SB',wt=67,ht=179,ftp=275,wp=20000)

#step two, define positions in paceline and scaling factor
pl = Paceline(riders=[r3,r1,r2,r6,r5,r4
                      ]
              ,scaling_factor=1.25)
print('Mean Power at Front:')
print(pl.target_power)

#step three, estimate power demands in the paceline
barray = asymp_func(x=np.arange(0,len(pl.riders),1),x0=0,y0=1.0,x1=4,y1=0.62)
#print('Draft Fractions by Position:')
pl.calc_power_demands(base_array=barray
                      ,height_factor=0.25 #higher the number, the more penalty being tall is
                      ,weight_factor=0.2 #higher then number, the more penalty being heavy is
                      ,draft_factor=0.9 #the higher the number, the more draft advantage is given to smaller riders 
                      )
# pl.calc_power_demands(base_array=barray
#                      ,height_factor=0.506 #higher the number, the more penalty being tall is
#                      ,weight_factor=0.271 #higher then number, the more penalty being heavy is
#                      ,draft_factor=0.7998 #the higher the number, the more draft advantage is given to smaller riders 
#                      )


turn_times = [45,45,45,45,30,30]
#step four, define turn times in the paceline and estimate duration to create a "workout_dataframe" for each rider
pl.build_wodf(dur=38*60,
              turn_times=turn_times
              )

#step 5, now that each rider has a "workout" based on the power demands from the paceline, calculate wbal over the duration of the event
fig, ax = plt.subplots()
for r in pl.riders:
    r.calc_wbal()
    ax.plot(r.wbal['Time']/60,r.wbal['Wbal'],label=r.name)
ax.legend()
ax.set_ylabel('Wbal')
ax.set_xlabel('Duration (min)')
ax.set_ylim(0,20000)
plt.grid()

print('Power Demands')
print(pl.pwr_df)
#%% printout of pull plan
pp_df = pl.pwr_df[['rider',0]].rename(columns={0:'Power (w)'})
pp_df['Turn (s)'] = turn_times
pp_df

# (pp_df['Turn (s)'].sum()+45+30+30)*47000/60/60

 #%%           
#"calibration" of height,weight and draft factors + base array
# needs to be cleaned up/more formalized but rough idea for now
# populate observations from riders e.g "in p3 i was averaging x today"

def calcRes(a):         
    r1 = Rider(name='GM',wt=64,ht=173,ftp=297,wp=25000)
    r2 = Rider(name='ER',wt=81,ht=184,ftp=330,wp=25000)
    r3 = Rider(name='SG',wt=68.2,ht=177,ftp=320,wp=25000)
    r4 = Rider(name='EK',wt=76,ht=183,ftp=306,wp=25000)
    r5 = Rider(name='DK',wt=76,ht=178,ftp=302,wp=25000)
    r6 = Rider(name='JW',wt=78,ht=185,ftp=294,wp=25000)
    r7 = Rider(name='MB',wt=77.4,ht=175,ftp=301,wp=25000)
    r8 = Rider(name='SP',wt=88.9,ht=179,ftp=287,wp=25000)
    
    pl = Paceline(riders=[r1,r2,r3,r4,r5,r6,r7,r8]
                  ,scaling_factor=1.25)
    
    height_factor = a[0]
    weight_factor = a[1]
    draft_factor = a[2]
    pl.calc_power_demands(base_array=asymp_func(x=np.arange(0,7,1),x0=0,y0=1.0,x1=4,y1=0.62)
                          ,height_factor=height_factor,weight_factor=weight_factor,draft_factor=draft_factor)
    #observations
    o1 = (r2.power_array[1] - 340)**2
    o2 = (r2.power_array[2] - 308)**2
    o3 = (r2.power_array[3] - 286)**2
    o4 = (r3.power_array[1] - 301)**2
    o5 = (r3.power_array[2] - 271)**2
    o6 = (r3.power_array[3] - 253)**2
    o7 = (r4.power_array[1] - 326)**2
    o8 = (r4.power_array[2] - 294)**2
    o9 = (r4.power_array[3] - 274)**2
    o10 = (r1.power_array[1] - 286)**2
    o11 = (r1.power_array[2] - 258)**2
    o12 = (r1.power_array[3] - 241)**2

    return np.sum([o1,o2,o3,o4,o5,o6,o8,o9,o10,o11,o12])

# r = calcRes(height_factor=1.05,weight_factor=0.1,draft_factor=0.75)

from scipy.optimize import minimize

ie = [0.2,0.3,0.75]
res = minimize(calcRes,x0=ie)
print(res.x)
#^^


    
    
