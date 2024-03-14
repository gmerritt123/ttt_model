# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:11:48 2024

@author: Gaelen
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


import matplotlib.pyplot as plt

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
        '''translates a "workout" {'Time':[x,y],'Power':[px,py]} into a dataframe given a total estimated duration.
        workout will repeat itself until duration is used up
        returns a dataframe
        '''
        
        for i,r in self.pwr_df.iterrows():
            #rearrange the power demands based on position
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

r1 = Rider(name='AL',wt=60,ht=170,ftp=300,wp=25000)
r2 = Rider(name='GM',wt=64,ht=173,ftp=297,wp=25000)
r3 = Rider(name='DS',wt=81,ht=188,ftp=335,wp=25000)
r4 = Rider(name='ER',wt=85,ht=184,ftp=322,wp=25000)
r5 = Rider(name='JW',wt=78,ht=185,ftp=305,wp=25000)
r6 = Rider(name='JM',wt=74,ht=171,ftp=260,wp=25000)

pl = Paceline(riders=[r1,r2,r3,r4,r5,
                        r6
                      ]
              ,scaling_factor=1.28)
print('Mean Power at Front:')
print(pl.target_power)
barray = asymp_func(x=np.arange(0,len(pl.riders),1),x0=0,y0=1.0,x1=4,y1=0.62)
#
print('Draft Fractions by Position:')
print(barray)
pl.calc_power_demands(base_array=barray
                     ,height_factor=0.2 #higher the number, the more penalty being tall is
                     ,weight_factor=0.15 #higher then number, the more penalty being heavy is
                     ,draft_factor=0.9 #the higher the number, the more draft advantage is given to smaller riders 
                     )
pl.build_wodf(dur=45*60,
              turn_times=[90,60,60,45,45,5]
              )
print('Power Demands')
print(pl.pwr_df)

fig, ax = plt.subplots()
for r in pl.riders:
    r.calc_wbal()
    ax.plot(r.wbal['Time']/60,r.wbal['Wbal'],label=r.name)
ax.legend()
ax.set_ylabel('Wbal')
ax.set_xlabel('Duration (min)')
ax.set_ylim(0,30000)
plt.grid()
#%%
pp_df = pl.pwr_df[['rider',0]].rename(columns={0:'Power (w)'})
pp_df['Turn (s)'] = [90,60,60,45,45,5]


 #%%           
#"calibration" of height,weight and draft factors + base array
# needs to be cleaned up/more formalized but rough idea for now
# populate observations from riders e.g "in p3 i was averaging x today"

def calcRes(a):         
    r1 = Rider(name='AL',wt=60,ht=170,ftp=300,wp=30000)
    r2 = Rider(name='GM',wt=64,ht=173,ftp=300,wp=30000)
    r3 = Rider(name='DS',wt=81,ht=188,ftp=335,wp=30000)
    r4 = Rider(name='DK',wt=75,ht=178,ftp=312,wp=30000)
    r5 = Rider(name='CP',wt=80,ht=180,ftp=300,wp=30000)
    r6 = Rider(name='SM',wt=82,ht=182,ftp=300,wp=30000)
    r7 = Rider(name='ER',wt=85,ht=184,ftp=322,wp=30000)
    r8 = Rider(name='MB',wt=77.6,ht=180,ftp=320,wp=30000)
    
    pl = Paceline(riders=[r1,r2,r3,r4,r5,r6,r7,r8],turn_times=[90,45,30],scaling_factor=1.3)
    
    height_factor = a[0]
    weight_factor = a[1]
    draft_factor = a[2]
    pl.calc_power_demands(base_array=asymp_func(x=np.arange(0,7,1),x0=0,y0=1.0,x1=7,y1=0.57)
                          ,height_factor=height_factor,weight_factor=weight_factor,draft_factor=draft_factor)
    #observations
    o1 = (r2.power_array[1] - 315)**2
    o2 = (r2.power_array[5] - 245)**2
    o3 = (r2.power_array[2] - 285)**2
    o4 = (r3.power_array[1] - 380)**2
    o5 = (r6.power_array[5] - 300)**2
    o6 = (r7.power_array[5] - 300)**2
    # o7 = (r7.power_array[1] - 360)**2
    o8 = (r1.power_array[0] - 360)**2
    return np.sum([o1,o2,o3,o4,o5,o6,o8])

# r = calcRes(height_factor=1.05,weight_factor=0.1,draft_factor=0.75)

from scipy.optimize import minimize

ie = [0.2,0.3,0.75]
res = minimize(calcRes,x0=ie)
print(res.x)


    
    
