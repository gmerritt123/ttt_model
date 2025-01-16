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
def power_speed(v,cd,a,wc,wb,g,vhw=0,rho=1.22601,crr=0.004,lossdt=2,draft_factor=1):
    '''
    power given speed
    v = speed (m/s)
    cd = drag coefficient, cadex with disc is 0.7185 --> https://zwifterbikes.web.app/whatif
    a = frontal area
    rho = air density (1.22601 kg/m3 --> https://gribble.org/cycling/air_density.html)
    vhw = headwind velocity (m/s)
    wc = weight cyclist (kg)
    wb = weight bike (kg) --> cadex w disc is 9.221 kg --> https://zwifterbikes.web.app/whatif
    g = grade (rise/run*100)
    crr = rolling resistance coefficient (0.004 on pavement --> https://zwiftinsider.com/crr/)
    lossdt = drivertrain losses (/100)
    draft_factor = reduction in the "aero term" component of the equation due to drafting
    '''
    dtl_term = (1-lossdt/100)**-1
    w = wc+wb+2.5 #2.5 kg "weight of kit"  -->https://zwifterbikes.web.app/howitworks
    g_term = 9.8067 * w * (np.sin(np.arctan(g/100))+crr*np.cos(np.arctan(g/100)))
    a_term = 0.5*cd*a*rho*(v+vhw)**2*draft_factor
    return dtl_term * (g_term + a_term)*v


def frontal_area(height,weight):
    '''https://www.researchgate.net/publication/7906171_The_Science_of_Cycling_Factors_Affecting_Performance_Part_2
    height in m
    weight in kg
    '''
    return 0.0293*height**0.725*weight**0.425+ 0.0604


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
    cd: float #drag coefficient of their bike
    wb: float #weight of their bike
    
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
    target_speed: float
    grade: float
    crr: float   
    
    def __post_init__(self):
        assert all([isinstance(x,Rider) for x in self.riders])            
        for r in self.riders:
            tp = power_speed(v=self.target_speed/3.6 #to m/s
                             , cd=r.cd
                             , a=r.frontal_area
                             , wc=r.wt
                             , wb=r.wb
                             , g=self.grade
                             )
            r.target_power = tp

        
        
        
    def calc_power_demands(self,base_array):
        '''given certain scaling factors, calculates the estimated demands for each rider in each position
        scaling factors include:
            -base_array array of numbers [1.0,...0.6 ] etc. that provide a "base array" for estimating draft benefit as a function of position
                - asymp function can be helpful for playing with this
            -height/weight factors scale the power demands based on some combo of rider height/weight, aiming to roughly approximate the zwift voodo
            -draft factor scales the base_array even further, giving additional draft benefit to more slippery riders and penalizing the less aero riders.
            adds a pwr_df (dataframe) to the instance
        '''

        for i,r in enumerate(self.riders):
            parr = []
            for b in base_array:
                p = power_speed(v=self.target_speed/3.6 #to m/s
                                 , cd=r.cd
                                 , a=r.frontal_area
                                 , wc=r.wt
                                 , wb=r.wb
                                 , g=self.grade
                                 ,draft_factor=b
                                 )
                parr.append(p)
            r.power_array = parr    
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


#step one --> define riders, cd and wb are from zwifter bikes for cadex disc
r1 = Rider(name='AL',wt=60,ht=170,ftp=295,wp=20000,cd=0.7185,wb=9.221)
r2 = Rider(name='GM',wt=64,ht=173,ftp=305,wp=20000,cd=0.7185,wb=9.221)
r3 = Rider(name='DS',wt=83,ht=188,ftp=350,wp=20000,cd=0.7185,wb=9.221)

r4 = Rider(name='SG',wt=67,ht=179,ftp=290,wp=20000,cd=0.7185,wb=9.221)
r5 = Rider(name='EK',wt=76,ht=183,ftp=315,wp=20000,cd=0.7185,wb=9.221)
r6 = Rider(name='TR',wt=75,ht=180,ftp=335,wp=20000,cd=0.7185,wb=9.221)



# r1 = Rider(name='ER',wt=81,ht=184,ftp=330,wp=20000,cd=0.7185,wb=9.221)
# r2 = Rider(name='DK',wt=76,ht=178,ftp=322,wp=20000,cd=0.7185,wb=9.221)
# r3 = Rider(name='SB',wt=67,ht=179,ftp=290,wp=20000,cd=0.7185,wb=9.221)
# r4 = Rider(name='LN',wt=77,ht=182,ftp=329,wp=20000,cd=0.7185,wb=9.221)
# r5 = Rider(name='JW',wt=78,ht=185,ftp=311,wp=20000,cd=0.7185,wb=9.221)

# r7 = Rider(name='JM',wt=74,ht=171,ftp=265,wp=20000)
# r8 = Rider(name='SP',wt=88.9,ht=179,ftp=287,wp=20000)
# r8 = Rider(name='SB',wt=67,ht=179,ftp=275,wp=20000)
# r9 = Rider(name='MB',wt=77.4,ht=180,ftp=301,wp=20000)

#step two, define positions in paceline and scaling factor
pl = Paceline(riders=[r1,r3,r2,r6,r5,r4
                      ]
              ,target_speed=47.7
              ,grade = 0, crr= 0.004)

# pl = Paceline(riders=[r1,r2,r3,r4,r5
#                       ]
#               ,target_speed=46.5
#               ,grade = 0, crr= 0.004)

for r in pl.riders:
    print(r)
    print(r.target_power)

#step three, estimate power demands in the paceline
barray = asymp_func(x=np.arange(0,len(pl.riders),1),x0=0,y0=1.0,x1=4,y1=0.62)
#print('Draft Fractions by Position:')
pl.calc_power_demands(base_array=barray)


turn_times = [60,45,45,45,30,30]
# turn_times = [30,30,30,30,30]
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
pl = Paceline(riders=[r2,r3,r6,r5,r4,r1
                      ]
              ,target_speed=33
              ,grade = 3.1, crr= 0.004)
for r in pl.riders:
    print(r)
    print(r.target_power)

    
    
