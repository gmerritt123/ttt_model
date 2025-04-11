# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:11:48 2024

@author: Gaelen
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt

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

# power_speed(v=31.8/3.6,cd=0.7185
#             ,a=frontal_area(1.73,64)
#             ,wc=64
#             ,wb=9.221
#             ,g=1.7
#             ,vhw=0,rho=1.22601
#             ,crr=0.016,lossdt=2,draft_factor=1)

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
    distance: float
    turn_times: list
    
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
                                 ,crr=self.crr
                                 ,draft_factor=b
                                 )
                parr.append(np.max([0,p]))
            r.power_array = parr    
        df = pd.DataFrame(index=[x.name for x in pl.riders]
                          ,data=np.array([x.power_array for x in pl.riders])
                          ,columns=[i for i in range(len(pl.riders))]).round(0).reset_index().rename(columns={'index':'rider'})
        self.pwr_df = df       

    def build_wodf(self):
        '''with power demands calculated, given a total duration and an array of turn times (len = len(self.riders)
        calculates a "workout_df" for each rider in the paceline        
        '''
        dur = self.distance/self.target_speed*60*60 #hours to seconds
        turn_times = self.turn_times
        for i,r in self.pwr_df.iterrows():
            #rearrange the power demands based on positioning
            c = r[list(range(0,len(self.pwr_df)))].to_list()
            pa = c[:i+1][::-1]+c[i+1:][::-1]          
            df = pd.DataFrame(data={'Time':turn_times,'Power':pa})
            nrnds = int(np.ceil(dur/df['Time'].sum()))
            df = pd.concat([df for x in range(nrnds)]).reset_index(drop=True)

            df['cs'] = df['Time'].cumsum()
            df = df[df.index <= df[df['cs']<dur].index.max() + 1].reset_index(drop=True)
            df.loc[df['cs']>dur,'Time'] = (dur-df['cs'].shift(1)).round(0) #need to round to nearest second for wbal calc later..
            df = df.drop(columns='cs')

            r = [x for x in self.riders if x.name == r['rider']][0]
            if hasattr(r,'wo_df'):
                r.wo_df = pd.concat([r.wo_df,df])
            else:
                r.wo_df = df

aero_rb = {'cd':0.814,'wb':6.636}
tt_b = {'cd':0.7185,'wb':9.221}

bike = tt_b
#step one --> define riders, cd and wb are from zwifter bikes for cadex disc

al = Rider(name='AL',wt=60,ht=170,ftp=300,wp=20000,**bike)
gm = Rider(name='GM',wt=64,ht=173,ftp=305,wp=20000,**bike)
ds = Rider(name='DS',wt=81.5,ht=188,ftp=360,wp=20000,**bike)
ek = Rider(name='EK',wt=76,ht=183,ftp=320,wp=20000,**bike)
er = Rider(name='ER',wt=81,ht=184,ftp=345,wp=20000,**bike)
jw = Rider(name='JW',wt=78,ht=185,ftp=311,wp=20000,**bike)
dk = Rider(name='DK',wt=76,ht=178,ftp=310,wp=20000,**bike)
ln = Rider(name='LN',wt=77,ht=182,ftp=334,wp=20000,**bike)
cp = Rider(name='CP',wt=80,ht=183,ftp=300,wp=20000,**bike)
tr = Rider(name='TR',wt=75,ht=180,ftp=335,wp=20000,**bike)
sg = Rider(name='SG',wt=68,ht=179,ftp=288,wp=20000,**bike)
dm1 = Rider(name='DMY',wt=80,ht=182,ftp=290,wp=20000,**bike)
mb = Rider(name='MB',wt=77.4,ht=180,ftp=310,wp=20000,**bike)

ks = Rider(name='KS',wt=80.1,ht=183,ftp=294,wp=20000,**bike)
sm = Rider(name='SM',wt=80,ht=187,ftp=281,wp=20000,**bike)
cm = Rider(name='CM',wt=71,ht=173,ftp=250,wp=20000,**bike)
pl = Rider(name='PL',wt=73,ht=180,ftp=254,wp=20000,**bike)
mw = Rider(name='MW',wt=76,ht=180,ftp=275,wp=20000,**bike)

# sb = Rider(name='SB',wt=70.8,ht=175,ftp=269,wp=20000,**bike)
sb = Rider(name='SB',wt=68.3,ht=175,ftp=294,wp=20000,**bike)
wy = Rider(name='WA-Y',wt=58,ht=168,ftp=200,wp=20000,**bike)
dn = Rider(name='DN',wt=69.8,ht=175,ftp=240,wp=20000,**bike)
cb = Rider(name='CB',wt=78,ht=180,ftp=268,wp=20000,**bike)
# jm = Rider(name='JM',wt=62,ht=167.6,ftp=245,wp=20000,**bike) #joe monks
jm = Rider(name='JM',wt=72.8,ht=186,ftp=318,wp=20000,**bike)

# riders = [gm,,mb,ln]
#flat section
pl_1 = Paceline(riders=[gm,ln,jm,dk,mb]
              ,target_speed=46.3
              ,grade = 0.0, crr= 0.004
              ,distance=12.3*3
              ,turn_times=[90,45,30,30,30]
              )


pls = [pl_1]



# pls = [pl_1,pl_2,pl_3,pl_4,pl_5,pl_6,pl_7,pl_8
#        ]
descs = ['Route'
         ]
      

# pls = np.array([pls for x in range(3)]).flatten() #do 2 laps
# descs = np.array([descs for x in range(3)]).flatten() #do 2 laps
# pls
barray = np.array([1,0.85,0.75,0.71,0.65,0.63,0.625,0.62])
# barray = np.array([1,0.89,0.77,0.69,0.65,0.63,0.62,0.62])


total_dur = 0
for i,pl in enumerate(pls):
    pl.calc_power_demands(base_array=barray[0:len(pl.riders)])
    pl.build_wodf()
    print(descs[i])
    print(pl.pwr_df)
    dur = pl.distance/pl.target_speed*60 #hours to minutes)
    print(dur)
    total_dur = total_dur+dur 

#step 5, now that each rider has a "workout" based on the power demands from the paceline, calculate wbal over the duration of the event
fig, ax = plt.subplots()
for r in pl_1.riders:
    r.calc_wbal()
    ax.plot(r.wbal['Time']/60,r.wbal['Wbal'],label=r.name)
ax.legend()
ax.set_ylabel('Wbal')
ax.set_xlabel('Duration (min)')
ax.set_ylim(0,20000)
plt.grid()
#%%
print('pred Finish Time: '+str(total_dur)+' Minutes')
avg_speed = np.sum([x.distance for x in pls]) / (total_dur/60) #total distance in km / time in hours
print('pred Avg. Speed: '+str(avg_speed)+' km/h')
print('total Distance: '+ str(np.sum([x.distance for x in pls])))

fig, ax = plt.subplots(figsize=(10,8))
for r in pl_1.riders:
    pdf = r.wbal
    pdf['30sPower'] = pdf['Power'].rolling(window=30).mean()
    pdf['qcm'] = (pdf['30sPower']**4).cumsum()
    pdf['NP'] = (pdf['qcm']/pdf['Time'])**0.25
    pdf['IF'] = pdf['NP']/r.ftp
    ax.plot(pdf['Time']/60,pdf['NP'],label=r.name)
ax.legend(loc='lower right')
ax.set_ylabel('Normalized Power')
ax.set_xlabel('Duration (min)')
# ax.set_ylim(0.8,1.2)
ax.set_ylim(200,400)
plt.grid() 


fig, ax = plt.subplots(figsize=(10,8))
for r in pl_1.riders:
    pdf = r.wbal
    pdf['30sPower'] = pdf['Power'].rolling(window=30).mean()
    pdf['qcm'] = (pdf['30sPower']**4).cumsum()
    pdf['NP'] = (pdf['qcm']/pdf['Time'])**0.25
    pdf['IF'] = pdf['NP']/r.ftp
    ax.plot(pdf['Time']/60,pdf['IF'],label=r.name)
ax.legend(loc='lower right')
ax.set_ylabel('Intensity Factor')
ax.set_xlabel('Duration (min)')
ax.set_ylim(0.8,1.2)
# ax.set_ylim(200,400)
plt.grid() 
#%%
# print('Power Demands')
# print(pl.pwr_df)
#printout of pull plan
pp_df = pl_1.pwr_df[['rider',0]].rename(columns={0:'Power (w)'})
pp_df['Turn (s)'] = pl_1.turn_times
pp_df
#%%
edf = pd.read_csv(r'C:\Repo\ttt_model\RoadToRuinsStartPier.csv')
ddf = pd.DataFrame(data={'distance_m':[p.distance for p in pls],'speed_kmh':[p.target_speed for p in pls]})
ddf['distance_m'] = ddf['distance_m'].cumsum()*1000
edf = pd.concat([edf,ddf]).sort_values(by='distance_m').reset_index(drop=True).set_index('distance_m')
edf['altitude_m'] = edf['altitude_m'].interpolate(method='index')
edf['speed_kmh'] = edf['speed_kmh'].bfill()
edf = edf.reset_index()
edf['delta_m'] = (edf['distance_m']-edf['distance_m'].shift(1)).fillna(0)
edf['Time_min'] = (edf['delta_m']/1000) / (edf['speed_kmh'])*60
edf['Time_min'] = edf['Time_min'].cumsum()
edf

#%%
from bokeh.plotting import figure, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Range1d, DataRange1d, LinearAxis, CustomJSTickFormatter


p = figure(height=500,width=2000)
p.xaxis[0].axis_label = 'Time (min)'
p.yaxis[0].axis_label = "W'bal"


src_dict = {}
for r in pl_1.riders:
    df = r.wbal.copy()
    df['Time'] = df['Time']/60
    src_dict[r.name] = ColumnDataSource(df)

elev_src = ColumnDataSource(edf)

p.extra_y_ranges['elev'] = DataRange1d()
elev_rend = p.varea(x='Time_min',y2='altitude_m',y1=edf['altitude_m'].min(),fill_alpha=0.5,source=elev_src,y_range_name='elev',fill_color='grey')
p.extra_y_ranges['elev'].renderers=[elev_rend]
ax2 = LinearAxis(y_range_name="elev", axis_label="Elevation (m)")
p.add_layout(ax2, 'right')

dax = LinearAxis(axis_label='Distance (km)')
p.add_layout(dax,'below')

ct = CustomJSTickFormatter(args=dict(src=elev_src)
                           ,code='''

                           for (var i = 0; i < src.data['Time_min'].length; i++){
                                   
                                   if (src.data['Time_min'][i]>=tick){
                                           break
                                           }
                                   }
                           var d2 = src.data['distance_m'][i]/1000
                           var d1 = src.data['distance_m'][i-1]/1000
                           var t2 = src.data['Time_min'][i]
                           var t1 = src.data['Time_min'][i-1]
                           var m = (d2-d1)/(t2-t1)
                           var b = d1 - m*t1
                           
                           return Math.round(b+m*tick,0)
                           ''')
dax.formatter = ct           

from bokeh.palettes import Category20
for i,k in enumerate(src_dict.keys()):
    pl = Category20[len(src_dict.keys())]
    r = p.line(x='Time',y='Wbal',line_width=3,line_color=pl[i],source=src_dict[k],legend_label=k)

save(p,r'C:\Repo\ttt_model\Viewer.html')








# (pp_df['Turn (s)'].sum()+45+30+30)*47000/60/60

 #%%           


    
    
