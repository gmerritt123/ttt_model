# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:11:48 2024

@author: Gaelen
"""

import pandas as pd
import numpy as np



import matplotlib.pyplot as plt

def build_wodf(wo,dur):
    '''translates a "workout" {'Time':[x,y],'Power':[px,py]} into a dataframe given a total estimated duration.
    workout will repeat itself until duration is used up
    returns a dataframe
    '''
    df = pd.DataFrame(data=wo)
    nrnds = int(np.ceil(dur/df['Time'].sum()))
    df = pd.concat([df for x in range(nrnds)]).reset_index(drop=True)
    df['cs'] = df['Time'].cumsum()
    df = df[df.index <= df[df['cs']<dur].index.max() + 1]
    df.iloc[-1]['Time'] = dur-df.iloc[-2]['cs']
    return df.drop(columns='cs')

def calcWbal(df,cp,wp):
    '''given a df output from build_wodf, critical power (cp) and W' (wp), calculates Wbal over time @ 1 sec resolution)
    stolen from here --> https://forum.intervals.icu/t/using-w-balance/804/4
    '''
    tdf = pd.DataFrame(data={'Time':np.arange(0,df['Time'].max()+1,1)})
    df= pd.merge(df,tdf,how='right',on='Time')
    df['Power'] = df['Power'].bfill()
    wb = np.zeros(len(df))
    for i in range(len(wb)):
        if i == 0:
            wb[i] = wp
            continue
        t = df['Time'].iloc[i]
        deltaW = cp - df['Power'].iloc[i]
        if deltaW > 0:
            w = wb[i-1] + deltaW * (wp - wb[i-1])/wp
        else:
            w = wb[i-1]+deltaW
        wb[i] = w
    df['Wbal'] = wb
    return df

#application

turn_times = [90,60,75,75,30,20] #proposed turn times on the flat AL, GM, TR etc.

p1c = 5.5 #target w/kg of p1 rider on volcano climb

cd =  {1:1.0,2:0.89,3:0.82,4:0.80,5:0.80,6:0.80} #estimate of w/kg reduction on climb by position

#each rider's stats and proposed workout (two segments flat + volcano w estimated durations)
#volcano segment plan is for GM and AL to swap pulls, everyone else hangs on

rd = {'AL':{'segments':{'flat':{'Time':turn_times,'Power':[364,211,211,222,247,291],'dur':25*60}
                                   ,'volcano':{'Time':[120,120],'Power':[290,340],'dur':7*60}} 
                        ,'cp': 300
                        ,'wp':30000
                        ,'wt':60
                        }
      ,'GM':{'segments':{'flat':{'Time':turn_times,'Power':[298,372,216,216,227,253],'dur':25*60}
                       ,'volcano':{'Time':[120,120],'Power':[350,300],'dur':7*60}} 
            ,'cp': 300
            ,'wp':30000
            ,'wt':64
            }
      ,'TR':{'segments':{'flat':{'Time':turn_times,'Power':[271,319,399,243,231,231],'dur':25*60}
                       ,'volcano':{'Time':[120,120],'Power':[360,360],'dur':7*60}} 
            ,'cp': 323
            ,'wp':30000
            ,'wt':75
            }
      ,'SB':{'segments':{'flat':{'Time':turn_times,'Power':[237,264,311,388,225,225],'dur':25*60}
                       ,'volcano':{'Time':[120,120],'Power':[334,334],'dur':7*60}} 
            ,'cp': 299
            ,'wp':30000
            ,'wt':69.5
            }
      ,'SG':{'segments':{'flat':{'Time':turn_times,'Power':[222,233,260,306,382,222],'dur':25*60}
                       ,'volcano':{'Time':[120,120],'Power':[320,320],'dur':7*60}} 
            ,'cp': 275
            ,'wp':30000
            ,'wt':67.5
            }
     
      ,'JW':{'segments':{'flat':{'Time':turn_times,'Power':[238,238,250,279,328,410],'dur':25*60}
                             ,'volcano':{'Time':[120,120],'Power':[350,350],'dur':7*60}} 
                  ,'cp': 297
                  ,'wp':30000
                  ,'wt':78
                  }

      }
#janky way of updating the power demands on the volcano climb via adjusting ONLY p1c
for i,k in enumerate(rd.keys()):
    m = rd[k]['wt']
    if i == 0:
        rd[k]['segments']['volcano']['Power'] = [p1c*cd[i+1]*m,p1c*cd[i+2]*m]
    elif i == 1:
        rd[k]['segments']['volcano']['Power'] = [p1c*cd[i+1]*m,p1c*cd[i]*m]
    else:
        rd[k]['segments']['volcano']['Power'] = [p1c*cd[i+1]*m,p1c*cd[i+1]*m]


fig, ax = plt.subplots()

#for each rider, stack the demands of each segment, and calc Wbal over time for them
for r in rd.keys():
    rdf = pd.DataFrame()
    for seg,wo in rd[r]['segments'].items():
        print(seg)
        df = build_wodf(wo={k:wo[k] for k in ['Time','Power']},dur=wo['dur'])
        rdf=pd.concat([rdf,df])
    rdf['Time'] = rdf['Time'].cumsum()
    df = calcWbal(df=rdf,cp=rd[r]['cp'],wp=rd[r]['wp'])
    ax.plot(df['Time']/60,df['Wbal'],label=r)

ax.legend()
ax.set_ylabel('Wbal')
ax.set_xlabel('Duration (min)')
ax.set_ylim(0,30000)
plt.grid()
    
    
