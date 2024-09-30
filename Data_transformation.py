import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import codecademylib3_seaborn 

## add code below
## read in csv file
data = pd.read_csv('State_Control.csv')

print(data.dtypes)

## set variable
times = data['time_x']
x = data['x']
y = data['y']
z = data['z']
roll = data['roll']
pitch = data['pitch']
yaw = data['yaw']
xdot = data['xdot']
ydot = data['ydot']
z_dot = data['zdot']
p = data['p']
q = data['q']
r = data['r']

u1 = data['thrust']
u2 = data['roll angle']
u3 = data['pitch angle']
u4 = data['yaw angle']
