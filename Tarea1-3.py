# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:03:11 2026

@author: Mauricio Gómez Acosta
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#Para obviar las unidades, dejamos todo de manera adimensional

#Campo magnético
B=np.array([0,0,5])


#Damos una velocidad inicial aleatoria
V=np.array([np.random.normal(0,1),np.random.normal(0,1),np.random.normal(0,1)])
m=1
bet=1
q=1
K=1
T=1
n=20000
dt=0.0001    #Diferencia de tiempo
t=np.linspace(0,n*dt,n+1)



#Donde guardar las velocidades que van cambiando
Vx=np.zeros(n+1)
Vy=np.zeros(n+1)
Vz=np.zeros(n+1)

for i in range(n):
    
    #Guardamos la velocidad
    Vx[i]=V[0]
    Vy[i]=V[1]
    Vz[i]=V[2]
    
    #Actualizamos el producto cruz
    C=np.cross(B,V)
    
    #Definimos el ruido blanco como
    N=np.random.normal(0,1,3)
    sigma=np.sqrt(2*bet*K*T)/m
    
    #La actualizamos y repite el loop
    V = V + (-(bet/m)*V -(q/m)*C)*dt + sigma*np.sqrt(dt)*N
    
Vx[n]=V[0]
Vy[n]=V[1]
Vz[n]=V[2]

plt.plot(t, Vx, label="Vx")
plt.plot(t, Vy, label="Vy")
plt.plot(t, Vz, label="Vz")
plt.xlabel("t")
plt.ylabel("Velocidad")
plt.legend()
plt.title("Dinamica de la velocidad")