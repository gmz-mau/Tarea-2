# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 22:18:26 2026

@author: Mauricio Gómez Acosta
"""

import numpy as np
import matplotlib.pyplot as plt

#Parametros para el modelo

A=2
B=1
K=1
n=2
Omega=[10,50,200]
Tmax=200

def g(x):
    return 1/(1+(x/K)**n)

def f(Om):
    #Supongamos que comenzamos con r1=0 y r2=0
    r1=0
    r2=0
    t=0

    int_tiempo=[t]
    num_r1=[r1]
    num_r2=[r2]

    #Creamos todas las reacciones
    while t<Tmax:
        a1=A*g(r2/Om)
        a2=A*g(r1/Om) 
        a3=B*r1/Om if r1>0 else r1==0  #Evitamos cantidades negativas
        a4=B*r1/Om if r2>0 else r2==0
        a0=a1+a2+a3+a4
        
        if a0==0:
            break

        #Avanzamos en el tiempo, como estamos hablando de una probabilidad respecto al tiempo, es un proceso de Poisson
        tao=np.random.exponential(1/a0) #Tiempo estocastico de que ocurra alguna reacción
        t +=tao
    
    
        #Reacción que ocurrira
        r=np.random.rand()*a0         #Creamos el intervalo [0,a0)
        
        if r<a1:
            r1+=1                #Ocurre la primera reacción, la cual resulta en r1+1
        elif r<a1+a2:
            r2+=1                #Mismo que el anterior con la segunda reacción
        elif r<a1+a2+a3:
            r1-=1                #Ocurre la 3ra reacción, la cual resulta en r1-1
        else:
            r2-=1                #Mismo que el anterior con la 4ta reacción
                        
        int_tiempo.append(t)
        num_r1.append(r1)
        num_r2.append(r2)
    return int_tiempo,num_r1,num_r2

for O in Omega:
    int_tiempo,num_r1,num_r2=f(O)
    plt.plot(int_tiempo,num_r1,label=f'Especie 1, Omg={O}')
    plt.plot(int_tiempo,num_r2,label=f'Especie 2, Omg={O}')

plt.xlabel("Tiempo (s)")
plt.ylabel("Cantidad de especie")
plt.legend()

#Podemos notar que hay saltos muy grandes al hacer Omega chica, pero se mantienen bastante juntas al tener una Omega muy grande
#Igual si tenemos una Omega mediana, por un momento parece igual que la Omega grande, pero tiene cambios bruscos que parecen a las de Omegas pequeñas








    
    
        
























