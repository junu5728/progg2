""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import numba
import matplotlib.pyplot as plt
import math as m
import numpy as np
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit
from multiprocessing import Pool

# Exc1
def approximate_pi(n):# n is the number of points
    inside_x=[]
    inside_y=[]
    outside_x=[]
    outside_y=[]
    for i in range(n):
        x=random.uniform(-1, 1)
        y=random.uniform(-1, 1)
        if (x**2 + y**2)<=1:
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
    n_c=len(inside_x)
    pi_approx=4*(n_c/n)
    plt.figure(figsize=(6, 6))
    plt.scatter(inside_x,inside_y, color="red")
    plt.scatter(outside_x, outside_y, color="blue")
    plt.savefig(f"pi_{n}.png")
    plt.show()
    plt.close()
    return pi_approx

# Exc2, approximation
def sphere_volume(n, d): # n is the number of points, # d is the number of dimensions of the sphere 
    points=[[random.uniform(-1,1) for i in range(d)]for ii in range(n)]
    hypersphere=map(lambda x:sum(xi**2 for xi in x), points)
    inside=list(filter((lambda y:y<=1),hypersphere))      
    n_c=len(inside)
    volume_cube=2**d
    volume_approx=2**d * (n_c/n)
    return volume_approx

#Exc2, real value
def hypersphere_exact(n, d):
    points=[[random.uniform(-1,1) for i in range(d)]for ii in range(n)]
    hypersphere=map(lambda x:sum(xi**2 for xi in x), points)
    inside=list(filter((lambda y:y<=1),hypersphere))
    Vd=((m.pi)**(d/2))/(m.gamma(d/2 +1))
    return Vd

#Exc3: numba version
#Regular sphere_volume:
#Time 1: 2.4369426667690277 s
#Time 2: 2.4385787919163704 s
#Time 3: 2.4133323333226144 s

#Numba version:
#Time 1: 1.0115995001979172 s
#Time 2: 0.854049582965672 s
#Time 3: 0.7302932920865715 s

@njit
def sphere_volume_numba(n:int, d:int)->float:# n is the number of points
    points=[[random.uniform(-1,1) for i in range(d)]for ii in range(n)]
    inside=[]
    for x in points:
        sum=0
        for xi in x:
             sum+=xi**2
        if sum <= 1:
             inside.append(sum)
    n_c=len(inside)
    volume_cube=2**d
    volume_approx=2**d * (n_c/n)
    return volume_approx
    # d is the number of dimensions of the sphere
    #np is the number of processes


#Exc4: parallel code - parallelize actual computations by splitting data
def sep_processes(args):
    n,d = args
    count=0
    points=[[random.uniform(-1,1) for i in range(d)]for ii in range(n)]
    for x in points:
        s=0
        for xi in x:
            s+=xi**2
        if s <= 1:
            count+=1
    return count

def sphere_volume_parallel(n, d, np=10):
    with Pool(np) as pool:
        args=[(n//np,d) for i in range(np)]
        counts=pool.map(sep_processes, args)
        return 2**d * (sum(counts)/n)
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Numba time is: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop-start}")
   


if __name__ == '__main__':
	main()
