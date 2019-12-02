# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 00:17:28 2019

@author: simd9
"""

#%% Imports

import numpy as np
import copy
import matplotlib.pyplot as plt

#%% Initial constants

rho = 0.4
gamma = 0.6
beta = 0.08
nt = 5
s = 0.03
l0 = 5.0

#%% Search domain
class worm():
    """
    Worm class that will move in the search space
    """
    
    def __init__(self,gso_problem):
        
        self.problem = gso_problem
        self.array = np.random.rand(self.problem.dimension)*(self.problem.bounds[:,1]-self.problem.bounds[:,0])+self.problem.bounds[:,0]
        self.luciferin = l0
        self.neighborhood = []
        self.range = self.problem.rs
        self.translation = s*np.random.rand(self.problem.dimension)
    
    def __eq__(self,other):
        
        return( np.array_equal(self.array,other.array))
    
    def update_luciferin(self):
        
        if np.array_equal(self.translation,np.zeros_like(self.translation)):
            pass
        else:
            self.luciferin = (1-rho)*self.luciferin + gamma*self.problem.fitness(self.array)
        
    def update_range(self):
        
        self.range = min(self.problem.rs,max(0,self.range+beta*(nt-len(self.neighborhood))))
        
    def find_neighborhood(self,population):
        
        self.neighborhood = []
        for other_worm in population.list:
            cdn1 = other_worm.luciferin > self.luciferin
            cdn2 = np.linalg.norm(other_worm.array-self.array) < self.range
            if cdn1 and cdn2:
                self.neighborhood.append(other_worm)
       
    def pick_neighboor(self):
        
        if(not self.neighborhood):
            return(None)
        else : 
            nb_of_neighboors = len(self.neighborhood)
            xk = np.arange(0,nb_of_neighboors)
            pk = np.zeros(nb_of_neighboors)
            
            for j in range(nb_of_neighboors):
                
                pk[j] = self.neighborhood[j].luciferin-self.luciferin
            
            pk = pk/sum(pk)
            
            return(self.neighborhood[np.random.choice(xk,p=pk)])
        
    def set_translation(self,other,s):
        
        if other is None :
            
            self.translation = 0
            
        else :
            
            self.translation = s * (other.array - self.array)/np.linalg.norm(other.array - self.array)

    def translate(self):
        
        out_of_bounds = False
        self.array = self.array + self.translation
        self.update_range()
        for i in range(self.problem.dimension):
            
            if self.array[i] < self.problem.bounds[i,0] or self.array[i] > self.problem.bounds[i,1] :
                
                out_of_bounds = True

        if(out_of_bounds):
            self.array = self.array - 2*self.translation
                                        
class population():

    def __init__(self,gso_problem):
        
        self.size = gso_problem.n
        self.dim = gso_problem.dimension
        self.list = []
        
        for _ in range(self.size):
            
            self.list.append(worm(gso_problem))

    def luciferin_phase(self):
        
        for worms in self.list:
            worms.update_luciferin()
            
            
    def movement_phase(self,s):
        
        for worms in self.list:
            
            worms.find_neighborhood(self)
            worms.set_translation(worms.pick_neighboor(),s)
        
        for worms in self.list:
            
            worms.translate()

    def space_array(self):
        
        Z = np.zeros((self.size,self.dim))
        
        for i in range(self.size):
            
            Z[i,:] = copy.deepcopy(self.list[i].array)
            
        return(Z)
        
class gso_optimization():
    
    def __init__(self,fitness,dimension,bounds,n,rs):
        
        self.fitness = fitness
        self.dimension = dimension
        self.bounds = bounds
        self.n = n
        self.rs = rs
        self.pop = population(self)
        
    def solve(self,nb_of_iterations=100,anim=False):
        
        if(anim):
            animation_list = []
           
            for i in range( nb_of_iterations):
                
                animation_list.append(self.pop.space_array())
                self.pop.luciferin_phase()
                self.pop.movement_phase(s)
            
            return(animation_list)
            
        else:
            
            for i in range(nb_of_iterations):
                self.pop.luciferin_phase()
                self.pop.movement_phase()
                
        
    def analyse(self):
    
        peaks = []        
        array_list = [copy.deepcopy(x.array) for x in self.pop.list]
        
        for array in array_list:
            placed = False
            for peak in peaks:
                for other_array in peak:
                    if np.linalg.norm(array-other_array)<1 and not placed:
                        placed = True
                        peak.append(array)
                        
            if not placed :
                peaks.append([array])
        
        analysed_peaks = []
        
        for peak in peaks:
            
            analysed_peaks.append({"mean":np.mean(peak,axis=0),"var":np.var(peak,axis=0)})
            
        return(analysed_peaks)
     
        
if __name__ == '__main__':
    
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    
    def Himmelblau(X):
        
        x = X[0]
        y = X[1]
        
        return -((x**2+y-11)**2 +(x+y**2-7)**2)
    
    #%%Paramètres du problème
    dim = 2
    bounds = np.array([[-5,5],[-5,5]])
    n_individuals = 200
    rs = 5
    
    #%%Création et résolution
    gso = gso_optimization(Himmelblau,dim,bounds,n_individuals,rs)
    nb_of_iterations = 1000
    animation_list = gso.solve(nb_of_iterations,anim=True)  
    solution = gso.analyse()
    
    #%%Animation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [],"o", markersize=2)
    ax.set_xlim(bounds[0,0],bounds[0,1])
    ax.set_ylim(bounds[1,0],bounds[1,1])
    
    #Contour of Himmerblau's function
    
    xlist = np.linspace(-5, 5, 100)
    ylist = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(xlist, ylist)
    
    Z = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            
            Z[i,j] = Himmelblau([X[i,j],Y[i,j]])
    
    cp = ax.contourf(X,Y,Z,25)
    plt.colorbar(cp)
    
    #Animation of the fireflies
    scat = ax.scatter([],[])
    
    def animate(i):
            Z = animation_list[i]
            scat.set_offsets(Z)
            
    anim = FuncAnimation(fig, animate, interval=60, frames=nb_of_iterations)

    plt.draw()
    plt.show()