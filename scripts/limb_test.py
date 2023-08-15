#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:53:27 2023

@author: clairehe

Script to test whether we find the optimal 3D transformation
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


n = 1000
centers = np.array([[5,5,5], [-5, -10, -10], [10, -10, -2]])

np.random.seed(123)
X=[]
for i in range(3):
    sigma = np.diag(np.random.uniform(0,1,size=3))
    X.append(np.random.multivariate_normal([0,0,0], sigma, size=n)+np.repeat([centers[i]],n, axis=0))

X = np.array(X)

def visualize_3d(X):
    color = ['r','g','b']
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i,:,0],X[i,:,1],X[i,:,2], color=color[i])
    plt.title('3D plot')

visualize_3d(X)



L = np.array([np.random.uniform(size=3) for i in range(3)])

q = X@L

visualize_3d(q)


from eks.multiview_pca_fish2 import find_linear_transformation, objective 


#%%
from scipy.optimize import minimize
def objective2(L,q):
    n = int(np.sqrt(len(L)))
    d = 0
    L = np.asarray(L).reshape((n,n))
    upper_indices = np.triu_indices(n,1) #offset to diagonal
    # (n-1)n/2 
    L[upper_indices] = np.zeros((n*(n-1)//2))
    s = np.zeros(q.shape[1])
    for i in range(q.shape[0]):
        for j in range(q.shape[0]):
            if j>i:
                s += np.linalg.norm((q[i,:]-q[j,:])@L,axis=1)
            else:
                pass
        d += np.var(np.linalg.norm(q[i,:]@L, axis=1))
    return np.var(s)





def find_linear_transformation2(q, L_initial):
    '''
    Find the linear transformation in the latent space that keeps variance of limbs almost constant
    
    Parameters
    ----------
      L : ndarray (number of latents, number of latents)
          transformation matrix initialisation.
      q : ndarray (number of keypoints, number of timesteps, number of latents)
          latents vector.
    
      Returns
    -------
        L :
            3D transformation 
    '''
    n = int(np.sqrt(len(L_initial)))
    # Define the optimization problem with the objective function and constraint
    problem = {
        'fun': objective2,
        'x0': L_initial,
        'args':q
        }
    
    # Solve the optimization problem
    result = minimize(**problem, options={'gtol': 1e-6, 'disp': True})
    
    # Get the optimal solution for L
    optimal_L = result.x
    
    return optimal_L.reshape((n,n))


#%%%
visualize_3d(X)


O = find_linear_transformation(q, np.ones((3,3)).flatten())

O2 = find_linear_transformation2(q, np.ones((3,3)).flatten())

visualize_3d(q@O)
visualize_3d(q@O2)


#%%

color = ['r','g']
style = ['-.','dotted']
fig, ax = plt.subplots(1,1,figsize=(20,10))
for i in range(len(q)):
    for j in range(q[0].shape[1]):
        ax.plot(q[i, 1:,j], color = color[i], linestyle = style[j], label='limb length edge {}'.format(constraint[j]) +' with mu {}'.format(mu[i]))
ax.legend()
plt.title('Limb length trace per frame with constraint mu {}'.format(mu[1])+' against no constraint')