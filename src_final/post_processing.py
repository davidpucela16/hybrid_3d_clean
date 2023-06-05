#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:23 2023

@author: pdavid
"""
import os 
path=os.path.dirname(__file__)
os.chdir(path)
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from neighbourhood import GetNeighbourhood, GetUncommon
from small_functions import TrilinearInterpolation, auto_TrilinearInterpolation
from small_functions import FromBoundaryGetNormal, AppendSparse, GetBoundaryStatus
from scipy.sparse.linalg import spsolve as dir_solve
from assembly import AssemblyDiffusion3DBoundaries, AssemblyDiffusion3DInterior
from Second_eq_functions import node, InterpolateFast,GetInterpolationKernelFast,GetI1Fast, InterpolatePhiBarBlock

import scipy as sp
from scipy.sparse import csc_matrix

import multiprocessing
from multiprocessing import Pool
from assembly_1D import FullAdvectionDiffusion1D

from numba.typed import List

from mesh import GetID, Get8Closest
from mesh_1D import KernelIntegralSurfaceFast, KernelPointFast
from small_functions import in1D
from GreenFast import SimpsonSurface

from numba import njit, prange
from numba.experimental import jitclass
from numba import int64, float64
from numba import int64, types, typed
import numba as nb
import matplotlib.pylab as pylab

import dask
from dask import delayed

plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)

import time


# =============================================================================
# 
# class Visualization3D():
#     def __init__(self,lim, res, prob, num_proc, vmax, *trans):
#         
#         self.vmax=vmax
#         self.vmin=0
#         self.lim=lim
#         self.res=res
#         self.prob=prob
#         a=(lim[1]-lim[0])*np.array([1,2,3])/4
#         LIM_1=[lim[0], lim[0], lim[1], lim[1]]
#         LIM_2=[lim[0], lim[1], lim[0], lim[1]]
#         
#         perp_x=np.zeros([3,4,3])
#         for i in range(3):
#             perp_x[i]=np.array([a[i]+np.zeros(4),LIM_1 , LIM_2]).T
#             
#         perp_y, perp_z=perp_x.copy(),perp_x.copy()
#         
#         perp_y[:,:,0]=perp_x[:,:,1]
#         perp_y[:,:,1]=perp_x[:,:,0]
#         
#         perp_z[:,:,0]=perp_x[:,:,2]
#         perp_z[:,:,2]=perp_x[:,:,0]
#         
#         if trans:
#             perp_x+=trans
#             perp_y+=trans
#             perp_z+=trans
#             
#             
#         data=np.empty([9, res, res])
#         for i in range(3):
#             for j in range(3):
#                 #cor are the corners of a square 
#                 if i==0: cor=perp_x
#                 if i==1: cor=perp_y
#                 if i==2: cor=perp_z
#                 
#                 a,b=self.getCoordReconstChat(cor[j], res, num_processes=num_proc)
#                 data[i*3+j]=b.reshape(res, res)
#                 
#         self.data=data
# # =============================================================================
# #         arg=0
# #         for i in range(3):
# #             for j in range(3):
# #                 #cor are the corners of a square 
# #                 if i==0: cor=perp_x
# #                 if i==1: cor=perp_y
# #                 if i==2: cor=perp_z
# #                 
# #                 arg.append(self.getCoordReconstChat(cor[j], res, num_processes=num_proc))
# #                 a,b=self.getCoordReconstChat(cor[j], res, num_processes=num_proc)
# #                 
# #                 data[i*3+j]=b.reshape(res, res)
# #                 
# #         self.data=data
# # =============================================================================
#         self.perp_x=perp_x
#         self.perp_y=perp_y
#         self.perp_z=perp_z
#         
#         self.plot(data, lim)
#         
#         return
#     
#     def getCoordReconstChat(self, corners, resolution, num_processes=4):
#         """Corners given in order (0,0),(0,1),(1,0),(1,1)
#         
#         This function is to obtain the a posteriori reconstruction of the field"""
#         print("Number of processes= ", num_processes)
#         crds=np.zeros((0,3))
#         tau=(corners[2]-corners[0])/resolution
#         
#         h=(corners[1]-corners[0])/resolution
#         L_h=np.linalg.norm((corners[1]-corners[0]))
#         
#         local_array= np.linspace(corners[0]+h/2, corners[1]-h/2 , resolution )
#         for j in range(resolution):
#             arr=local_array.copy()
#             arr[:,0]+=tau[0]*(j+1/2)
#             arr[:,1]+=tau[1]*(j+1/2)
#             arr[:,2]+=tau[2]*(j+1/2)
#             
#             crds=np.vstack((crds, arr))
#         
# # =============================================================================
# #         # Create a pool of worker processes
# #         pool = Pool(processes=num_processes)
# #         
# #         # Use map function to apply Interpolate_helper to each coordinate in parallel
# #         results = pool.map(Interpolate_helper, [(self, k) for k in crds])
# #         
# #         # Close the pool to free up resources
# #         pool.close()
# #         pool.join()
# #         
# #         # Convert the results to a numpy array
# #         rec = np.array(results)
# # =============================================================================
#         rec=np.array([])        
#         for k in crds:
#             rec=np.append(rec, InterpolateHelper((self.prob, k)))
#         
#         return crds, rec     
#     
#     def plot(self, data, lim):
#         
#         res=self.res
#         perp_x, perp_y, perp_z=self.perp_x, self.perp_y, self.perp_z
#         
#         # Create a figure with 3 rows and 3 columns
#         fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
#         
#         # Set the titles for each row of subplots
#         row_titles = ['X', 'Y', 'Z']
#         
#         # Set the titles for each individual subplot
#         subplot_titles = ['x={:.2f}'.format(perp_x[0,0,0]), 'x={:.2f}'.format(perp_x[1,0,0]), 'x={:.2f}'.format(perp_x[2,0,0]),
#                           'y={:.2f}'.format(perp_y[0,1,1]), 'y={:.2f}'.format(perp_y[1,1,1]), 'y={:.2f}'.format(perp_y[2,1,1]),
#                           'z={:.2f}'.format(perp_z[0,0,2]), 'z={:.2f}'.format(perp_z[1,0,2]), 'z={:.2f}'.format(perp_z[2,2,2])]
#         
#         
#         # Loop over each row of subplots
#         for i, ax_row in enumerate(axs):
#             # Set the title for this row of subplots
#             ax_row[0].set_title(row_titles[i], fontsize=16)
#             
#             # Loop over each subplot in this row
#             for j, ax in enumerate(ax_row):
#                 # Plot some data in this subplot
#                 x = [1, 2, 3]
#                 y = [1, 4, 9]
#                                              
#                 b=self.data[i*3+j]
#                 
#                 im=ax.imshow(b.reshape(res,res), origin='lower', vmax=self.vmax, vmin=self.vmin, extent=[lim[0], lim[1], lim[0], lim[1]])
#                 # Set the title and y-axis label for this subplot
#                 ax.set_title(subplot_titles[i*3 + j], fontsize=14)
#                 if i==0: ax.set_ylabel('z'); ax.set_xlabel('y')
#                 if i==1: ax.set_ylabel('z'); ax.set_xlabel('x')
#                 if i==2: ax.set_ylabel('x'); ax.set_xlabel('y')
#                 
#         # Set the x-axis label for the bottom row of subplots
#         #axs[-1, 0].set_xlabel('X-axis', fontsize=12)
#         
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
#         
#         # Show the plot
#         plt.show()
# 
# def get_coord_reconst_chat(prob, corners, resolution, num_processes=4):
#     """Corners given in order (0,0),(0,1),(1,0),(1,1)
#     
#     This function is to obtain the a posteriori reconstruction of the field"""
#     print("Number of processes= ", num_processes)
#     crds=np.zeros((0,3))
#     tau=(corners[2]-corners[0])/resolution
#     
#     h=(corners[1]-corners[0])/resolution
#     L_h=np.linalg.norm((corners[1]-corners[0]))
#     
#     local_array= np.linspace(corners[0]+h/2, corners[1]-h/2 , resolution )
#     for j in range(resolution):
#         arr=local_array.copy()
#         arr[:,0]+=tau[0]*(j+1/2)
#         arr[:,1]+=tau[1]*(j+1/2)
#         arr[:,2]+=tau[2]*(j+1/2)
#         
#         crds=np.vstack((crds, arr))
#     
#     # Create a pool of worker processes
#     pool = Pool(processes=num_processes)
#     
#     # Use map function to apply interpolate_helper to each coordinate in parallel
#     results = pool.map(InterpolateHelper, [(prob, k) for k in crds])
#     
#     # Close the pool to free up resources
#     pool.close()
#     pool.join()
#     
#     # Convert the results to a numpy array
#     rec = np.array(results)
# # =============================================================================
# #     rec=np.array([])        
# #     for k in crds:
# #         rec=np.append(rec, InterpolateHelper((prob, k)))
# # =============================================================================
#     
#     return crds, rec   
# 
# =============================================================================
@njit
def ReconstructionCoordinatesFast(crds, n, cells_x, cells_y,cells_z, h_3D,pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D, s, q):
    """This function is thought to estimate the concentration field at a given set of coordinates 
    using parallel dask computation"""
    print("Executing reconstruction set of coordinates")
    phi=np.zeros(len(crds), dtype=np.float64)
    e=0
    
    for i in crds:
        a,b,c,d=InterpolateFast(i, n, cells_x, cells_y,
                              cells_z, h_3D,GetBoundaryStatus(i,h_3D, cells_x, cells_y, cells_z), 
                              pos_cells,s_blocks, source_edge,
                              tau, pos_s, h_1D, R, D)
        phi[e]=a.dot(s[b]) + c.dot(q[d]) 
        e+=1
    return phi

@njit(parallel=True)
def ReconstructionCoordinatesParallel(crds, n, cells_x, cells_y,cells_z, h_3D,pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D, s, q):
    """This function is thought to estimate the concentration field at a given set of coordinates 
    using parallel dask computation"""
    print("Executing reconstruction set of coordinates, parallel")
    phi=np.zeros(len(crds), dtype=np.float64)
    e=0
    
    for j in prange(len(crds)):
        i=crds[j]
        a,b,c,d=InterpolateFast(i, n, cells_x, cells_y,
                              cells_z, h_3D,GetBoundaryStatus(i,h_3D, cells_x, cells_y, cells_z), 
                              pos_cells,s_blocks, source_edge,
                              tau, pos_s, h_1D, R, D)
        phi[j]=a.dot(s[b]) + c.dot(q[d]) 
        e+=1
    return phi


# =============================================================================
# def ReconstructionCoordinatesFast(crds, prob):
#     """This function is thought to estimate the concentration field at a given set of coordinates 
#     using parallel dask computation"""
#     phi=[]
#     c=0
#     for i in crds:
#         phi.append(InterpolateHelperDask((prob, i)))
#         c+=1
#     return dask.compute(phi)
# =============================================================================

def InterpolateHelper(args):
    prob,coords=args
    a,b,c,d = prob.Interpolate(coords)
    return a.dot(prob.s[b]) + c.dot(prob.q[d]) 

@dask.delayed
def InterpolateHelperDask(args):
    prob,coords=args
    a,b,c,d = prob.Interpolate(coords)
    return a.dot(prob.s[b]) + c.dot(prob.q[d]) 



#%% - The following functions are written more calmly and will probably substitute many of the functions above
def GetPlaneReconstructionFast(plane_coord,plane_axis, i_axis, j_axis,corners, resolution, prob, property_array, *save):
    print("Plane Reconstruction Fast")
    crds=GetCoordsPlane(corners, resolution)
    mask=GetPlaneIntravascularComponent(plane_coord, prob.mesh_1D.pos_s, prob.mesh_1D.source_edge, 
                                        plane_axis, i_axis, j_axis, corners, prob.mesh_1D.tau, 
                                        resolution, prob.mesh_1D.R, prob.mesh_1D.h, prob.mesh_1D.cells)
    intra=property_array[mask-1]
    result = np.where(mask == 0, np.nan, intra)
    new_mask=mask > 0
    phi=ReconstructionCoordinatesFast(crds, prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z,
                                         prob.mesh_3D.h,prob.mesh_3D.pos_cells,
                                         prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
                                         prob.R, 1, prob.s, prob.q)
    phi_2=phi.reshape(resolution, resolution).copy()
# =============================================================================
#     plt.imshow(mask)
#     plt.show()
#     
#     plt.imshow(phi.reshape(resolution,resolution))
#     plt.show()
#     
#     plt.imshow(result)
#     plt.show()
# =============================================================================
# =============================================================================
#     plt.imshow(phi_2, origin="lower", extent=[0,prob.mesh_3D.L[i_axis],0,prob.mesh_3D.L[j_axis]])
#     plt.xlabel(np.array(["x","y","z"])[i_axis])
#     plt.ylabel(np.array(["x","y","z"])[j_axis])
#     plt.title("Extravascular $\phi$; "+ np.array(["x","y","z"])[plane_axis] + '={:.1f}'.format(plane_coord))
#     plt.colorbar()
#     plt.show()
# =============================================================================
    
    phi_final=phi.reshape(resolution,resolution)
    phi_final[new_mask]=result[new_mask]
    if save:
        dirs=np.array(["x","y","z"])
        np.save(os.path.join(save[0], 'phi_intra_{}={:04g}_{}_{}'.format(dirs[plane_axis], int(plane_coord), dirs[i_axis], dirs[j_axis])),phi_final)
        np.save(os.path.join(save[0], 'phi_extra_{}={:04g}_{}_{}'.format(dirs[plane_axis], int(plane_coord), dirs[i_axis], dirs[j_axis])), phi_2)
        #np.save(os.path.join(save[0], 'crds_{}={}_{}_{}'.format(dirs[plane_axis], int(plane_coord), dirs[i_axis], dirs[j_axis])), crds)
    
    return  phi_final,result,mask, phi_2, crds

def GetPlaneIntravascularComponent(plane_coord, pos_s, source_edge,
                                   plane_axis, i_axis, j_axis, corners, 
                                   tau_array, resolution,R, h_1D, cells_per_segment):
    """This function aims to provide the voxels of the plane defined by plane_coord whose center fall
    within a source cylinder. We work in 2D so it is not as confusing
    1st - We figure out which sources intersect the plane 
    2nd - Out of those possible sources, we loop through each of them to find which voxels fall within the cylinder
    
    plane_coord -> The coordinates of the plane on the axis perpendicular to itself
    pos_s -> position of the sources
    plane_axis -> 1, 2 or 3 for x, y or z
    corners_plane -> self explanatory
    tau_array -> the axial unitary vector for each source
    i_axis -> axis that we want to be in the horizontal direction of the matrix
    j_axis -> axis that we want to be in the vertical direction of the matrix
    
    Corners are given in the following order:
        (0,0), (0,1), (1,0), (1,1)
    where the first entry is the horizontal direction and the second is the vertical direction
    according to the i_axis and j_axis"""
    
    #crds=GetCoordsPlane(corners, resolution)
    #crds_plane=np.delete(crds, plane_axis, axis=1) #We eliminate the component perpendicular to the plane
    
    #First - Figure out the sources that intersect the plane
    pos_s_plane=pos_s[:,plane_axis]
    sources=np.where(np.abs(pos_s_plane - plane_coord) < np.repeat(h_1D, cells_per_segment)/2)[0]
    
    mask=np.zeros((resolution, resolution), dtype=np.int64)
    
    tau=np.linalg.norm((corners[2]-corners[0])/resolution)
    h=np.linalg.norm((corners[1]-corners[0])/resolution)
    
    x=np.linspace(corners[0,i_axis]+tau/2, corners[2,i_axis]-tau/2, resolution)
    y=np.linspace(corners[0,j_axis]+h/2, corners[1,j_axis]-h/2, resolution)
    X,Y=np.meshgrid(x,y)
    
    for s in sources:
# =============================================================================
#         x_dist=X-pos_s[s,i_axis]
#         y_dist=Y-pos_s[s,j_axis]
#         
#         a=(x_dist*tau_array[s,i_axis]**2+y_dist*tau_array[s,j_axis]**2 < h_1D[s]**2)
#         b=(x_dist*tau_array[s,j_axis]**2+y_dist*tau_array[s,i_axis]**2 < R[s]**2)
#         d=np.where(a & b)
# =============================================================================
        #The following 
        P=np.zeros(list(X.shape) + [3])
        P[:,:,i_axis]=X - pos_s[s,i_axis] #i_axis distance of every point of the grid to the center of the source s
        P[:,:,j_axis]=Y - pos_s[s,j_axis] #j_axis distance of every point of the grid to the center of the source s
        P[:,:, plane_axis]=np.zeros_like(X) #plane_axis distance of every point of the grid to the center of the source s
        
        #P = np.stack((X - pos_s[s,0], Y - pos_s[s,1], np.zeros_like(X)), axis=-1)
        
        # Project the position vectors onto the direction vector to get the scalar value t
        v=tau_array[source_edge[s]]
        t = np.sum(P *v, axis=-1)
        
        # Calculate the perpendicular distance of each point from the cylinder axis
        d = np.linalg.norm(P - (t[:, :, np.newaxis] * v), axis=-1)
        
        # Find the indices of points that are within the cylinder
        indices = np.where((d <= R[source_edge[s]]) & (np.abs(t) <= h_1D[source_edge[s]]/2))
        mask[indices]=s+1
    return mask


def GetCoordsPlane(corners, resolution):
    """We imagine the plane with a horizontal (x) and vertical direction (y).
    Corners are given in the following order:
        (0,0), (0,1), (1,0), (1,1)
        
        - tau indicates the discretization size in horizontal direction
        - h indicates the discretization size in vertical direcion"""
    crds=np.zeros((0,3))
    tau=(corners[2]-corners[0])/resolution
    
    h=(corners[1]-corners[0])/resolution
    L_h=np.linalg.norm((corners[1]-corners[0]))
    
    local_array= np.linspace(corners[0]+tau/2, corners[2]-tau/2 , resolution ) #along the horizontal direction
    for j in range(resolution):
        arr=local_array.copy()
        arr[:,0]+=h[0]*(j+1/2)
        arr[:,1]+=h[1]*(j+1/2)
        arr[:,2]+=h[2]*(j+1/2)
        
        crds=np.vstack((crds, arr))
    return(crds)


# =============================================================================
# def GetPlaneReconstructionParallel(plane_coord,plane_axis, i_axis, j_axis,corners, resolution, prob, property_array):
#     crds=GetCoordsPlane(corners, resolution)
#     mask=GetPlaneIntravascularComponent(plane_coord, prob.mesh_1D.pos_s, prob.mesh_1D.source_edge, 
#                                         plane_axis, i_axis, j_axis, corners, prob.mesh_1D.tau, 
#                                         resolution, prob.mesh_1D.R, prob.mesh_1D.h, prob.mesh_1D.cells)
#     intra=property_array[mask-1]
#     result = np.where(mask == 0, np.nan, intra)
#     new_mask=mask > 0
#     
#     phi=ReconstructionCoordinatesParallel(crds, prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z,
#                                          prob.mesh_3D.h,prob.mesh_3D.pos_cells,
#                                          prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
#                                          prob.R, 1, prob.s, prob.q)
#     phi2=phi.copy()
#     
#     plt.imshow(phi.reshape(resolution,resolution), origin="lower")
#     plt.show()
#     
#     plt.imshow(result, origin="lower")
#     plt.show()
#     
#     phi_final=phi.reshape(resolution,resolution)
#     phi_final[new_mask]=result[new_mask]
#     
#     return phi_final,result,phi2.reshape(resolution, resolution)
# =============================================================================
        
        

