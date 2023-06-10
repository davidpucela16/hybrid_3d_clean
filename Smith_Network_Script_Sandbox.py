#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

import os 
import sys
import pandas as pd
import numpy as np 
import pdb 
from numba import njit
import scipy as sp
from scipy.sparse.linalg import spsolve as dir_solve
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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


# Constants
factor_flow=1
cells_3D=20
n=3
Network=1
gradient="y"


# Paths
script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300" #The path with the network
path_network=os.path.join(path_script, "../Synthetic_ROIs_300x300x300")

#path_output="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Rea{}".format(Network)
path_output=os.path.join(path_script, "../Synthetic_Rea{}".format(Network))


#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#Directory to save the divided fiiles of the network
output_dir_network=os.path.join(path_matrices, "divided_files_{}".format(gradient))

#Path where the .am is located and the name of the file 
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300" #The path with the network
path_network=os.path.join(path_script, "../Synthetic_ROIs_300x300x300")
filename=os.path.join(path_network,"Rea{}_synthetic_{}.Smt.SptGraph.am".format(Network, gradient))

#Directory to save the reconstruction

#output_dir_network = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir_network, exist_ok=True)  # Create the output directory if it doesn't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
path_output_data=os.path.join(path_matrices, gradient)
os.makedirs(path_output_data, exist_ok=True)
os.makedirs(os.path.join(path_matrices, "E_portion"), exist_ok=True)

#True if no need to compute
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
#I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
I_assembly_bool=False
#True if need to compute
Computation_bool = True
rec_bool=False
Constant_Cv=False

#######################################################################################
sys.path.append(os.path.join(path_script, "src_final"))
from mesh_1D import mesh_1D
from hybridFast import hybrid_set_up
from mesh import cart_mesh_3D
from post_processing import GetPlaneReconstructionFast
from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate
from PrePostTemp import GetEdgeConcentration,GetInitialGuess, SplitFile, SetArtificialBCs, ClassifyVertices, get_phi_bar, Get9Lines, VisualizationTool, GetCoarsePhi

if Computation_bool: 
    output_files = SplitFile(filename, output_dir_network)
    print("Split files:")
    for file in output_files:
        print(file)

#%
df = pd.read_csv(output_dir_network + '/output_0.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_0.txt', 'r') as file:
    # Read the first line
    output_zero = file.readline()
pos_vertex=df.values

df = pd.read_csv(output_dir_network + '/output_1.txt', skiprows=1, sep="\s+", names=["init", "end"])
with open(output_dir_network + '/output_1.txt', 'r') as file:
    # Read the first line
    output_one = file.readline()
edges=df.values

df = pd.read_csv(output_dir_network + '/output_2.txt', skiprows=1, sep="\s+", names=["cells_per_segment"])
with open(output_dir_network + '/output_2.txt', 'r') as file:
    # Read the first line
    output_two = file.readline()
cells_per_segment=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_3.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_3.txt', 'r') as file:
    # Read the first line
    output_three= file.readline()
points=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_4.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_4.txt', 'r') as file:
    # Read the first line
    output_four= file.readline()
diameters=np.ndarray.flatten(df.values)

diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(output_dir_network + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Pressure=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()

#I increase the Pe because it is way too slow
Flow_rate=np.ndarray.flatten(df.values)*factor_flow


K=np.average(diameters)/np.ndarray.flatten(diameters)
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9 #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)


########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################

for i in range(len(edges)):
    gradient=Pressure[2*i+1] - Pressure[2*i]
    if gradient<0:
        print("Modifying edge ", i)
        edges[i,0]=endVertex[i]
        edges[i,1]=startVertex[i]    
    
startVertex=edges[:,0]
endVertex=edges[:,1]

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)

#%% - Creation of the 3D and Network objects
L_3D=np.array([300,300,300])

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(U))
net.U=np.ndarray.flatten(U)

mesh=cart_mesh_3D(L_3D,cells_3D)
net.PositionalArraysFast(mesh)

cumulative_flow=np.zeros(3)
for i in range(len(Flow_rate)):
    cumulative_flow+=Flow_rate[i]*net.tau[i]
    
print("cumulative flow= ", cumulative_flow)


#%%
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)
#prob.intra_exit_BC="Dir"
#TRUE if no need to compute the matrices
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool

#%%

if Computation_bool:
    prob.AssemblyProblem(path_matrices)
    #M_D=0.001
    M_D=0.002
    Real_diff=1.2e5 #\mu m^2 / min
    CMRO2=Real_diff * M_D
    prob.Full_ind_array[:prob.F]-=M_D*mesh.h**3
    print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
    #plt.spy(prob.Full_linear_matrix)
    #
    if Constant_Cv:
        #Constant Cv
        Lin_matrix=prob.Full_linear_matrix.tolil()[:prob.F+prob.S,:prob.F+prob.S]
        ind_array=prob.Full_ind_array[:prob.F+prob.S]
        ind_array[-prob.S:]+=prob.F_matrix.dot(np.ones(prob.S))
        sol=dir_solve(Lin_matrix,-ind_array)
        sol=np.concatenate((sol, np.ones(prob.S)))
    else:
        sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)

    # =============================================================================
    # a=GetInitialGuess(np.ones(len(edges)), prob)
    # sol=sp.sparse.linalg.bicgstab(prob.Full_linear_matrix, -prob.Full_ind_array, x0=np.concatenate((a[0], a[1], a[2])))[0]
    # =============================================================================
    np.save(os.path.join(path_matrices, 'sol'),sol)

sol=np.load(os.path.join(path_matrices, 'sol.npy'))
prob.q=sol[-2*prob.S:-prob.S]
prob.s=sol[:-2*prob.S]
prob.Cv=sol[-prob.S:]
# =============================================================================
# prob.q=a[0]
# prob.s=a[1]
# prob.Cv=a[2]
# =============================================================================
res=20
aaz=VisualizationTool(prob, 2,0,1, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
aaz.GetPlaneData(path_output_data)


aaz.PlotData(path_output_data)
pdb.set_trace()

#%%
#res=100
simple_plotting=True
corners=np.array([[16,16],[16,289],[289,16],[289,289]])
if simple_plotting:    
    
    aax=VisualizationTool(prob, 0,1,2, corners, res)
    aax.GetPlaneData(path_output_data)
    aay=VisualizationTool(prob, 1,0,2, corners, res)
    aay.GetPlaneData(path_output_data)
    
    aaz=VisualizationTool(prob, 2,1,0, corners, res)
    aaz.GetPlaneData(path_output_data)
    
    aax2=VisualizationTool(prob, 0,2,1, corners, res)
    aax2.GetPlaneData(path_output_data)
    
    aax.PlotData(path_output_data)
    aay.PlotData(path_output_data)
    aaz.PlotData(path_output_data)
    
    aax2.PlotData(path_output_data)

if rec_bool:
    num_processes=25
    process=0 #This must be kept to zero for the parallel reconstruction to go right
    perp_axis_res=50
    path_vol_data=os.path.join(path_output_data, "vol_data")
    aaz=VisualizationTool(prob, 2,0,1, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aaz.GetVolumeData(num_processes, process, perp_axis_res, path_vol_data)

phi_coarse=GetCoarsePhi(prob, prob.q, prob.Cv, prob.s)

#%% - Data for Avizo
def GetEnteringExiting(vertex_to_edge, init):
    label=np.zeros(len(vertex_to_edge))
    c=0
    for i in vertex_to_edge:
        if len(i)==1:
            ed=i[0]
            if c in init:
                #entering
                label[c]=1
            else:
                #exiting
                label[c]=2
        c+=1
    return(label)

edge_concentration=GetEdgeConcentration(prob)
vertex_label=GetEnteringExiting(vertex_to_edge, startVertex)
title="@8 # Edge Concentration"
np.savetxt(os.path.join(path_matrices, "Edge_concentration.txt"), edge_concentration, fmt='%f', delimiter=' ', header=title, comments='')
title="@9 # Entry Exit"
np.savetxt(os.path.join(path_matrices, "Entry_Exit.txt"), vertex_label, fmt='%d', delimiter=' ', header=title, comments='')

#Now we obtain which edges are entering and which are exiting
filtered_lists = [lst for lst, lbl in zip(vertex_to_edge, vertex_label) if lbl == 1]
# Convert the filtered lists to a NumPy array
entering =np.squeeze(np.array(filtered_lists))

filtered_lists = [lst for lst, lbl in zip(vertex_to_edge, vertex_label) if lbl == 2]
# Convert the filtered lists to a NumPy array
exiting =np.squeeze(np.array(filtered_lists))

def GetSources(pos_s, pos_vertex, vertex_label, propert, entry_exit):
    if entry_exit=="entering":
        print("Entering segments")
        pos=pos_vertex[np.where(vertex_label==1)[0]]
    elif entry_exit=="exiting":
        print("Exiting segments")
        pos=pos_vertex[np.where(vertex_label==2)[0]]
    else:
        print("Non boundary segments")
        pos=pos_vertex[np.where(vertex_label==0)[0]]
        
    IDs=np.zeros(0, dtype=np.int64)
    for i in pos:
        source=np.argmin(np.sum((pos_s-i)**2, axis=1))
        if source>len(pos_s): pdb.set_trace()
        IDs=np.append(IDs, source)
    return propert[IDs]
    
plt.plot(GetSources(net.pos_s, pos_vertex, vertex_label, prob.Cv, "entering"))
plt.plot(GetSources(net.pos_s, pos_vertex, vertex_label, prob.Cv, "exiting"))

#%% - Plot profile of concentration on entering edges
def GetEdgeConc(cells_per_segment, prop, edge):
    return prop[np.sum(cells_per_segment[:edge]):np.sum(cells_per_segment[:edge+1])]

#def GetVertexConc(vertex_to_edge, startVertex, )

for i in entering:
    #plt.plot(GetEdgeConc(net.cells, prob.Cv, i))
    plt.plot(GetEdgeConc(net.cells, prob.Cv, i)[::-1])
    plt.ylim((0,1))
    plt.show()


#%%

# Assuming you have the arrays phi and x defined
bins=50

h = L_3D[0]/bins  # Set the distance threshold 'h'

# Calculate average concentration for each position within distance 'h'
unique_x = np.linspace(h/2, L_3D[0]-h/2, bins)
average_phi = []
for pos in unique_x:
    mask = np.abs(net.pos_s[:,0] - pos) <= h
    average_phi.append(np.mean(prob.Cv[mask]))

# Plotting the average concentration
plt.plot(unique_x, average_phi)
plt.xlabel('Position')
plt.ylabel('Average Concentration')
plt.title('Average Concentration vs Position (within distance h)')
plt.show()
#%%

# Assuming you have the arrays phi and x defined
bins=50

h = L_3D[0]/bins  # Set the distance threshold 'h'

# Calculate average concentration for each position within distance 'h'
unique_x = np.linspace(h/2, L_3D[0]-h/2, bins)
average_phi = []
for pos in unique_x:
    mask = np.abs(net.pos_s[:,0] - pos) <= h
    average_phi.append(np.sum(mask))

# Plotting the average concentration
plt.plot(unique_x, average_phi)
plt.xlabel('Position')
plt.ylabel('Average Concentration')
plt.title('Average Concentration vs Position (within distance h)')
plt.show()
