# -*- coding: utf-8 -*-
"""
 Simple Patch Antenna Tutorial

 Tested with
  - python 3.10
  - openEMS v0.0.34+

 (c) 2015-2023 Thorsten Liebig <thorsten.liebig@gmx.de>

"""

### Import Libraries
import os, tempfile
from pylab import *

from CSXCAD  import ContinuousStructure
from openEMS import openEMS

from openEMS.physical_constants import *
import matplotlib.pyplot as plt

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
# from CSXCAD.CSPrimitives import primitives_mesh_setup
# from CSXCAD.CSProperties import properties_mesh_setup
from Automesher import Automesher
from CSXCAD.SmoothMeshLines import SmoothMeshLines

### General parameter setup
Sim_Path = os.path.realpath(os.path.join('.', 'Simple_Patch_Antenna_Test_automesh_points'))

post_proc_only = True


# patch width (resonant length) in x-direction
patch_width  = 38.5 #
# patch length in y-direction
patch_length = 30

#substrate setup
substrate_epsR   = 3.38
substrate_kappa  = 1e-3 * 2*pi*2.45e9 * EPS0*substrate_epsR
substrate_width  = 46.6
substrate_length = 38
substrate_thickness = 1.6
substrate_cells = 4
msl_length = 11.22
msl_width = 2.88
ls = 4
delta = 0.78

#setup feeding
feed_pos = -6 #feeding position in x-direction
feed_R = 50     #feed resistance

# size of the simulation box
SimBox = np.array([150, 150, 100])*2

# setup FDTD parameter & excitation function
f0 = 200e6 # center frequency
fc = 200e6 # 20 dB corner frequency

### FDTD setup
## * Limit the simulation to 30k timesteps
## * Define a reduced end criteria of -40dB
FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite( f0, fc )
FDTD.SetBoundaryCond( ['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'] )


CSX = ContinuousStructure()

FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0/(f0+fc)/1e-3/20
# mesh_res=int(1)

global_mesh_setup = {
    'dirs': 'y',
    'mesh_resolution': mesh_res,
    'drawing_unit': 1e-3,
}
properties_mesh_setup={}
primitives_mesh_setup={}

### Generate properties, primitives and mesh-grid
#initialize the mesh with the "air-box" dimensions
mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2])
mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2]          )
mesh.AddLine('z', [-SimBox[2]/3, SimBox[2]*2/3]        )
metal_edge_res=mesh_res/2
# create patch
mesh_hint = {

    'dirs': 'x',
    'metal_edge_res' : None,
    'mesh_resolution' : mesh_res
}
patch = CSX.AddMetal( 'patch') # create a perfect electric conductor (PEC)
properties_mesh_setup[patch] = mesh_hint

start = [-patch_width/2, substrate_length/2-patch_length-delta, substrate_thickness]
stop  = [ patch_width/2, substrate_length/2 -delta            , substrate_thickness]
mesh_hint = {

    'dirs': 'xy',
    # 'metal_edge_res': 0.5,
}
# box1=patch.AddBox(priority=1, start=start, stop=stop) # add a box-primitive to the metal property 'patch'
# primitives_mesh_setup[box1] = mesh_hint

# FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)
mesh_hint = {

    'dirs': 'xy'
}
# air1 = CSX.AddMaterial( 'air1', epsilon=1, kappa=0) # create a perfect electric conductor (PEC)
# properties_mesh_setup[air1] = mesh_hint
start = [-msl_width-msl_width/2, substrate_length/2-patch_length-delta   , substrate_thickness]
stop  = [-msl_width/2          , substrate_length/2-patch_length-delta+ls, substrate_thickness]
mesh_hint = {

    'dirs': 'xy'
}
# asd = air1.AddBox(priority=0, start=start, stop=stop, mesh_hint=mesh_hint) 
# FDTD.AddEdges2Grid(dirs='xy', properties=air1, metal_edge_res=mesh_res/2)

start = [msl_width/2            , substrate_length/2-patch_length-delta   , substrate_thickness]
stop  = [msl_width/2 + msl_width, substrate_length/2-patch_length-delta+ls, substrate_thickness]
# air1.AddBox(priority=0, start=start, stop=stop) 
# FDTD.AddEdges2Grid(dirs='xy', properties=air1, metal_edge_res=mesh_res/2)

# start = [-patch_width/8, substrate_length/2-patch_length+patch_length/3-delta, substrate_thickness]
# stop  = [ patch_width/8, substrate_length/2 -delta-patch_length/3            , substrate_thickness]
# air1.AddBox(priority=0, start=start, stop=stop) 

#  create patch with polygon
x = [patch_width/2           , patch_width/2                        , msl_width/2+msl_width                , msl_width/2+msl_width                   , msl_width/2                             , msl_width/2                                        , -msl_width/2                                       , -msl_width/2                            , -msl_width/2-msl_width                  , -msl_width/2-msl_width               , -patch_width/2                       ,  -patch_width/2           , patch_width/8            , patch_width/8                           , -patch_width/8                          , -patch_width/8                                        , patch_width/8                                        , patch_width/8            , patch_width/2]
y = [substrate_length/2-delta , substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls-msl_length, substrate_length/2-patch_length-delta+ls-msl_length, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta,  substrate_length/2 -delta, substrate_length/2 -delta, substrate_length/2 -delta-patch_length/3, substrate_length/2 -delta-patch_length/3,  substrate_length/2 -delta-patch_length+patch_length/3, substrate_length/2 -delta-patch_length+patch_length/3, substrate_length/2 -delta, substrate_length/2 -delta]

x = [85/2 , -85/2,-85/2-5, 85/2, 85/2, 85/2-5, 85/2-5, 85/2-10,   85/2-10,   -85/2+10, -85/2+10, -85/2+5, -85/2+5, -85/2+10,   -85/2+10,   85/2-10, 85/2-10, 85/2, 85/2]
y = [-64/2, -64/2, 64/2, 64/2, -64/2+5,-64/2+5, 64/2-5, 64/2-5, 2.5,  2.5,  64/2-5,  64/2-5, -64/2+5, -64/2+5, -2.5,-2.5,-64/2+5, -64/2+5,-64/2]
# x = [patch_width/2            , patch_width/2                        , msl_width/2+msl_width                , msl_width/    2+msl_width               , msl_width/2                             , msl_width/2                                        , -msl_width/2                                       , -msl_width/2                            , -msl_width/2-msl_width                  , -msl_width/2-msl_width               , -patch_width/2                       ,  -patch_width/2           , 0                        ,0                                        ,-patch_width/8]  
# y = [substrate_length/2 -delta, substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls-msl_length, substrate_length/2-patch_length-delta+ls-msl_length, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta+ls, substrate_length/2-patch_length-delta, substrate_length/2-patch_length-delta,  substrate_length/2 -delta, substrate_length/2 -delta, substrate_length/2 -delta-patch_length/3, substrate_length/2 -delta-patch_length/3]

# x = [patch_width/2, -patch_width/2, -patch_width/2, patch_width/2, +patch_width/2]
# y = [substrate_length/2 - delta, substrate_length/2 - delta, substrate_length/2 - patch_length - delta, substrate_length/2 - patch_length - delta, substrate_length/2 - delta]
# Define the outer polygon (rectangle)
# x = [patch_width/2, patch_width/2, -patch_width/2, -patch_width/2, patch_width/2]
# y = [substrate_length/2 - delta, substrate_length/2 - patch_length - delta, substrate_length/2 - patch_length - delta, substrate_length/2 - delta, substrate_length/2 - delta]
x = [i *2 for i in x]
y = [i *2 for i in y]
# gap = 6
# x= [-10,  0, 0, -3,  3, gap,  gap, 10,  10, -10, -10]
# y= [10 , 10, 5, -5, -5,   5,   10, 10, -10, -10, 10]
points = [x,y]
mesh_hint = {

     'metal_edge_res': mesh_res/2, 'dirs': 'xy'
}
polygon1 = patch.AddPolygon(points, 'z', elevation = substrate_thickness, priority = 1000)
primitives_mesh_setup[polygon1] = mesh_hint
plt.plot(x,y,marker='o')
# plt.show()
# FDTD.AddEdges2Grid(dirs='xy', properties= patch, metal_edge_res=mesh_res/2)

# create msl
# msl = CSX.AddMetal('msl')
# start = [-msl_width/2, substrate_length/2-patch_length-delta+ls-msl_length, substrate_thickness]
# stop  = [ msl_width/2, substrate_length/2-patch_length-delta+ls           , substrate_thickness]
# msl.AddBox(priority = 0, start=start, stop=stop)
# FDTD.AddEdges2Grid(dirs='xy', properties=msl, metal_edge_res=mesh_res/2)

# create substrate
# substrate = CSX.AddMaterial( 'substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
start = [-substrate_width/2, -substrate_length/2, 0]             
stop  = [ substrate_width/2, substrate_length/2 , substrate_thickness]
# substrate.AddBox( priority=1, start=start, stop=stop )

# add extra cells to discretize the substrate thickness
# mesh.AddLine('z', linspace(0,substrate_thickness,substrate_cells+1))

# create ground (same size as substrate)
# gnd = CSX.AddMetal( 'gnd' ) # create a perfect electric conductor (PEC)
start[2]=0
stop[2] =0
# gnd.AddBox(priority=10,start=start, stop=stop, dirs= 'xy', FDTD=FDTD) 

# FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
L=[]
L=CSX.GetAllProperties()
P=[]
P=L[0].GetAllPrimitives()
# print (P[0])
# print (L[0])
MM = Automesher()
MM.GenMesh(CSX, global_mesh_setup,primitives_mesh_setup,properties_mesh_setup)
# print(primitives_mesh_setup)
# print(properties_mesh_setup)
# print(primitives_mesh_setup)
# print(properties_mesh_setup)
# apply the excitation & resist as a current source
# start = [0, -substrate_length/2, 0]
# stop  = [0, -substrate_length/2, substrate_thickness]
# port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5)

# mesh.SmoothMeshLines('x', mesh_res, 1.5)
# mesh.AddLine('x',SmoothMeshLines([-SimBox[0]/2, 0], mesh_res*2, 1.1))


# Add the nf2ff recording box
# nf2ff = FDTD.CreateNF2FFBox()

### Run the simulation
if 1:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'Simple_Patch_Antenna_Test_automesh_points')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

if not post_proc_only:
    FDTD.Run(Sim_Path, verbose=3, cleanup=True)


### Post-processing and plotting
# f = np.linspace(max(1e9,f0-fc),f0+fc,401)
# port.CalcPort(Sim_Path, f)
# s11 = port.uf_ref/port.uf_inc
# s11_dB = 20.0*np.log10(np.abs(s11))
# figure()
# plot(f/1e9, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
# grid()
# legend()
# # ylabel('S-Parameter (dB)')
# xlabel('Frequency (GHz)')

# idx = np.where((s11_dB<-5) & (s11_dB==np.min(s11_dB)))[0]
# if not len(idx)==1:
#     print('No resonance frequency found for far-field calulation')
# else:
#     f_res = f[idx[0]]
#     theta = np.arange(-180.0, 180.0, 2.0)
#     phi   = [0., 90.]
#     nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0,0,1e-3])

#     figure()
#     E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10.0*np.log10(nf2ff_res.Dmax[0])
#     plot(theta, np.squeeze(E_norm[:,0]), 'k-', linewidth=2, label='xz-plane')
#     plot(theta, np.squeeze(E_norm[:,1]), 'r--', linewidth=2, label='yz-plane')
#     grid()
#     ylabel('Directivity (dBi)')
#     xlabel('Theta (deg)')
#     title('Frequency: {} GHz'.format(f_res/1e9))
#     legend()

# Zin = port.uf_tot/port.if_tot
# figure()
# plot(f/1e9, np.real(Zin), 'k-', linewidth=2, label='$\Re\{Z_{in}\}$')
# plot(f/1e9, np.imag(Zin), 'r--', linewidth=2, label='$\Im\{Z_{in}\}$')
# grid()
# legend()
# ylabel('Zin (Ohm)')
# xlabel('Frequency (GHz)')

# show()

