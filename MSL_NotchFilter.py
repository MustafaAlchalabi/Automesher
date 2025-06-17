### Import Libraries
import os
import numpy as np
from CSXCAD  import ContinuousStructure
from openEMS import openEMS
import openEMS.physical_constants as PC
import matplotlib.pyplot as plt


### Setup the simulation
Sim_Path = os.path.abspath( os.path.join('simData', 'MSL_NotchFilter') )
post_proc_only = False

unit = 1e-6 # specify everything in um
MSL_length = 50000.0
MSL_width = 600.0
substrate_thickness = 254.0
substrate_epr = 3.66
stub_length = 12e3
f_max = 7e9

MSL_Zc = 44.51

R0 = 50.0

### Setup FDTD parameters & excitation function
FDTD = openEMS(MaxTime=10e-9)
FDTD.SetGaussExcite( f_max/2, f_max/2 )
FDTD.SetBoundaryCond( ['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'] )

### Setup Geometry & Mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution_0 = PC.C0/f_max/50 /unit
resolution = resolution_0/np.sqrt(substrate_epr)

## Do manual meshing
mesh.AddLine('x', 0)
mesh.AddLine('x',  MSL_width/2)
mesh.AddLine('x', -MSL_width/2)
mesh.SmoothMeshLines('x', resolution/4)

mesh.AddLine('x', [-MSL_length, MSL_length])
mesh.SmoothMeshLines('x', resolution)

mesh.AddLine('y', 0)
mesh.AddLine('y',  MSL_width/2)
mesh.AddLine('y', -MSL_width/2)
mesh.SmoothMeshLines('y', resolution/4)

mesh.AddLine('y', [-15*MSL_width, 15*MSL_width+stub_length])
mesh.AddLine('y', (MSL_width/2+stub_length))
mesh.SmoothMeshLines('y', resolution)

mesh.AddLine('z', np.linspace(0,substrate_thickness,5))
mesh.AddLine('z', 3000)
mesh.SmoothMeshLines('z', resolution)

## Add the substrate
substrate = CSX.AddMaterial('RO4350B', epsilon=substrate_epr)
start = [-MSL_length, -15*MSL_width, 0]
stop  = [+MSL_length, +15*MSL_width+stub_length, substrate_thickness]
substrate.AddBox(start, stop, priority=100)

## MSL line and stub
port = [None, None]
pec = CSX.AddMetal('PEC')
start = [-MSL_length, -MSL_width/2, substrate_thickness]
stop  = [ MSL_length,  MSL_width/2, substrate_thickness]
pec.AddBox(start, stop, priority=200)

start = [-MSL_width/2,  MSL_width/2, substrate_thickness]
stop  = [ MSL_width/2,  MSL_width/2+stub_length, substrate_thickness]
pec.AddBox(start, stop, priority=200)

## Ports
port_start = [mesh.GetLines('x')[10], -MSL_width/2, substrate_thickness]
port_stop  = [mesh.GetLines('x')[10],  MSL_width/2, 0.0]
port[0] = FDTD.AddLumpedPort(1, MSL_Zc, port_start, port_stop, 'z', 1.0, priority=900)

port_start = [mesh.GetLines('x')[-10], -MSL_width/2, substrate_thickness]
port_stop  = [mesh.GetLines('x')[-10],  MSL_width/2, 0.0]
port[1] = FDTD.AddLumpedPort(2, MSL_Zc, port_start, port_stop, 'z', 0.0, priority=900)

### Field Dump
Et = CSX.AddDump('Et', file_type=0, sub_sampling=[2,2,2])
start = [mesh.GetLines('x')[ 0]] + [mesh.GetLines('y')[ 0]] + [substrate_thickness/2]
stop  = [mesh.GetLines('x')[-1]] + [mesh.GetLines('y')[-1]] + [substrate_thickness/2]
Et.AddBox(start, stop)


### Run the simulation
if 1:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'notch.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))


if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True)

### Post-processing and plotting
f = np.linspace( 1e6, f_max, 1601 )
for p in port:
    p.CalcPort(Sim_Path, f, ref_impedance = R0)

s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc

fig, axes = plt.subplots()

axes.plot(f/1e9,20*np.log10(abs(s11)),'k-',linewidth=2 , label='$S_{11}$')
axes.grid()
axes.plot(f/1e9,20*np.log10(abs(s21)),'r--',linewidth=2 , label='$S_{21}$')
axes.legend()
axes.set_ylabel('S-Parameter (dB)')
axes.set_xlabel('frequency (GHz)')
axes.set_ylim([-40, 2])

plt.show()
