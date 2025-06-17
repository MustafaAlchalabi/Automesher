### Import Libraries
import os
import numpy as np
import CSXCAD
from openEMS import openEMS
import openEMS.physical_constants as PC
import matplotlib.pyplot as plt

from Automesher import Automesher

class AutoMeshFDTDWrapper:
    def __init__(self, FDTD, AutoMesh):
        self.FDTD = FDTD
        self.AutoMesh = AutoMesh

    def __getattr__(self, name):
        if name == "SetCSX":
            def wrapper(CSX, *args, **kwargs):
                print("Wrapped:", name)
                out = self.FDTD.SetCSX(CSX.CSX, *args, **kwargs)
                return out
            return wrapper
        elif name == "AddLumpedPort":
            def wrapper(*args, **kwargs):
                print("Wrapped:", name)
                out = getattr(self.FDTD, name)(*args, **kwargs)
                self.AutoMesh.primitives_mesh_setup[out] = self.AutoMesh.mesh_hint_common
                return out
            return wrapper
        else:
            return getattr(self.FDTD, name) 

class AutoMeshPropertyWrapper:
    def __init__(self, Property, AutoMesh):
        self.Property = Property
        self.AutoMesh = AutoMesh

    def __getattr__(self, name):
        if name == "AddBox":
            def wrapper(*args, **kwargs):
                print("Wrapped:", name)
                out = getattr(self.Property, name)(*args, **kwargs)
                self.AutoMesh.primitives_mesh_setup[out] = self.AutoMesh.mesh_hint_common
                return out
            return wrapper
        else:
            return getattr(self.Property, name)

class AutoMeshCSXWrapper:
    def __init__(self, CSX, AutoMesh):
        self.CSX = CSX
        self.AutoMesh = AutoMesh

    def __getattr__(self, name):
        if name in ["AddMaterial", "AddMetal"]:
            def wrapper(*args, **kwargs):
                print("Wrapped:", name)
                out = getattr(self.CSX, name)(*args, **kwargs)
                return AutoMeshPropertyWrapper(out, self.AutoMesh)
            return wrapper
        else:
            return getattr(self.CSX, name)

### Setup the simulation
Sim_Path = os.path.abspath( os.path.join('simData', 'MSL_NotchFilter_autoMesher_decorators') )
post_proc_only = False

unit = 1e-6 # specify everything in um
MSL_length = 50000
MSL_width = 600
substrate_thickness = 254
substrate_epr = 3.66
stub_length = 12e3
f_max = 7e9

MSL_Zc = 44.51

R0 = 50.0

### Grid and AutoMesher Setup
resolution_0 = PC.C0/f_max/50 /unit
resolution = resolution_0/np.sqrt(substrate_epr)

AutoMesh = Automesher()
AutoMesh.primitives_mesh_setup = {}
AutoMesh.properties_mesh_setup = {}
AutoMesh.global_mesh_setup = {
    'dirs': 'xyz',
    # 'refined_cellsize': 1,
    # 'min_cellsize': 1,
    'drawing_unit': unit,
    'start_frequency': 0.0,
    'stop_frequency': f_max,
    # 'mesh_resolution': 'very_high',
    'max_cellsize': resolution,
    'metal_edge_res': None
}
AutoMesh.mesh_hint_common = {}

### Setup FDTD parameters & excitation function
FDTD = openEMS()
FDTD = AutoMeshFDTDWrapper(FDTD, AutoMesh)
FDTD.SetGaussExcite( f_max/2, f_max/2 )
FDTD.SetBoundaryCond( ['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'] )

### Setup Geometry & Mesh
CSX = CSXCAD.ContinuousStructure()
CSX = AutoMeshCSXWrapper(CSX, AutoMesh)
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

## Add bounding sim_box
sim_box = CSX.AddMaterial('sim_box', epsilon=1)
start = [-MSL_length, -15*MSL_width,             0]
stop  = [+MSL_length, +15*MSL_width+stub_length, 3000]
obj = sim_box.AddBox(start, stop, priority=0)

## Add the substrate
substrate = CSX.AddMaterial('RO4350B', epsilon=substrate_epr)
start = [-MSL_length, -15*MSL_width,             0]
stop  = [+MSL_length, +15*MSL_width+stub_length, substrate_thickness]
obj = substrate.AddBox(start, stop, priority=100)

## MSL line and stub
port = [None, None]
pec = CSX.AddMetal('PEC')
start = [-MSL_length, -MSL_width/2, substrate_thickness]
stop  = [ MSL_length,  MSL_width/2, substrate_thickness]
obj = pec.AddBox(start, stop, priority=200)

start = [-MSL_width/2,  MSL_width/2, substrate_thickness]
stop  = [ MSL_width/2,  MSL_width/2+stub_length, substrate_thickness]
obj = pec.AddBox(start, stop, priority=200)

## Ports
port_start = [-MSL_length+10*resolution, -MSL_width/2, substrate_thickness]
port_stop  = [-MSL_length+10*resolution,  MSL_width/2, 0.0]
port[0] = FDTD.AddLumpedPort(1, MSL_Zc, port_start, port_stop, 'z', 1.0, priority=900)

port_start = [ MSL_length-10*resolution, -MSL_width/2, substrate_thickness]
port_stop  = [ MSL_length-10*resolution,  MSL_width/2, 0.0]
port[1] = FDTD.AddLumpedPort(2, MSL_Zc, port_start, port_stop, 'z', 0.0, priority=900)


### Create auto mesh
AutoMesh.GenMesh(CSX, AutoMesh.global_mesh_setup, AutoMesh.primitives_mesh_setup, AutoMesh.properties_mesh_setup)

# Manual overwride since z dir not meshed correctly
mesh = CSX.GetGrid()
mesh.ClearLines('z')
mesh.AddLine('z', np.linspace(0,substrate_thickness,5))
mesh.AddLine('z', 3000)
mesh.SmoothMeshLines('z', resolution)

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
    p.CalcPort( Sim_Path, f, ref_impedance = 50)

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
