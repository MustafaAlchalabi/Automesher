### Import Libraries
import os
# os.environ["OPENEMS_INSTALL_PATH"] = 'H:/Desktop/openEMS'                                          # immer benutzen 

from pylab import *
from CSXCAD import ContinuousStructure
sys.path.append(os.path.join(os.path.dirname(__file__), 'automesher_tools'))
from openEMS import openEMS
from openEMS.physical_constants import *
from CSXCAD.SmoothMeshLines import SmoothMeshLines
from openEMS.physical_constants import C0
from automesher_tools.automesher_main import GenerateMesh, enhance_csx_for_auto_mesh, enhance_FDTD_for_auto_mesh

### Setup the simulation
Sim_Path = os.path.realpath(os.path.join('.', 'MUT_Taro'))


post_proc_only = True
preview_only = False

unit = 1e-6

# Materialeigenschften 
MUT_epsilon_r = 2.0
MUT_kappa = 1e-12
MUT_mu_r = 1

# frequency range of interest
f_start = 26.5e9
f_0     = 30e9
f_stop  = 40e9
lambda0 = C0/f_0  / unit
lambda_MUT = C0/(f_0*sqrt(MUT_epsilon_r * MUT_mu_r))  / unit 

# waveguide dimensions
# WR28
waveguide_a = 7112  # x-Ausdehnung
waveguide_b = 3556  # y-Ausdehnung
waveguide_wall_tx = (9200 - waveguide_a)/2 # xy-Ausdehnung der Wand
waveguide_wall_ty = (5650 - waveguide_b)/2 # xy-Ausdehnung der Wand
port_l = 40000 # z-Ausdehnung

# Definetion der Material
MUT_width = 15000  # Materiallänge in x-Richtung
MUT_height = 15000  # Materiallänge in y-Richtung
MUT_thickness = 2000  # Materiallänge in z-Richtung

# Flange
flange_a = 19000  # x-Ausdehnung
flange_b = 19000  # y-Ausdehnung
flange_t = 4200  # z-Ausdehnung


# Waveguide TE-mode definition
TE_mode = 'TE10'

# Targeted mesh resolution
mesh_res = lambda0 / 15                                                                            
mesh_res_MUT = lambda_MUT / 15
ds = mesh_res_MUT/3  # Small distance for mesh refinement

### Setup FDTD parameter & excitation function 

global_mesh_setup = {
    'dirs': 'xyz',
    'drawing_unit': unit,
    'start_frequency': f_start,
    'stop_frequency': f_stop,
    'mesh_resolution': 'high',
    'boundary_distance': [lambda0, lambda0, None, None, None, None], # value, 'auto' or None
    # 'refined_cellsize': mesh_res_MUT,
    # 'max_cellsize': mesh_res*15,
    # 'min_cellsize': mesh_res/2,
    # 'f0': f_0,
    # 'num_lines': 3
}
primitives_mesh_setup = {}

FDTD = openEMS(NrTS=1e5);                                                                        # NrTs:Number of Time Steps
FDTD.SetGaussExcite(0.5*(f_start+f_stop),0.5*(f_stop-f_start))                                   #  Gauss-Impulsfunktion (Gauss Excitation) als Anregung für die Simulation fest

# boundary conditions

FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

   # ‘PEC’ : perfect electric conductor (default)
   # ‘PMC’ : perfect magnetic conductor, useful for symmetries

   # ‘MUR’ : simple MUR absorbing boundary conditions
   # ‘PML-8’ : PML absorbing boundary conditions

### Setup geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)                                                                                # Verknüpfen der Struktur mit der FDTD-Simulation
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

CSX = enhance_csx_for_auto_mesh(CSX, primitives_mesh_setup)                 # enhance CSX for auto-meshing
FDTD = enhance_FDTD_for_auto_mesh(FDTD, primitives_mesh_setup)               # enhance FDTD for auto-meshing
# Apply the waveguide port
ports = []
# Port 1 (Waveguide Excitation) im negativen z-Achsenbereich
start = [-waveguide_a/2, -waveguide_b/2, -port_l - MUT_thickness/2 + 10*mesh_res]
stop =  [ waveguide_a/2,  waveguide_b/2, -port_l - MUT_thickness/2 + 15*mesh_res]
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])
port1= FDTD.AddRectWaveGuidePort(0, start, stop, 'z', waveguide_a * unit, waveguide_b * unit, TE_mode, 1)      # Die Leistung hier ist ein 1 KW (letzte Argument)
ports.append(port1)     # Die Leistung hier ist ein 1 KW (letzte Argument)
# primitives_mesh_setup[port1] = mesh_hint

# Port 2 (Waveguide Excitation) im positiven z-Achsenbereich
start = [-waveguide_a/2, -waveguide_b/2,  port_l + MUT_thickness/2 - 10*mesh_res]
stop =  [ waveguide_a/2,  waveguide_b/2,  port_l + MUT_thickness/2 - 15*mesh_res]
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])
port2 = FDTD.AddRectWaveGuidePort(1, start, stop, 'z', waveguide_a * unit, waveguide_b * unit, TE_mode, 0)  
ports.append(port2)
# primitives_mesh_setup[port2] = mesh_hint

# Hohlleiter im negativen z-Achsenbereich
Kupfer1 = CSX.AddMetal('Kupfer1')
start = [-waveguide_a/2 - waveguide_wall_tx/2, -waveguide_b/2 - waveguide_wall_ty/2, -port_l - MUT_thickness/2]
stop =  [ waveguide_a/2 + waveguide_wall_tx/2,  waveguide_b/2 + waveguide_wall_ty/2,         - MUT_thickness/2]
Box1 = Kupfer1.AddBox(priority=100, start=start, stop=stop)
# primitives_mesh_setup[Box1] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])

start = [-flange_a/2, -flange_b/2, -flange_t - MUT_thickness/2]
stop =  [ flange_a/2,  flange_b/2,           - MUT_thickness/2]
Box2 = Kupfer1.AddBox(priority=100, start=start, stop=stop)
# primitives_mesh_setup[Box2] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])


air = CSX.AddMaterial('air', epsilon=1, kappa=0)
start = [-waveguide_a/2, -waveguide_b/2, -port_l - MUT_thickness/2]
stop =  [ waveguide_a/2,  waveguide_b/2,         - MUT_thickness/2]
Box3 = air.AddBox(priority=110, start=start, stop = stop)
# primitives_mesh_setup[Box3] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])
# mesh.AddLine('x', [start[0]-ds, stop[0]-ds, start[0]+ds, stop[0]+ds])
# mesh.AddLine('y', [start[1]-ds, stop[1]-ds, start[1]+ds, stop[1]+ds])
# mesh.AddLine('z', [stop[2]-ds, stop[2]+ds])


# Hohlleiter im positiven z-Achsenbereich
Kupfer1 = CSX.AddMetal('Kupfer1')
start = [-waveguide_a/2 - waveguide_wall_tx/2, -waveguide_b/2 - waveguide_wall_ty/2,  port_l + MUT_thickness/2]
stop =  [ waveguide_a/2 + waveguide_wall_tx/2,  waveguide_b/2 + waveguide_wall_ty/2,         + MUT_thickness/2]
Box4 = Kupfer1.AddBox(priority=100, start=start, stop=stop)
# primitives_mesh_setup[Box4] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])

start = [-flange_a/2, -flange_b/2,  flange_t + MUT_thickness/2]
stop =  [ flange_a/2,  flange_b/2,           + MUT_thickness/2]
Box5 = Kupfer1.AddBox(priority=100, start=start, stop=stop)
# primitives_mesh_setup[Box5] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])


air = CSX.AddMaterial('air', epsilon=1, kappa=0)
start = [-waveguide_a/2, -waveguide_b/2,  port_l + MUT_thickness/2]
stop =  [ waveguide_a/2,  waveguide_b/2,         + MUT_thickness/2]
Box6 = air.AddBox(priority=110, start=start, stop = stop)
# primitives_mesh_setup[Box6] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])
# mesh.AddLine('x', [start[0]-ds, stop[0]-ds, start[0]+ds, stop[0]+ds])
# mesh.AddLine('y', [start[1]-ds, stop[1]-ds, start[1]+ds, stop[1]+ds])
# mesh.AddLine('z', [stop[2]-ds, stop[2]+ds])

# MUT sample
MUT_material = CSX.AddMaterial('MUT_material', epsilon=MUT_epsilon_r, kappa=MUT_kappa)
start = [-MUT_width/2, -MUT_height/2, -MUT_thickness/2]
stop =  [ MUT_width/2,  MUT_height/2,  MUT_thickness/2]
Box7 = MUT_material.AddBox(priority=100, start=start, stop=stop)
# primitives_mesh_setup[Box7] = mesh_hint
# mesh.AddLine('x', [start[0], stop[0]])
# mesh.AddLine('y', [start[1], stop[1]])
# mesh.AddLine('z', [start[2], stop[2]])
# mesh.AddLine('x', [start[0]-ds, stop[0]-ds, start[0]+ds, stop[0]+ds])
# mesh.AddLine('y', [start[1]-ds, stop[1]-ds, start[1]+ds, stop[1]+ds])

# mesh.AddLine('x', [-19000-lambda0/2, 19000+lambda0/2])
# mesh.AddLine('y', [-19000-lambda0/2, 19000+lambda0/2])

properties_mesh_setup = {}
GenerateMesh(CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup)


### Run the simulation
if 1:                                                                       # debugging only
    CSX_file = os.path.join(Sim_Path, 'MUT_Taro.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

if preview_only:
    quit()  
if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True )
    


### Postprocessing & plotting
freq = linspace(f_start,f_stop,201)
for port in ports:
    port.CalcPort(Sim_Path, freq)

s11 = ports[0].uf_ref / ports[0].uf_inc
s21 = ports[1].uf_ref / ports[0].uf_inc
# s12 = ports[0].uf_ref / ports[1].uf_inc                                                             # um symetrie zu überprüfen
# s22 = ports[1].uf_ref / ports[1].uf_inc  
ZL  = ports[0].uf_tot / ports[0].if_tot
ZL_a = ports[0].ZL # analytic waveguide impedance



## Plot S-parameter
figure()
plot(freq*1e-6,20*log10(abs(s11)),'k-',linewidth=2, label='$S_{11}$')
grid()
plot(freq*1e-6,20*log10(abs(s21)),'r--',linewidth=2, label='$S_{21}$')
# plot(freq * 1e-6, 20 * log10(abs(s12)), 'b-.', linewidth=2, label='$S_{12}$') #
# plot(freq * 1e-6, 20 * log10(abs(s22)), 'g:', linewidth=2, label='$S_{22}$')   #
legend()
ylabel('S-Parameter (dB)')
xlabel(r'frequency (MHz) $\rightarrow$')

## Compare analytic and numerical wave-impedance
figure()
plot(freq*1e-6,real(ZL), linewidth=2, label='$\Re\{Z_L\}$')
grid()
plot(freq*1e-6,imag(ZL),'r--', linewidth=2, label='$\Im\{Z_L\}$')
plot(freq*1e-6,ZL_a,'g-.',linewidth=2, label='$Z_{L, analytic}$')
ylabel('ZL $(\Omega)$')
xlabel(r'frequency (MHz) $\rightarrow$')
legend()



# show()
