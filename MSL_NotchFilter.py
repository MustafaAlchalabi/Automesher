import os, tempfile
from pylab import *
sys.path.append(os.path.join(os.path.dirname(__file__), 'automesher_tools'))
from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
from automesher_tools.automesher_main import GenerateMesh, enhance_csx_for_auto_mesh, enhance_FDTD_for_auto_mesh


Sim_Path = os.path.join(tempfile.gettempdir(), 'NotchFilter')
post_proc_only = True

unit = 1e-6 # specify everything in um
MSL_length = 50000
MSL_width = 600
substrate_thickness = 254
substrate_epr = 3.66
stub_length = 12e3
f_max = 7e9

FDTD = openEMS()
FDTD.SetGaussExcite( f_max/2, f_max/2 )
FDTD.SetBoundaryCond( ['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'] )

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
resolution = C0/(f_max*sqrt(substrate_epr))/unit/50 # resolution of lambda/50

primitives_mesh_setup = {}
properties_mesh_setup = {}
global_mesh_setup = {
    'dirs': 'xyz',
    'drawing_unit': unit,
    'start_frequency': 0,
    'stop_frequency': f_max,
    'mesh_resolution': 'medium',
    'boundary_distance': [None, None, None, None, 3000, None], # value, 'auto' or None
    # 'refined_cellsize': mesh_res_MUT,
    # 'max_cellsize': mesh_res*15,
    # 'min_cellsize': mesh_res/2,
    # 'f0': f_0,
    # 'num_lines': 3
}

CSX = enhance_csx_for_auto_mesh(CSX, primitives_mesh_setup)
FDTD = enhance_FDTD_for_auto_mesh(FDTD, primitives_mesh_setup)


substrate = CSX.AddMaterial( 'RO4350B', epsilon=substrate_epr)
start = [-MSL_length, -15*MSL_width, 0]
stop  = [+MSL_length, +15*MSL_width+stub_length, substrate_thickness]
substrate.AddBox(start, stop )

pec = CSX.AddMetal( 'PEC' )
start = [-MSL_width/2,  MSL_width/2, substrate_thickness]
stop  = [ MSL_width/2,  MSL_width/2+stub_length, substrate_thickness]
pec.AddBox(start, stop, priority=10 )

# GenerateMesh(CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup)

third_mesh = array([2*resolution/3, -resolution/3])/4
# mesh.AddLine('x', 0)
# mesh.AddLine('x',  MSL_width/2+third_mesh)
# mesh.AddLine('x', -MSL_width/2-third_mesh)
mesh.AddLine('x', np.linspace(-MSL_length, MSL_length,6))
port = [None, None]
portstart = [ -MSL_length, -MSL_width/2, substrate_thickness]
portstop  = [ 0,  MSL_width/2, 0]
port[0] = FDTD.AddMSLPort( 1,  pec, portstart, portstop, 'x', 'z', excite=-1, FeedShift=10*resolution, MeasPlaneShift=MSL_length/3, priority=10)

portstart = [MSL_length, -MSL_width/2, substrate_thickness]
portstop  = [0         ,  MSL_width/2, 0]
port[1] = FDTD.AddMSLPort( 2, pec, portstart, portstop, 'x', 'z', MeasPlaneShift=MSL_length/3, priority=10 )

GenerateMesh(CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup)






if 1:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'notch.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))


if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True)




    f = linspace( 1e6, f_max, 1601 )
for p in port:
    p.CalcPort( Sim_Path, f, ref_impedance = 50)

s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc

plot(f/1e9,20*log10(abs(s11)),'k-',linewidth=2 , label='$S_{11}$')
grid()
plot(f/1e9,20*log10(abs(s21)),'r--',linewidth=2 , label='$S_{21}$')
legend()
ylabel('S-Parameter (dB)')
xlabel('frequency (GHz)')
ylim([-40, 2])

# show()