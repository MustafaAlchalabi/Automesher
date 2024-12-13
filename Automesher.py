import sys
import numpy as np

from CSXCAD import CSPrimitives
from CSXCAD import CSProperties
from CSXCAD.Utilities import CheckNyDir, GetMultiDirs
from openEMS.physical_constants import C0
from CSXCAD.SmoothMeshLines import SmoothMeshLines

# from CSXCAD.CSPrimitives import primitives_mesh_setup
# from CSXCAD.CSProperties import properties_mesh_setup

   
class Automesher:
    def __init__(self):
        self.properties_mesh_setup = {}
        self.primitives_mesh_setup = {}
        self.global_mesh_setup = {} 
        self.edge_data = {}

    def GenMesh(self, CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup, **kw):

        self.properties_mesh_setup = properties_mesh_setup
        self.primitives_mesh_setup = primitives_mesh_setup
        self.global_mesh_setup = global_mesh_setup


        """ Beispiel Darstellung vom dict: 
            if 'mesh_hint' in kw:
                mesh_hint = kw.pop('mesh_hint')
                for key, value in mesh_hint.items():
                    properties_mesh_setup[(self, key)] = value
            else:
                properties_mesh_setup[(self, '')] = ''      

            primitives_mesh_setup = {(<CSXCAD.CSPrimitives.CSPrimBox object at 0x7f7fedeba740>, 'dirs'): [0, 1, 2],
                                        (<CSXCAD.CSPrimitives.CSPrimBox object at 0x7f7fedeba740>, 'metal_edge_res'): 0.5, 
                                        (<CSXCAD.CSPrimitives.CSPrimBox object at 0x7f7fedeba9c0>, 'dirs'): [2, 2, 2]}
            usefull commands: 
                list(primitives_mesh_setup.keys())[0] : (<CSXCAD.CSPrimitives.CSPrimBox object at 0x7f7fedeba740>, 'dirs')
                list(primitives_mesh_setup.keys())[0][0] : <CSXCAD.CSPrimitives.CSPrimBox object at 0x7f7fedeba740>
                first_Key=list(primitives_mesh_setup.keys())[0]
                primitives_mesh_setup[first_Key]: [0, 1, 2]
                first_Key[0].GetStart(): array([ -4.32, -11.78,   1.6 ]) ## start point of the box
                unique_properties = list({key[0] for key in properties_mesh_setup.keys()})
                dirs = primitives_mesh_setup.get((polygon, 'dirs'), None)
        """
        csx = CSX
        unique_primitives = list(self.primitives_mesh_setup.keys())
        # print(unique_primitives)
        unique_properties = list(self.properties_mesh_setup.keys())
        # print(unique_properties)
        # print(self.primitives_mesh_setup)

        if csx is None:
            raise Exception('Unable to access CSX!')
        
        grid = csx.GetGrid()
        for prim in unique_primitives:
            self.collect_edge_data(prim, grid, **kw)
            # hint = self.mesh_hint_from_primitive(prim, **kw)
            # if hint is None:
            #     continue
            # for n in range(3):
            #     if hint[n] is None:
            #         continue
            #     grid.AddLine(n, hint[n])
        self.create_mesh_lines(grid)

    def collect_edge_data(self, primitive, grid, **kw):

        hint = None
        if primitive.GetType() == CSPrimitives.POINT:
            hint = self.mesh_hint_from_point(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.BOX:
            hint = self.mesh_hint_from_box(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.POLYGON:
            (hint,dirs,metal_edge_res) = self.mesh_hint_from_polygon(primitive, grid, **kw)
        
        if hint is not None:
            self.edge_data[primitive] = (hint,dirs,metal_edge_res)
        # print(self.edge_data)
    def create_mesh_lines(self, grid):

        for primitive in self.edge_data:
            for n in range(3):
                if self.edge_data[primitive][0][n] is not None:
                    grid.AddLine(n, self.edge_data[primitive][0][n])

    def mesh_combine(mesh1, mesh2, sort=True):
        mesh = [None, None, None]
        for ny in range(3):
            if mesh1[ny] is None and mesh2[ny] is None:
                continue
            elif mesh1[ny] is None:
                mesh[ny] = mesh2[ny]
            elif mesh2[ny] is None:
                mesh[ny] = mesh1[ny]
            else:
                mesh[ny] = list(sorted(mesh1[ny] + mesh2[ny]))
        return mesh

    # def mesh_hint_from_primitive(self, primitive, **kw):(unique_properties)
        # print(self.primi
    #     if primitive.GetType() == CSPrimitives.POINT:
    #         return self.mesh_hint_from_point(primitive, **kw)
    #     if primitive.GetType() == CSPrimitives.BOX:
    #         return self.mesh_hint_from_box(primitive, **kw)
    #     if primitive.GetType() == CSPrimitives.POLYGON:
    #         return self.mesh_hint_from_polygon(primitive, **kw)
    #     else:
    #         return None

    def mesh_hint_from_point(self, point, **kw):
        """ mesh_hint_from_point(point, dirs)

        Get a grid hint for the coordinates of the point

        :param dirs: str -- 'x','y','z' or 'xy', 'yz' or 'xyz' or 'all'
        :param mesh: combine mesh hint to existing mesh
        :returns: (3,) list of mesh hints
        """
        hint = [None, None, None]
        coord = point.GetCoord()
        dirs = self.primitives_mesh_setup.get((point, 'dirs'), None)
        for ny in GetMultiDirs(dirs):
            hint[ny] = [coord[ny],]
        if 'mesh' in kw:
            return self.mesh_combine(hint, kw['mesh'])
        return hint

    def mesh_hint_from_box(self, box, **kw):
        """ mesh_hint_from_box(box, dirs, metal_edge_res=None, **kw)

        Get a grid hint for the edges of the given box with an an optional 2D metal
        edge resolution.

        :param dirs: str -- 'x','y','z' or 'xy', 'yz' or 'xyz' or 'all'
        :param metal_edge_res: float -- 2D flat edge resolution
        :param up_dir: bool -- Enable upper edge
        :param down_dir: bool -- Enable lower edge
        :param mesh: combine mesh hint to existing mesh
        :returns: (3,) list of mesh hints
        """
        dirs = self.primitives_mesh_setup.get((box, 'dirs'), None)
        metal_edge_res = self.primitives_mesh_setup.get((box, 'metal_edge_res'), None)
        up_dir   = kw.get('up_dir'  , True)
        down_dir = kw.get('down_dir', True)

        if metal_edge_res is None:
            mer = 0
        else:
            mer = np.array([-1.0, 2.0])/3 * metal_edge_res
        if box.HasTransform():
            sys.stderr.write('FDTD::automesh: Warning, cannot add edges to grid with transformations enabled\n')
            return
        hint = [None, None, None]
        start = np.fmin(box.GetStart(), box.GetStop())
        stop  = np.fmax(box.GetStart(), box.GetStop())
        if dirs is not None:
            for ny in GetMultiDirs(dirs):
                hint[ny] = []
                if metal_edge_res is not None and stop[ny]-start[ny]>metal_edge_res:
                    if down_dir:
                        hint[ny].append(start[ny]-mer[0])
                        hint[ny].append(start[ny]-mer[1])
                    if up_dir:
                        hint[ny].append(stop[ny]+mer[0])
                        hint[ny].append(stop[ny]+mer[1])
                elif stop[ny]-start[ny]:
                    if down_dir:
                        hint[ny].append(start[ny])
                    if up_dir:
                        hint[ny].append(stop[ny])
                else:
                    hint[ny].append(start[ny])
        else:
            hint = [None, None, None]

        if 'mesh' in kw:
            return self.mesh_combine(hint, kw['mesh'])
        return hint

    def mesh_hint_from_polygon(self, polygon, grid, **kw):
        
        dirs = self.primitives_mesh_setup.get(polygon).get('dirs', None)
        if dirs is None:
            dirs = self.properties_mesh_setup.get(polygon.GetProperty()).get('dirs', None)
        if dirs is None:
            dirs = self.global_mesh_setup.get('dirs', None)
        metal_edge_res = self.primitives_mesh_setup.get(polygon).get('metal_edge_res', None)
        if metal_edge_res is None:
            metal_edge_res = self.properties_mesh_setup.get(polygon.GetProperty()).get('metal_edge_res', None)
        if metal_edge_res is None:
            metal_edge_res = self.global_mesh_setup.get('metal_edge_res', None)
        mesh_res = self.global_mesh_setup.get('mesh_resolution', None)
        if metal_edge_res is None:
            mer = 0
        else:
            mer = np.array([-1.0, 2.0])/3 * metal_edge_res

        N = polygon.GetQtyCoords()
        hint = [[], [], None]
        x = polygon.GetCoords()[0]
        y = polygon.GetCoords()[1]

        min_distance_x = float('inf')
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                distance = abs(x[i] - x[j])
                if distance > 0 and distance < min_distance_x:
                    min_distance_x = distance
        min_distance_y = float('inf')
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                distance = abs(y[i] - y[j])
                if distance > 0 and distance < min_distance_y:
                    min_distance_y = distance
        xedges = []
        yedges = []
        otheredges = []
        if dirs is not None:
            for i in range(N - 1):
                if x[i] == x[i + 1]:
                    xedges.append([x[i], y[i], y[i + 1]])
                    # if metal_edge_res is not None:
                    #     if self.yline_in_polygon(polygon.GetCoords(), x[i]+min_distance_x/2, y[i], y[i+1]) and not self.yline_in_polygon(polygon.GetCoords(), x[i]-min_distance_x/2, y[i], y[i+1]):
                    #         hint[0].append(x[i] - mer[0])
                    #         hint[0].append(x[i] - mer[1])
                    #     elif self.yline_in_polygon(polygon.GetCoords(), x[i]-min_distance_x/2, y[i], y[i+1]) and self.yline_in_polygon(polygon.GetCoords(), x[i]+min_distance_x/2, y[i], y[i+1]):
                    #         continue
                    #     else:
                    #         hint[0].append(x[i] + mer[0])
                    #         hint[0].append(x[i] + mer[1])
                    # else:
                    #     if self.yline_in_polygon(polygon.GetCoords(), x[i]+min_distance_x/2, y[i], y[i+1]) and self.yline_in_polygon(polygon.GetCoords(), x[i]-min_distance_x/2, y[i], y[i+1]):
                    #         continue
                    #     else:
                    #         hint[0].append(x[i])

                if y[i] == y[i + 1]:
                    yedges.append([y[i], x[i], x[i + 1]])
                    # if metal_edge_res is not None:
                    #     if self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]+min_distance_y/2) and not self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]-min_distance_y/2):
                    #         hint[1].append(y[i] - mer[0])
                    #         hint[1].append(y[i] - mer[1])
                    #     elif self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]+min_distance_y/2) and self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]-min_distance_y/2):
                    #         continue
                    #     else:
                    #         hint[1].append(y[i] + mer[0])
                    #         hint[1].append(y[i] + mer[1])
                    # else:
                    #     if self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]+min_distance_y/2) and self.xline_in_polygon(polygon.GetCoords(), x[i], x[i+1], y[i]-min_distance_y/2):
                    #         continue
                    #     else:
                    #         hint[1].append(y[i])
                if x[i] != x[i + 1] and y[i] != y[i + 1]:
                    otheredges.append([x[i], x[i+1], y[i], y[i+1]])
                    # Calculate the slope
                    # m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    
                    # # Determine the number of points based on the slope
                    # if abs(m) > 1:  # Steeper line
                    #     y_vals = SmoothMeshLines([y[i], y[i + 1]], mesh_res)
                    #     x_vals = np.linspace(x[i], x[i + 1], 5)
                    #     if np.min(abs(np.diff(x_vals))) > mesh_res:
                    #         x_vals =[]
                    #         x_vals = SmoothMeshLines([x[i], x[i + 1]], mesh_res)
                    #     # y_vals = np.append(y_vals, (y[i] + y[i + 1]) / 2)                        # num_x_lines = int(abs_m * 2) + 1  # More y points
                    #     # num_y_lines = int(num_x_lines / abs_m)  # Fewer x points
                    # else:  # Less steep line
                    #     x_vals = SmoothMeshLines([x[i], x[i + 1]], mesh_res)
                    #     y_vals = np.linspace(y[i], y[i + 1], 5)
                    #     if np.min(abs(np.diff(y_vals))) > mesh_res:
                    #         y_vals =[]
                    #         y_vals = SmoothMeshLines([y[i], y[i + 1]], mesh_res)

             
                    # # Interpolate points along the line
                    # for k in range(len(x_vals)):
                    #     # if k < len(x_vals) and k < len(y_vals):
                    #     hint[0].append(x_vals[k])
                    # for k in range(len(y_vals)):
                    #     hint[1].append(y_vals[k])
        else:
            hint = [None, None, None]
    # point_in_polygon(polygon.GetCoords(), [(x[i]+x[i+1])/2, y[i]-min_distance_y/2])
        # hint[0] = np.unique(x)
        # hint[1] = np.unique(y)
        # xedges = np.unique(xedges[i][0] for i in range(len(xedges)-1))
        xedges.sort(key=lambda edge: edge[0])
        sorted_xedges = xedges
        yedges.sort(key=lambda edge: edge[0])
        sorted_yedges = yedges
        unique_xedges = np.unique([edge[0] for edge in xedges])
        unique_xedges = np.sort(unique_xedges)
        print(unique_xedges)
        unique_yedges = np.unique([edge[0] for edge in yedges])
        unique_yedges = np.sort(unique_yedges)
        otheredges = np.array(otheredges, dtype=float).tolist()
        print(unique_yedges)
        x_diffs = np.diff(unique_xedges)
        y_diffs = np.diff(unique_yedges)
        other_edges_in_range_x = []
        other_edges_in_range_y = []
        sorted_x = np.sort(x) # type: ignore
        sorted_y = np.sort(y)
        for edge in otheredges:
            x_start, x_end, y_start, y_end = edge
            x_in_range = [x for x in unique_xedges if x_start <= x <= x_end or x_start >= x >= x_end]
            y_in_range = [y for y in unique_yedges if y_start <= y <= y_end or y_start >= y >= y_end]
            # other_edges_in_range_x = [other_edge for other_edge in otheredges if (x_start < other_edge[0] < x_end or x_start < other_edge[1] < x_end or x_start > other_edge[0] > x_end or x_start > other_edge[1] > x_end)]
            # other_edges_in_range_x.append([other_edge for other_edge in otheredges if (x_start < other_edge[0] < x_end or x_start < other_edge[1] < x_end or x_start > other_edge[0] > x_end or x_start > other_edge[1] > x_end)])
            # other_edges_in_range.append([value for other_edge in otheredges if (x_start < other_edge[0] < x_end or x_start < other_edge[1] < x_end or x_start > other_edge[0] > x_end or x_start > other_edge[1] > x_end) for value in other_edge])
            # other_edges_in_range_x.append([[other_edge[0], other_edge[1]] for other_edge in otheredges if (x_start < other_edge[0] < x_end or x_start < other_edge[1] < x_end or x_start > other_edge[0] > x_end or x_start > other_edge[1] > x_end)])
            other_edges_in_range_x.extend([other_edge[0:2] for other_edge in otheredges if (x_start < other_edge[0] < x_end or x_start < other_edge[1] < x_end or x_start > other_edge[0] > x_end or x_start > other_edge[1] > x_end)])
            other_edges_in_range_y.extend([other_edge[2:4] for other_edge in otheredges if (y_start < other_edge[2] < y_end or y_start < other_edge[3] < y_end or y_start > other_edge[2] > y_end or y_start > other_edge[3] > y_end)])
            # other_edges_in_range_y.append([[other_edge[2], other_edge[3]] for other_edge in otheredges if (y_start < other_edge[2] < y_end or y_start < other_edge[3] < y_end or y_start > other_edge[2] > y_end or y_start > other_edge[3] > y_end)])
            if not other_edges_in_range_x:
                if not x_in_range:
                    m = (edge[3] - edge[2]) / (edge[1] - edge[0])
                    if abs(m) > 1:
                        if metal_edge_res is not None:
                            x_vals = np.linspace(edge[0], edge[1], 5)[1:-1]
                        else:
                            x_vals = np.linspace(edge[0], edge[1], 5)
                        if np.min(abs(np.diff(x_vals))) > mesh_res:
                            x_vals = []
                            x_vals = SmoothMeshLines([edge[0], edge[1]], mesh_res)
                        hint[0].extend(x_vals)
                    elif abs(m) < 1:
                        if metal_edge_res is not None:
                            x_vals = SmoothMeshLines([edge[0], edge[1]], mesh_res/2)[1:-1]
                        else:
                            x_vals = SmoothMeshLines(edge[0], edge[1], mesh_res)
                        hint[0].extend(x_vals)
                    else :
                        if metal_edge_res is not None:
                            x_vals = np.linspace(edge[0], edge[1], 5)[1:-1]
                        else:
                            x_vals = np.linspace(edge[0], edge[1], 5)
                        if np.min(abs(np.diff(x_vals))) > mesh_res:
                            x_vals = []
                            x_vals = SmoothMeshLines([edge[0], edge[1]], mesh_res)
                        x_vals=SmoothMeshLines(x_vals, mesh_res/2)
                        hint[0].extend(x_vals)
                if x_in_range:
                    if metal_edge_res is not None:
                        x_min = np.min([np.min(x_in_range), edge[0], edge[1]])
                        x_max = np.max([np.max(x_in_range), edge[0], edge[1]])
                        xlines = np.linspace(x_min, x_max, 5)
                    else:
                        xlines = np.linspace(min(x_in_range),max(x_in_range),5)   
                    xlines=SmoothMeshLines(xlines, mesh_res/2)
                    n = [x for x in xlines if x_min < x < x_max]
                    hint[0].extend(n)
            if not other_edges_in_range_y:
                if not y_in_range:
                    # print('y_in_range', y_in_range)
                    m = (edge[3] - edge[2]) / (edge[1] - edge[0])
                    if abs(m) > 1:
                        if metal_edge_res is not None:
                            y_vals=SmoothMeshLines([edge[2], edge[3]], mesh_res/2)[1:-1]
                            # y_vals = np.linspace(edge[2], edge[3], 5)[1:-1]
                        else:
                            y_vals=SmoothMeshLines([edge[2], edge[3]], mesh_res/2)[1:-1]
                        if np.min(abs(np.diff(y_vals))) > mesh_res:
                            y_vals = []
                            y_vals = SmoothMeshLines([edge[2], edge[3]], mesh_res)
                        hint[1].extend(y_vals)
                    elif abs(m) < 1:
                        if metal_edge_res is not None:
                            # y_vals = SmoothMeshLines([edge[2], edge[3]], mesh_res/4)[1:-1]
                            y_vals = np.linspace(edge[2], edge[3], 5)[1:-1]

                        else:
                            y_vals = SmoothMeshLines([edge[2], edge[3]], mesh_res)
                        hint[1].extend(y_vals)
                    else:
                        if metal_edge_res is not None:
                            y_vals = np.linspace(edge[2], edge[3], 5)[1:-1]
                        else:
                            y_vals = np.linspace(edge[2], edge[3], 5)
                        if np.min(abs(np.diff(y_vals))) > mesh_res:
                            y_vals = []
                            y_vals = SmoothMeshLines([edge[2], edge[3]], mesh_res)
                        y_vals=SmoothMeshLines(y_vals, mesh_res/2)
                        hint[1].extend(y_vals)
        
                if y_in_range:
                    if metal_edge_res is not None:
                        y_min = np.min([np.min(y_in_range), edge[2], edge[3]])
                        y_max = np.max([np.max(y_in_range), edge[2], edge[3]])
                        ylines = np.linspace(y_min, y_max, 5)
                        print ('hi,hi')
                    else:
                        ylines = np.linspace(min(y_in_range),max(y_in_range),5)                   
                    ylines=SmoothMeshLines(ylines, mesh_res/2)
                    n = [y for y in ylines if y_min < y < y_max]
                    hint[1].extend(n)

                x_in_range = []
                y_in_range = []

            if other_edges_in_range_x:
                if any(edge == other_edge for other_edge in other_edges_in_range_x):
                    break
                else:
                    min_x = np.min([x_start,x_end, np.min(np.min(other_edges_in_range_x))])
                    max_x = np.max([x_start,x_end, np.max(np.max(other_edges_in_range_x))])
                    x_in_range = [x for x in unique_xedges if min_x < x < max_x]
                    y_in_range = [y for y in unique_yedges if y_start < y < y_end]
                    if x_in_range:
                        if metal_edge_res is not None:
                            x_in_range.extend([min_x, max_x])    
                            xlines=SmoothMeshLines(x_in_range, mesh_res/2)[1:-1]
                            hint[0].extend(xlines)
                        else:
                            x_in_range.extend([min_x, max_x])    
                            xlines=SmoothMeshLines(x_in_range, mesh_res/2)
                            hint[0].extend(xlines)                            
                    if not x_in_range:
                        if metal_edge_res is not None:               
                            xlines=SmoothMeshLines(xlines, mesh_res/2)
                            hint[0].extend(xlines)
                x_in_range = []

            if other_edges_in_range_y:
                print('other_edges_in_range_y', other_edges_in_range_y)
                if any(edge == other_edge for other_edge in other_edges_in_range_y):
                    break
                else:
                    min_y = np.min([y_start,y_end, np.min(np.min(other_edges_in_range_y))])
                    max_y = np.max([y_start,y_end, np.max(np.max(other_edges_in_range_y))])
                    y_in_range = [y for y in unique_yedges if min_y < y < max_y]
                    x_in_range = [x for x in unique_xedges if x_start < x < x_end]
                    if y_in_range:
                        if metal_edge_res is not None:
                            y_in_range.extend([min_y, max_y])    
                            ylines=SmoothMeshLines(y_in_range, mesh_res/2)
                            hint[1].extend(ylines)
                        else:
                            y_in_range.extend([min_y, max_y])    
                            ylines=SmoothMeshLines(y_in_range, mesh_res/2)
                            hint[1].extend(ylines)                            
                    if not y_in_range:
                        print('y_in_range', y_in_range)
                        if metal_edge_res is not None:               
                            ylines=SmoothMeshLines(ylines, mesh_res/2)
                            hint[1].extend(ylines)
                y_in_range = []
                    
        for i in range(len(unique_xedges) - 1):
            if metal_edge_res is not None:
                if x_diffs[i] <= mesh_res:
                    hint[0].extend(np.linspace(unique_xedges[i], unique_xedges[i + 1], 5)[1:-1])
                else:
                    continue    
            else:
                if x_diffs[i] <= mesh_res:
                    hint[0].extend(np.linspace(unique_xedges[i], unique_xedges[i + 1], 5))
                else:
                    xlines = grid.GetLines('x', do_sort=True)
                    xlines=SmoothMeshLines(xlines, mesh_res/2)
                    n = [x for x in xlines if unique_xedges[i] <= x <= unique_xedges[i + 1]]
                    hint[0].extend(n)

        for i in range(len(unique_yedges) - 1):
            if metal_edge_res is not None:
                if y_diffs[i] <= mesh_res:
                    hint[1].extend(np.linspace(unique_yedges[i], unique_yedges[i + 1], 5)[1:-1])
                else:
                    continue
            else:
                if y_diffs[i] <= mesh_res:
                    hint[1].extend(np.linspace(unique_yedges[i], unique_yedges[i + 1], 5))
                else:
                    ylines = grid.GetLines('y', do_sort=True)
                    ylines=SmoothMeshLines(ylines, mesh_res/2)
                    n = [y for y in ylines if unique_yedges[i] <= y <= unique_yedges[i + 1]]
                    hint[1].extend(n)
                    
        if metal_edge_res is not None:
            if unique_xedges[0] <= sorted_x[0] and unique_xedges[1]-unique_xedges[0] > mesh_res:
                hint[0].append(sorted_xedges[0][0]-mer[0])
            if unique_xedges[-1] >= sorted_x[-1] and unique_xedges[-1]-unique_xedges[-2] > mesh_res:
                hint[0].append(sorted_xedges[-1][0]+mer[0])
            for i in range(len(unique_xedges) - 1):
                if x_diffs[i] > mesh_res:
                    if otheredges:
                        x_in_range = []
                        for other_edge in otheredges:
                            x_in_range = [unique_xedges[i] for other_edge in otheredges if other_edge[0] < unique_xedges[i] < other_edge[1] or other_edge[1] < unique_xedges[i] < other_edge[0]]
                            x_start = unique_xedges[i]
                            x_end = unique_xedges[i + 1]
                            edge_start = [other_edge[0] for other_edge in otheredges]
                            edge_end = [other_edge[1] for other_edge in otheredges]
                            condition1 = any(x_start < es < x_end or x_start > es > x_end for es in edge_start)
                            condition1_es = [es for es in edge_start if x_start < es < x_end or x_start > es > x_end]
                            condition2 = any(x_start < ee < x_end  or x_start > ee > x_end for ee in edge_end)
                            condition2_ee = [ee for ee in edge_end if x_start < ee < x_end or x_start > ee > x_end]
                            condition3 = any((x_start in (es, ee)) and (x_end in (es, ee)) for es, ee in zip(edge_start, edge_end))
                            # print('cindition1_es', condition1_es, 'condition2_ee', condition2_ee)
                            if x_in_range:
                                break
                            else: 
                                if condition1 or condition2 or condition3:

                                    if unique_xedges[i] < condition1_es or unique_xedges[i] < condition2_ee:
                                        if condition1_es and condition2_ee:
                                            if any(es == ee for es in condition1_es for ee in condition2_ee):
                                                print('hi')
                                            else:
                                            # print('both')
                                                min_value = min(min(condition1_es), min(condition2_ee))
                                                xlines = SmoothMeshLines([min_value, unique_xedges[i]], mesh_res / 4)
                                            n = [x for x in xlines if unique_xedges[i] < x < min_value]
                                            hint[0].extend(n)
                                        if condition1_es and not (condition2_ee and condition1_es):
                                            # print('condition1_es', condition1_es)
                                            for es in condition1_es:
                                                xlines = SmoothMeshLines([es, unique_xedges[i]], mesh_res / 4)
                                                n = [x for x in xlines if unique_xedges[i] < x < es]
                                                hint[0].extend(n)
                                        if condition2_ee and not (condition2_ee and condition1_es):
                                            # print('condition2_ee', condition2_ee)
                                            for ee in condition2_ee:
                                                xlines = SmoothMeshLines([ee, unique_xedges[i]], mesh_res / 4)
                                                n = [x for x in xlines if unique_xedges[i] < x < ee]
                                                hint[0].extend(n)
                                        # n = [x for x in xlines if unique_xedges[i] < x < min(min(edge_end),min(edge_start))]
                                        hint[0].extend(n)
                                        # print('hi')
                                    if unique_xedges[i+1] > max(max(condition1_es),max(condition2_ee)):
                                        xlines=SmoothMeshLines([max(max(condition1_es),max(condition2_ee)), unique_yedges[i + 1]], mesh_res)
                                        n = [x for x in xlines if max(max(condition1_es),max(condition2_ee)) < x < unique_xedges[i + 1]]
                                        hint[0].extend(n)
                                    else:
                                        xlines = np.linspace(max(max(edge_end),max(edge_start)),unique_xedges[i + 1],5)
                                        xlines=SmoothMeshLines([max(max(edge_end),max(edge_start)), unique_xedges[i + 1]], mesh_res)
                                        n = [x for x in xlines if max(max(edge_end),max(edge_start)) < x < unique_xedges[i + 1]]
                                        hint[0].extend(n)
                                        print(unique_xedges[i], condition2_ee,condition1_es)
                                        # print('hello')
                                if not any([condition1, condition2, condition3]):
                                # if not any(unique_xedges[i] < other_edge[0] < unique_xedges[i + 1] or unique_xedges[i] < other_edge[1] < unique_xedges[i + 1] for other_edge in otheredges):
                                    xlines = grid.GetLines('x', do_sort=True)
                                    xlines = SmoothMeshLines(xlines, mesh_res)
                                    n = [x for x in xlines if unique_xedges[i] < x < unique_xedges[i + 1]]
                                    hint[0].extend(n)
                                    # continue

                    else:
                        xlines = grid.GetLines('x', do_sort=True)
                        xlines=SmoothMeshLines(xlines, mesh_res)
                        n = [x for x in xlines if unique_xedges[i] < x < unique_xedges[i + 1]]
                        hint[0].extend(n)   
                        continue 
                else:
                    continue




            if unique_yedges[0] <= sorted_y[0] and unique_yedges[1]-unique_yedges[0] > mesh_res:
                hint[1].append(sorted_yedges[0][0]-mer[0])
            if unique_yedges[-1] >= sorted_y[-1] and unique_yedges[-1]-unique_yedges[-2] > mesh_res:
                hint[1].append(sorted_yedges[-1][0]+mer[0])
            for i in range(len(unique_yedges) - 1):
                if y_diffs[i] > mesh_res:
                    if otheredges:
                        y_in_range = []
                        for other_edge in otheredges:
                            y_start = unique_yedges[i]
                            y_end = unique_yedges[i + 1]
                            y_in_range = [unique_yedges[i] for other_edge in otheredges if other_edge[2] <= unique_yedges[i] <= other_edge[3] or other_edge[3] <= unique_yedges[i] <= other_edge[2]]
                            # y_in_range = [y for y in unique_yedges if y_start < y < y_end or y_start > y > y_end]
                            edge_start = [other_edge[2] for other_edge in otheredges]
                            edge_end = [other_edge[3] for other_edge in otheredges]
                            condition1 = any(y_start < es < y_end or y_start > es > y_end for es in edge_start)
                            condition1_es = [es for es in edge_start if y_start < es < y_end or y_start > es > y_end]
                            condition2 = any(y_start < ee < y_end  or y_start > ee > y_end for ee in edge_end)
                            condition2_ee = [ee for ee in edge_end if y_start < ee < y_end or y_start > ee > y_end]
                            condition3 = any((y_start in (es, ee)) and (y_end in (es, ee)) for es, ee in zip(edge_start, edge_end))
                            if y_in_range:
                                break
                            else: 

                                if condition1 or condition2 or condition3:
                                    if unique_yedges[i] < condition1_es or unique_yedges[i] < condition2_ee:
                                        # xlines=SmoothMeshLines([min(min(edge_end),min(edge_start)),unique_yedges[i]], mesh_res/2)
                                        if condition1_es and condition2_ee:
                                            if any(es == ee for es in condition1_es for ee in condition2_ee):
                                                print('hi')
                                            else:
                                                print('both')
                                                min_value = min(min(condition1_es), min(condition2_ee))
                                                ylines = SmoothMeshLines([min_value, unique_yedges[i]], mesh_res / 4)
                                            n = [y for y in ylines if unique_yedges[i] < y < min_value]
                                            hint[1].extend(n)
                                        if condition1_es and not (condition2_ee and condition1_es):
                                            # print('condition1_es', condition1_es)
                                            for es in condition1_es:
                                                ylines = SmoothMeshLines([es, unique_yedges[i]], mesh_res / 4)
                                                n = [y for y in ylines if unique_yedges[i] < y < es]
                                                hint[1].extend(n)
                                        if condition2_ee and not (condition2_ee and condition1_es):
                                            # print('condition2_ee', condition2_ee)
                                            for ee in condition2_ee:
                                                ylines = SmoothMeshLines([ee, unique_yedges[i]], mesh_res / 4)
                                                n = [y for y in ylines if unique_yedges[i] < y < ee]
                                                hint[1].extend(n)
                                                # print('hi')
                                        # n = [y for y in ylines if unique_yedges[i] < y < min(min(edge_end),min(edge_start))]
                                        hint[1].extend(n)
                                        # print('hi')
                                    if unique_yedges[i+1] > max(max(condition1_es),max(condition2_ee)):
                                        ylines=SmoothMeshLines([max(max(condition1_es),max(condition2_ee)), unique_yedges[i + 1]], mesh_res/2)
                                        n = [y for y in ylines if max(max(condition1_es),max(condition2_ee)) < y < unique_yedges[i + 1]]
                                        hint[1].extend(n)
                                        # print('hello')
                                    else:
                                        ylines = np.linspace(max(max(edge_end),max(edge_start)),unique_yedges[i + 1],5)
                                        ylines=SmoothMeshLines([max(max(edge_end),max(edge_start)), unique_yedges[i + 1]], mesh_res)
                                        n = [y for y in ylines if max(max(edge_end),max(edge_start)) < y < unique_yedges[i + 1]]
                                        hint[1].extend(n)
                                        print(unique_yedges[i], condition2_ee,condition1_es)
                                        # print('hello')
                                if not any([condition1, condition2, condition3]):
                                    ylines = grid.GetLines('y', do_sort=True)
                                    ylines = SmoothMeshLines(ylines, mesh_res/2)
                                    n = [y for y in ylines if unique_yedges[i] < y < unique_yedges[i + 1]]
                                    hint[1].extend(n)
                                    # print('hi')
                                    # continue

                    else:
                        ylines = grid.GetLines('y', do_sort=True)
                        ylines=SmoothMeshLines(ylines, mesh_res)
                        n = [y for y in ylines if unique_yedges[i] < y < unique_yedges[i + 1]]
                        hint[1].extend(n)   
                        continue 
                else:
                    print('hello')
                    continue

            if unique_xedges[0] <= sorted_x[0]:
                hint[0].append(unique_xedges[0]-2*abs(unique_xedges[0]-min(hint[0])))
            if unique_xedges[-1] >= sorted_x[-1]:
                hint[0].append(unique_xedges[-1]+2*abs(unique_xedges[-1]-max(hint[0])))
            if unique_yedges[0] <= sorted_y[0]:
                hint[1].append(unique_yedges[0]-2*abs(unique_yedges[0]-min(hint[1])))
            if unique_yedges[-1] >= sorted_y[-1]:
                hint[1].append(unique_yedges[-1]+2*abs(unique_yedges[-1]-max(hint[1])))
            
            if unique_xedges[-1] < sorted_x[-1]:
                for other_edge in otheredges:
                    if other_edge[0] <= sorted_x[-1] <= other_edge[1] or other_edge[1] <= sorted_x[-1] <= other_edge[0]:
                        xlines = SmoothMeshLines([unique_xedges[-1], min(other_edge[0],other_edge[1])], mesh_res)
                        hint[0].extend(xlines)
            if unique_xedges[0] > sorted_x[0]:
                for other_edge in otheredges:
                    if other_edge[0] <= sorted_x[0] <= other_edge[1] or other_edge[1] <= sorted_x[0] <= other_edge[0]:
                        if abs(np.diff([unique_xedges[0],max(other_edge[0],other_edge[1])])) < mesh_res:
                            xlines = np.linspace(unique_xedges[0], max(other_edge[0],other_edge[1]), 5)[1:]
                        else:
                            xlines = SmoothMeshLines([max(other_edge[0],other_edge[1]), unique_xedges[0]], mesh_res)
                        hint[0].extend(xlines)
                        

        realhint = [None,None,None]
        if dirs is not None:
            for ny in GetMultiDirs(dirs):
                realhint[ny] = hint[ny]
        if 'mesh' in kw:
            return self.mesh_combine(hint, kw['mesh'])
        # realhint[0] = SmoothMeshLines(hint[0], mesh_res)    
        return (realhint,dirs,metal_edge_res)
    
    def mesh_estimate_cfl_timestep(self, mesh):
        """ mesh_estimate_cfl_timestep(mesh)

        Estimate the maximum CFL time step of the given mesh needed to ensure numerical stability,
        assuming propagation in pure vacuum.

        :returns: the maximum CFL time step, in seconds
        """
        invMinDiff = [None, None, None]
        for ny in range(3):
            invMinDiff[ny] = np.min(np.diff(mesh.GetLines(ny))) ** -2

        delta_t = mesh.GetDeltaUnit() / (C0 * np.sqrt(np.sum(invMinDiff)))

        return delta_t

    def point_in_polygon(self, polygon, point):
        """
        Raycasting Algorithm to find out whether a point is in a given polygon.
        Performs the even-odd-rule Algorithm to find out whether a point is in a given polygon.
        This runs in O(n) where n is the number of edges of the polygon.
        *
        :param polygon: an array representation of the polygon where polygon[i][0] is the x Value of the i-th point and polygon[i][1] is the y Value.
        :param point:   an array representation of the point where point[0] is its x Value and point[1] is its y Value
        :return: whether the point is in the polygon (not on the edge, just turn < into <= and > into >= for that)
        """

        # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
        odd = False
        # For each edge (In this case for each point of the polygon and the previous one)
        i = 0
        j = len(polygon[0]) - 1
        while i < len(polygon[0]) - 1:
            i = i + 1
            # If a line from the point into infinity crosses this edge
            # One point needs to be above, one below our y coordinate
            # ...and the edge doesn't cross our Y corrdinate before our x coordinate (but between our x coordinate and infinity)

            if (((polygon[1][i] > point[1]) != (polygon[1][j] > point[1])) and (point[0] < ((polygon[0][j] - polygon[0][i]) * (point[1] - polygon[1][i]) / (polygon[1][j] - polygon[1][i])) +polygon[0][i])):
                # Invert odd
                odd = not odd
            j = i
        # If the number of crossings was odd, the point is in the polygon
        return odd

    def yline_in_polygon(self, polygon, x_point, y_start, y_end):

        loop = np.linspace(y_start, y_end, 10)
        loop = loop[1:-1] 

        for y_val in loop:
            point = [x_point, y_val]
            if not self.point_in_polygon(polygon, point):
                return False

        return True

    def xline_in_polygon(self, polygon, x_start, x_end, y_point):

        loop = np.linspace(x_start, x_end, 10)
        loop = loop[1:-1] 

        for x_val in loop:
            point = [x_val, y_point]
            if not self.point_in_polygon(polygon, point):
                return False

        return True
