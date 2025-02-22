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
        self.mesh_data = {}

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
        # # print(unique_primitives)
        unique_properties = list(self.properties_mesh_setup.keys())
        # print(unique_properties)
        # print(self.primitives_mesh_setup)
        

        if csx is None:
            raise Exception('Unable to access CSX!')
        
        grid = csx.GetGrid()
        processed_pairs = set()
        combined_primitive = []
        for i, prim in enumerate(unique_primitives):
            if len(unique_primitives) == 1:
                self.collect_mesh_data(prim, grid, **kw)
                continue
            for other_prim in unique_primitives[i+1:]:
                if (prim, other_prim) in processed_pairs or (other_prim, prim) in processed_pairs:
                    continue
                processed_pairs.add((prim, other_prim))
                if prim.GetProperty() == other_prim.GetProperty():
                    # combined_primitive = (prim, other_prim)
                    combined_primitive.append(prim)
                    combined_primitive.append(other_prim)
                    # self.collect_mesh_data_for_multiple_primitives(combined_primitive, grid, **kw)
                else:
                    self.collect_mesh_data(prim, grid, **kw)
                    self.collect_mesh_data(other_prim, grid, **kw)
        combined_primitive = list(set(combined_primitive))
        if combined_primitive:
            self.collect_mesh_data_for_multiple_primitives(combined_primitive, grid, **kw)
            # hint = self.mesh_hint_from_primitive(prim, **kw)
            # if hint is None:
            #     continue
            # for n in range(3):
            #     if hint[n] is None:
            #         continue
            #     grid.AddLine(n, hint[n])
        self.filter_mesh_data()
        self.create_mesh_lines(grid)

    def collect_mesh_data_for_multiple_primitives(self, primitives, grid, **kw):
        hint = None
        if primitives[0].GetType() == CSPrimitives.POINT:
            hint = self.mesh_hint_from_point(primitives[0], **kw)
        elif primitives[0].GetType() == CSPrimitives.BOX:
            hint = self.mesh_hint_from_box(primitives[0], **kw)
        elif primitives[0].GetType() == CSPrimitives.POLYGON:
            (hint,dirs,metal_edge_res) = self.mesh_hint_from_polygon(primitives, grid, **kw)
        if hint is not None:
            self.mesh_data[tuple(primitives)] = (hint,dirs,metal_edge_res,primitives[0].GetPriority())

    def collect_mesh_data(self, primitive, grid, **kw):
        hint = None
        if primitive.GetType() == CSPrimitives.POINT:
            hint = self.mesh_hint_from_point(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.BOX:
            hint = self.mesh_hint_from_box(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.POLYGON:
            (hint,dirs,metal_edge_res) = self.mesh_hint_from_polygon(primitive, grid, **kw)
        
        if hint is not None:
            self.mesh_data[primitive] = (hint,dirs,metal_edge_res,primitive.GetPriority())

    def filter_mesh_data(self):
    
        for prim1, data1 in list(self.mesh_data.items()):
            hint1, dirs1, metal_edge_res1, priority1 = data1
            for prim2, data2 in list(self.mesh_data.items()):
                if prim1 == prim2:
                    continue
                hint2, dirs2, metal_edge_res2, priority2 = data2
                for n in range(3):
                    if hint1[n] is not None and hint2[n] is not None:
                        continue
            
                        # if hint1[n][0] < hint2[n][0] < hint1[n][-1] and hint1[n][0] < hint2[n][-1] < hint1[n][-1]:
                        #     if priority1 > priority2:
                        #         hint2[n] = None
                        #     elif priority1 < priority2:
                        #         hint1[n] = [val for val in hint1[n] if val < hint2[n][0] or val > hint2[n][-1]]
                        #     else:
                        #         hint2[n] = None
          
                # if prim2 in self.mesh_data:
                #     self.mesh_data[prim2] = (hint2, dirs2, metal_edge_res2, priority2)
                # if prim1 in self.mesh_data:
                #     del self.mesh_data[prim1]

        # self.mesh_data = {prim: data for prim, data in self.mesh_data.items() if any(data[0])}

    def create_mesh_lines(self, grid):

        for primitive in self.mesh_data:
            for n in range(3):
                if self.mesh_data[primitive][0][n] is not None:
                    grid.AddLine(n, self.mesh_data[primitive][0][n])

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
        hint = [[], [], None]
        otheredges = []
        xedges, yedges = [], []
        if isinstance(polygon, list):
            for prim in polygon:
                x = prim.GetCoords()[0]
                y = prim.GetCoords()[1]
                for i in range(len(x) - 1):
                    if x[i] != x[i + 1] and y[i] != y[i + 1]:
                        otheredges.append([x[i], x[i + 1], y[i], y[i + 1]])                       
            coords = [prim.GetCoords() for prim in polygon]
            x = np.concatenate([coord[0] for coord in coords])
            y = np.concatenate([coord[1] for coord in coords])
            N = len(x)
            dirs = self.primitives_mesh_setup.get(polygon[0], {}).get('dirs') or \
                self.properties_mesh_setup.get(polygon[0].GetProperty(), {}).get('dirs') or \
                self.global_mesh_setup.get('dirs')
            metal_edge_res = self.primitives_mesh_setup.get(polygon[0], {}).get('metal_edge_res') or \
                            self.properties_mesh_setup.get(polygon[0].GetProperty(), {}).get('metal_edge_res') or \
                            self.global_mesh_setup.get('metal_edge_res')


        else:
            x = polygon.GetCoords()[0]
            y = polygon.GetCoords()[1]
            N = polygon.GetQtyCoords()
            for i in range(len(x) - 1):
                if x[i] != x[i + 1] and y[i] != y[i + 1]:
                    otheredges.append([x[i], x[i + 1], y[i], y[i + 1]])
            dirs = self.primitives_mesh_setup.get(polygon, {}).get('dirs') or \
                self.properties_mesh_setup.get(polygon.GetProperty(), {}).get('dirs') or \
                self.global_mesh_setup.get('dirs')
            metal_edge_res = self.primitives_mesh_setup.get(polygon, {}).get('metal_edge_res') or \
                            self.properties_mesh_setup.get(polygon.GetProperty(), {}).get('metal_edge_res') or \
                            self.global_mesh_setup.get('metal_edge_res')

        mesh_res = self.global_mesh_setup.get('mesh_resolution', None)
        mer = np.array([-1.0, 2.0]) / 3 * metal_edge_res if metal_edge_res else 0

        # min_distance_x = self.calc_min_distance(x)
        # min_distance_y = self.calc_min_distance(y)
        xedges, yedges = [], []

        if dirs is not None:
            for i in range(N - 1):
                if x[i] == x[i + 1]:
                    xedges.append([x[i], y[i], y[i + 1]])
                if y[i] == y[i + 1]:
                    yedges.append([y[i], x[i], x[i + 1]])
        else:
            hint = [None, None, None]

        unique_xedges = np.unique(np.sort([edge[0] for edge in xedges]))
        unique_yedges = np.unique(np.sort([edge[0] for edge in yedges]))
        otheredges = np.array(otheredges, dtype=float).tolist()

        sorted_x = np.sort(x) 
        sorted_y = np.sort(y)
        self.handle_otheredges(otheredges, unique_xedges, unique_yedges, mesh_res, hint[0], 'x')
        self.handle_otheredges(otheredges, unique_xedges, unique_yedges, mesh_res, hint[1], 'y')

        self.mesh_tight_areas(unique_xedges, mesh_res, hint, 'x')
        self.mesh_tight_areas(unique_yedges, mesh_res, hint, 'y')
        
        self.mesh_boundaries(unique_xedges, sorted_x, otheredges, mesh_res, hint[0], 'x')
        self.mesh_boundaries(unique_yedges, sorted_y, otheredges, mesh_res, hint[1], 'y')

        hint[0].extend(unique_xedges)
        hint[1].extend(unique_yedges)

        if isinstance(polygon, list):
            coords = [prim.GetCoords() for prim in polygon]
            x = np.concatenate([coord[0] for coord in coords])
            y = np.concatenate([coord[1] for coord in coords])
        else:
            x = polygon.GetCoords()[0]
            y = polygon.GetCoords()[1]     

        # def metal_edge(edges, polygon, mesh_res, hint, dirs, metal_edge_res, direction):
        #     if isinstance(polygon, list):
        #         coords = [prim.GetCoords() for prim in polygon]
        #         x = np.concatenate([coord[0] for coord in coords])
        #         y = np.concatenate([coord[1] for coord in coords])
        #     else:
        #         x = polygon.GetCoords()[0]
        #         y = polygon.GetCoords()[1] 
        #     mer = np.array([-1.0, 2.0]) / 3 * metal_edge_res if metal_edge_res else 0
        #     if direction == 'x':
        #         min_distance_x = self.calc_min_distance(x)
        #     if direction == 'y':
        #         min_distance_y = self.calc_min_distance(y)
        #     if dirs is not None:
        #         for i in range(len(edges) - 1):
        #             if direction == 'x':
        #                 condition1 = self.yline_in_polygon(coords, edges[i][0]+min_distance_x/2, edges[i][1], edges[i][2]) and not self.yline_in_polygon(coords, edges[i][0]-min_distance_x/2, edges[i][1], edges[i][2])
        #                 condition2 = self.yline_in_polygon(coords, edges[i][0]-min_distance_x/2, edges[i][1], edges[i][2]) and self.yline_in_polygon(coords, edges[i][0]+min_distance_x/2, edges[i][1], edges[i][2])
        #             if direction == 'y':
        #                 condition1 = self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]+min_distance_y/2) and not self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]-min_distance_y/2)
        #                 condition2 = self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]-min_distance_y/2) and self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]+min_distance_y/2)
        #             if metal_edge_res is not None:
        #                 if i > 0 and abs(edges[i][0] - edges[i + 1][0]) > mesh_res and abs(edges[i][0] - edges[i - 1][0]) > mesh_res:
        #                     if condition1:
        #                         hint_in_range =  [hint for hint in hint if edges[i][0]-mer[1] <= hint <= edges[i][0]-mer[0]]
        #                         if not hint_in_range:
        #                             hint.append(edges[i][0]-mer[1])
        #                             hint.append(edges[i][0]-mer[0])
        #                         else:
        #                             hint = [h for h in hint if h not in hint_in_range]
        #                             hint.append(edges[i][0]-mer[1])
        #                             hint.append(edges[i][0]-mer[0])
        #                     elif condition2:
        #                         continue
        #                     else:
        #                         hint_in_range =  [hint for hint in hint if edges[i][0]+mer[0] <= hint <= edges[i][0]+mer[1]]
        #                         if not hint_in_range:
        #                             hint.append(edges[i][0]+mer[0])
        #                             hint.append(edges[i][0]+mer[1])
        #                         else:
        #                             hint = [h for h in hint if h not in hint_in_range]
        #                             hint.append(edges[i][0]+mer[0])
        #                             hint.append(edges[i][0]+mer[1])
    
        # metal_edge(xedges, polygon, mesh_res, hint[0], dirs, metal_edge_res, 'x')
        # metal_edge(yedges, polygon, mesh_res, hint[1], dirs, metal_edge_res, 'y')
    
        # if dirs is not None:
        #             for i in range(len(xedges)-1):
        #                 if metal_edge_res is not None:
        #                     if i > 0 and abs(xedges[i][0] - xedges[i + 1][0]) > mesh_res and abs(xedges[i][0] - xedges[i - 1][0]) > mesh_res:
        #                         if self.yline_in_polygon(polygon.GetCoords(), xedges[i][0]+min_distance_x/2, xedges[i][1], xedges[i][2]) and not self.yline_in_polygon(polygon.GetCoords(), xedges[i][0]-min_distance_x/2, xedges[i][1], xedges[i][2]):
        #                             hint_in_range =  [hint for hint in hint[0] if xedges[i][0]-mer[1] <= hint <= xedges[i][0]-mer[0]]
        #                             if not hint_in_range:
        #                                 hint[0].append(xedges[i][0]-mer[1])
        #                                 hint[0].append(xedges[i][0]-mer[0])
        #                             else:
        #                                 hint[0] = [h for h in hint[0] if h not in hint_in_range]
        #                                 hint[0].append(xedges[i][0]-mer[1])
        #                                 hint[0].append(xedges[i][0]-mer[0])
        #                         elif self.yline_in_polygon(polygon.GetCoords(), xedges[i][0]-min_distance_x/2, xedges[i][1], xedges[i][2]) and self.yline_in_polygon(polygon.GetCoords(), xedges[i][0]+min_distance_x/2, xedges[i][1], xedges[i][2]):
        #                             continue
        #                         else:
        #                             hint_in_range =  [hint for hint in hint[0] if xedges[i][0]+mer[0] <= hint <= xedges[i][0]+mer[1]]
        #                             if not hint_in_range:
        #                                 hint[0].append(xedges[i][0]+mer[0])
        #                                 hint[0].append(xedges[i][0]+mer[1])
        #                             else:
        #                                 hint[0] = [h for h in hint[0] if h not in hint_in_range]
        #                                 hint[0].append(xedges[i][0]+mer[0])
        #                                 hint[0].append(xedges[i][0]+mer[1])
        #             for i in range(len(yedges)-1):
        #                     if metal_edge_res is not None:
        #                         if i > 0 and abs(yedges[i][0] - yedges[i + 1][0]) > mesh_res and abs(yedges[i][0] - yedges[i - 1][0]) > mesh_res:
        #                             if self.xline_in_polygon(polygon.GetCoords(), yedges[i][1], yedges[i][2], yedges[i][0]+min_distance_y/2) and not self.xline_in_polygon(polygon.GetCoords(), yedges[i][1], yedges[i][2], yedges[i][0]-min_distance_y/2):
        #                                 hint_in_range =  [hint for hint in hint[1] if yedges[i][0]-mer[1] <= hint <= yedges[i][0]-mer[0]]
        #                                 if not hint_in_range:
        #                                     hint[1].append(yedges[i][0]-mer[1])
        #                                     hint[1].append(yedges[i][0]-mer[0])
        #                                 else:
        #                                     hint[1] = [h for h in hint[1] if h not in hint_in_range]
        #                                     hint[1].append(yedges[i][0]-mer[1])
        #                                     hint[1].append(yedges[i][0]-mer[0])
        #                             elif self.xline_in_polygon(polygon.GetCoords(), yedges[i][1], yedges[i][2], yedges[i][0]+min_distance_y/2) and self.xline_in_polygon(polygon.GetCoords(), yedges[i][1], yedges[i][2], yedges[i][0]-min_distance_y/2):
        #                                 continue
        #                             else:
        #                                 hint_in_range =  [hint for hint in hint[1] if yedges[i][0]+mer[0] <= hint <= yedges[i][0]+mer[1]]
        #                                 if not hint_in_range:
        #                                     hint[1].append(yedges[i][0]+mer[0])
        #                                     hint[1].append(yedges[i][0]+mer[1])
        #                                 else:
        #                                     hint[1] = [h for h in hint[1] if h not in hint_in_range]
        #                                     hint[1].append(yedges[i][0]+mer[0])
        #                                     hint[1].append(yedges[i][0]+mer[1])
        if metal_edge_res is not None:
            if unique_xedges[0] <= sorted_x[0]:
                hint_in_range =  [hint for hint in hint[0] if unique_xedges[0]-mer[1] <= hint <= unique_xedges[0]-mer[0]]
                if not hint_in_range:
                    hint[0].append(unique_xedges[0]-mer[1])
                    hint[0].append(unique_xedges[0]-mer[0])
                else:
                    hint[0] = [h for h in hint[0] if h not in hint_in_range]
                    hint[0].append(unique_xedges[0]-mer[1])
                    hint[0].append(unique_xedges[0]-mer[0])
            if unique_xedges[-1] >= sorted_x[-1]:
                hint_in_range =  [hint for hint in hint[0] if unique_xedges[-1]+mer[0] <= hint <= unique_xedges[-1]+mer[1]]
                if not hint_in_range:
                    hint[0].append(unique_xedges[-1]+mer[0])
                    hint[0].append(unique_xedges[-1]+mer[1])
                else:
                    hint[0] = [h for h in hint[0] if h not in hint_in_range]
                    hint[0].append(unique_xedges[-1]+mer[0])
                    hint[0].append(unique_xedges[-1]+mer[1])
            if unique_yedges[0] <= sorted_y[0]:
                hint_in_range =  [hint for hint in hint[1] if unique_yedges[0]-mer[1] <= hint <= unique_yedges[0]-mer[0]]
                if not hint_in_range:
                    hint[1].append(unique_yedges[0]-mer[1])
                    hint[1].append(unique_yedges[0]-mer[0])
                else:
                    hint[1] = [h for h in hint[1] if h not in hint_in_range]
                    hint[1].append(unique_yedges[0]-mer[1])
                    hint[1].append(unique_yedges[0]-mer[0])
            if unique_yedges[-1] >= sorted_y[-1]:
                hint_in_range =  [hint for hint in hint[1] if unique_yedges[-1]+mer[0] <= hint <= unique_yedges[-1]+mer[1]]
                if not hint_in_range:
                    hint[1].append(unique_yedges[-1]+mer[0])
                    hint[1].append(unique_yedges[-1]+mer[1])
                else:
                    hint[1] = [h for h in hint[1] if h not in hint_in_range]
                    hint[1].append(unique_yedges[-1]+mer[0])
                    hint[1].append(unique_yedges[-1]+mer[1])
      
        hint[0] = SmoothMeshLines(hint[0], mesh_res)    
        hint[1] = SmoothMeshLines(hint[1], mesh_res)

        hint[0] = hint[0].tolist()
        hint[1] = hint[1].tolist()

        for i in range(len(hint[0])-1):
            if abs(hint[0][i] - hint[0][i+1]) <= mesh_res/20:
                hint[0][i] = (hint[0][i] + hint[0][i+1])/2
                hint[0][i+1] = hint[0][i]
        for i in range(len(hint[1])-1):
            if abs(hint[1][i] - hint[1][i+1]) <= mesh_res/20:
                hint[1][i] = (hint[1][i] + hint[1][i+1])/2
                hint[1][i+1] = hint[1][i]

        realhint = [None, None, None]
        if dirs is not None:
            for ny in GetMultiDirs(dirs):
                realhint[ny] = hint[ny]       
        if 'mesh' in kw:
            return self.mesh_combine(hint, kw['mesh'])
        return (realhint, dirs, metal_edge_res)
    
    def handle_otheredges(self, otheredges, unique_xedges, unique_yedges, mesh_res, hint, direction):

        other_edges_in_range = []
        if direction == 'x':
            unique_edges = unique_xedges
            
        if direction == 'y':
            unique_edges = unique_yedges
        for edge in otheredges:
            if direction == 'x':
                start , end = edge[0], edge[1]
                other_edges_in_range = [other_edge for other_edge in otheredges if (start <= other_edge[0] <= end or start <= other_edge[1] <= end or start >= other_edge[0] >= end or start >= other_edge[1] >= end)]
            if direction == 'y':
                start , end = edge[2], edge[3]
                other_edges_in_range = [other_edge for other_edge in otheredges if (start <= other_edge[2] <= end or start <= other_edge[3] <= end or start >= other_edge[2] >= end or start >= other_edge[3] >= end)]
            x_start, x_end, y_start, y_end = edge
            lines_in_range = [direction for direction in unique_edges if start <= direction <= end or start >= direction >= end]

            if not other_edges_in_range:
                alpha = np.atan(abs((y_end-y_start))/abs((x_end-x_start)))
                resolution = mesh_res * np.cos(alpha)
                if not lines_in_range:
                    lines=SmoothMeshLines([start, end], resolution)    
                    if np.min(abs(np.diff(lines))) > mesh_res:
                        lines = []
                        lines = SmoothMeshLines([start, end], mesh_res)
                    hint.extend(lines)
                if lines_in_range:
                    min_line = np.min([np.min(lines_in_range), start, end])
                    max_line = np.max([np.max(lines_in_range), start, end])
                    lines = np.linspace(min_line, max_line, 5)
                    lines=SmoothMeshLines(lines, resolution/2)
                    hint.extend(lines)
            if other_edges_in_range:
                alpha = np.round(np.rad2deg(np.atan(abs((y_end-y_start))/abs((x_end-x_start)))),2)
                if direction == 'x':
                    other_edges_in_range_here = [other_edge[0:2] for other_edge in other_edges_in_range]
                    resolution = mesh_res * np.cos(np.deg2rad(alpha))
                    other_edges_in_range_here_all_coords = [other_edge for other_edge in other_edges_in_range if (start <= other_edge[0] <= end or start <= other_edge[1] <= end or start >= other_edge[0] >= end or start >= other_edge[1] >= end)]
                if direction == 'y':
                    other_edges_in_range_here = [other_edge[2:4] for other_edge in other_edges_in_range]
                    resolution = mesh_res * np.sin(np.deg2rad(alpha))
                    other_edges_in_range_here_all_coords = [other_edge for other_edge in other_edges_in_range if (start <= other_edge[2] <= end or start <= other_edge[3] <= end or start >= other_edge[2] >= end or start >= other_edge[3] >= end)]
                min_line = np.min([start,end, np.min(np.min(other_edges_in_range_here))])
                max_line = np.max([start,end, np.max(np.max(other_edges_in_range_here))])
                alphas_in_range = [(np.round(np.rad2deg(np.atan(abs((edge[3] - edge[2])) / abs((edge[1] - edge[0])))),2), edge) for edge in other_edges_in_range_here_all_coords]
                lines_in_hint_range = [direction for direction in hint if min_line <= direction <= max_line] 
                for line in lines_in_hint_range:
                    hint.remove(line)

                lines_in_range = [direction for direction in unique_edges if min_line < direction < max_line]
                if lines_in_range:
                    lines_in_range.extend([min_line, max_line])    
                    lines=SmoothMeshLines(lines_in_range, resolution)
                    hint.extend(lines)       
                if not lines_in_range:
                    lines=SmoothMeshLines([min_line,max_line], resolution)
                    hint.extend(lines)
                for i in range(len(alphas_in_range)):
                    if alphas_in_range[i][0] > alpha:
                        lines_in_range = [direction for direction in unique_edges if alphas_in_range[i][1][0] <= direction <= alphas_in_range[i][1][1]]
                        if lines_in_range:
                            lines_in_range.extend([alphas_in_range[i][1][0], alphas_in_range[i][1][1]])    
                            lines=SmoothMeshLines(lines_in_range, resolution)
                            hint.extend(lines)       
                        if not lines_in_range:
                            lines=SmoothMeshLines([alphas_in_range[i][1][0], alphas_in_range[i][1][1]], resolution)
                            hint.extend(lines)
                if direction == 'x':
                    for i in range(len(alphas_in_range)):
                        if alphas_in_range[i][0] > alpha:
                            lines_in_hint_range = [lines for lines in hint if alphas_in_range[i][1][0] <= lines <= alphas_in_range[i][1][1]]
                            for lines in lines_in_hint_range:
                                hint.remove(lines)
                            resolution = mesh_res * np.cos(np.deg2rad(alphas_in_range[i][0]))
                            xlines = SmoothMeshLines([alphas_in_range[i][1][0], alphas_in_range[i][1][1]], resolution)
                            hint.extend(xlines)
                if direction == 'y':
                    for i in range(len(alphas_in_range)):
                        if alphas_in_range[i][0] < alpha:
                            lines_in_hint_range = [lines for lines in hint if alphas_in_range[i][1][2] <= lines <= alphas_in_range[i][1][3]]
                            for lines in lines_in_hint_range:
                                hint.remove(lines)
                            resolution = mesh_res * np.sin(np.deg2rad(alphas_in_range[i][0]))
                            ylines = SmoothMeshLines([alphas_in_range[i][1][2], alphas_in_range[i][1][3]], resolution)
                            hint.extend(ylines)
        if direction == 'x':
            otheredges.sort(key=lambda edge: edge[0])
        if direction == 'y':
            otheredges.sort(key=lambda edge: edge[2])

        for i in range(len(otheredges)):
            for j in range(i + 1, len(otheredges)):
                line1 = otheredges[i]
                line2 = otheredges[j]
                p1 = np.array([line1[0], line1[2]])  # (x1, y1)
                p2 = np.array([line1[1], line1[3]])  # (x2, y2)
                q1 = np.array([line2[0], line2[2]])  # (x1, y1)
                q2 = np.array([line2[1], line2[3]])  # (x2, y2)
                dist = self.distance_between_segments(p1, p2, q1, q2)
                dist = [small_dist for small_dist in dist if small_dist[0] <= mesh_res]
                alpha = np.round(np.rad2deg(np.atan(abs((q2[1] - q1[1]) / abs((q2[0] - q1[0]))))), 2)
                if dist:
                    if direction == 'x':
                        coords_of_p = [item[1][0] for item in dist]
                        resolution = mesh_res * np.cos(np.deg2rad(alpha))
                    if direction == 'y':
                        coords_of_p = [item[1][1] for item in dist]
                        resolution = mesh_res * np.sin(np.deg2rad(alpha))
                    lines_in_range = [lines for lines in hint if np.min(coords_of_p) <= lines <= np.max(coords_of_p)]
                    for line in lines_in_range:
                        hint.remove(line)
                    hint.extend(SmoothMeshLines([np.min(coords_of_p), np.max(coords_of_p)], resolution/2))

    def mesh_tight_areas(self, unique_edges, mesh_res, hint, direction):
        for i in range(len(unique_edges) - 1):
            if abs(np.diff([unique_edges[i], unique_edges[i + 1]])) < mesh_res:
                if direction == 'x':
                    x_in_range = [x for x in hint[0] if unique_edges[i] <= x <= unique_edges[i + 1]]
                    for x in x_in_range:
                        hint[0].remove(x)
                    hint[0].extend(np.linspace(unique_edges[i], unique_edges[i + 1], 5)[1:-1])
                else:
                    y_in_range = [y for y in hint[1] if unique_edges[i] <= y <= unique_edges[i + 1]]
                    for y in y_in_range:
                        hint[1].remove(y)
                    hint[1].extend(np.linspace(unique_edges[i], unique_edges[i + 1], 5)[1:-1])

    def mesh_boundaries(self, unique_edges, sorted_points, otheredges, mesh_res, hint, direction):
        if unique_edges[-1] < sorted_points[-1]:
            for other_edge in otheredges:
                if direction == 'x':
                    start, end = other_edge[0], other_edge[1]
                if direction == 'y':
                    start, end = other_edge[2], other_edge[3]
                if start <= sorted_points[-1] <= end or end <= sorted_points[-1] <= start:
                    if abs(np.diff([unique_edges[-1], min(start, end)])) < mesh_res:
                        lines = np.linspace(unique_edges[-1], min(start, end), 5)[1:]
                    else:
                        lines = SmoothMeshLines([unique_edges[-1], min(start, end)], mesh_res)[1:]
                    hint.extend(lines)
        if unique_edges[0] > sorted_points[0]:
            for other_edge in otheredges:
                if direction == 'x':
                    start, end = other_edge[0], other_edge[1]
                if direction == 'y':
                    start, end = other_edge[2], other_edge[3]
                if start <= sorted_points[0] <= end or end <= sorted_points[0] <= start:
                    if abs(np.diff([unique_edges[0], max(start, end)])) < mesh_res:
                        lines = np.linspace(unique_edges[0], max(start, end), 5)[1:]
                    else:
                        lines = SmoothMeshLines([max(start, end), unique_edges[0]], mesh_res)
                    hint.extend(lines)               

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

    def distance_between_segments(self, p1, p2, q1, q2):
        p = np.linspace(p1, p2, 20)
        def point_to_line_distance(p, a, b):
            # Projektion des Punktes p auf die Linie a-b
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)  # Projektion auf das Segment beschränken
            closest_point = a + t * ab
            return np.linalg.norm(p - closest_point)
        distances = []
        for p_point in p:
            distances.append((point_to_line_distance(p_point, q1, q2), p_point, q1, q2))
        # distances = [
        #     point_to_line_distance(p1, q1, q2),
        #     point_to_line_distance(p2, q1, q2),
        #     point_to_line_distance(q1, p1, p2),
        #     point_to_line_distance(q2, p1, p2),
        # ]
        # print('distances:', distances)
        return distances
    
    def calc_min_distance(self, x):
        min_distance = float('inf')
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                distance = abs(x[i] - x[j])
                if distance > 0 and distance < min_distance:
                    min_distance = distance
        return min_distance
    
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

            if (((polygon[1][i] > point[1]) != (polygon[1][j] > point[1])) and (point[0] < ((polygon[0][j] - polygon[0][i]) * (point[1] - polygon[1][i]) / (polygon[1][j] - polygon[1][i])) +polygon[0][i])):                # Invert odd
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
    
