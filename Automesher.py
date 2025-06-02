import sys
import numpy as np

from CSXCAD import CSPrimitives
from CSXCAD import CSProperties
from CSXCAD.Utilities import CheckNyDir, GetMultiDirs
from openEMS.physical_constants import C0
from CSXCAD.SmoothMeshLines import SmoothMeshLines
   
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

        csx = CSX
        grid = csx.GetGrid()

        unique_primitives = list(self.primitives_mesh_setup.keys())
        unique_properties = list(self.properties_mesh_setup.keys())
        only_edges = []
        for primitive, mesh_setup in self.primitives_mesh_setup.items():
            if mesh_setup.get('edges_only', False):
                unique_primitives.remove(primitive)
                only_edges.append(primitive)
        if only_edges:
            if len(only_edges) == 1:
                self.collect_mesh_data(only_edges[0], **kw)
            else:
                self.collect_mesh_data_for_multiple_primitives(only_edges, **kw)

        if len(unique_primitives) == 1:
            self.collect_mesh_data(unique_primitives[0], **kw)
        else:
            self.collect_mesh_data_for_multiple_primitives(unique_primitives, **kw)

        self.add_mesh_lines(grid)

        lines = self.process_mesh_lines(grid)

        for i in range(3):
            grid.AddLine(i, lines[i][0])

    def collect_mesh_data_for_multiple_primitives(self, primitives, **kw):
        mesh_data = None
        (mesh_data,dirs,metal_edge_res) = self.get_mesh_data(primitives,  **kw)
        if mesh_data is not None:
            self.mesh_data[tuple(primitives)] = (mesh_data,dirs,metal_edge_res)

    def collect_mesh_data(self, primitive, **kw):
        mesh_data = None
        if primitive.GetType() == CSPrimitives.POLYGON:
            (mesh_data,dirs,metal_edge_res) = self.get_mesh_data(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.LINPOLY:
            (mesh_data,dirs,metal_edge_res) = self.get_mesh_data(primitive, **kw)
        
        if mesh_data is not None:
            self.mesh_data[primitive] = (mesh_data,dirs,metal_edge_res)

    def add_mesh_lines(self, grid):
        for primitive in self.mesh_data:
            for n in range(3):
                if self.mesh_data[primitive][0][n]:
                    grid.AddLine(n, self.mesh_data[primitive][0][n])

    def get_mesh_data(self, polygon, **kw):

        # Initialize the mesh_data list for x, y, and z directions
        mesh_data = [[], [], []]

        # Initialize lists to store edges and coordinates
        diagonal_edges = []
        xedges, yedges = [], []
        x, y= [], []

        # Check if the input is a list of primitives
        if isinstance(polygon, list):
            # Process each primitive
            for prim in polygon:
                self.process_primitive(prim, x, y, xedges, yedges, diagonal_edges)

            # Collect z-coordinates from the polygon
            z = self.collect_z_coordinates(polygon)

            # Process z-coordinates and add them to the mesh_data list
            self.process_z_coordinates(z, mesh_data)

            # Get the mesh setup parameters
            dirs = self.global_mesh_setup.get('dirs')
            # metal_edge_res = self.primitives_mesh_setup.get(polygon[0], {}).get('metal_edge_res') or \
                            # self.properties_mesh_setup.get(polygon[0].GetProperty(), {}).get('metal_edge_res') or \
                            # self.global_mesh_setup.get('metal_edge_res')
        else:
            # If the polygon is a single primitive, get its x and y coordinates
            x, y = polygon.GetCoords()[0], polygon.GetCoords()[1]

            # Process the single polygon to extract edges and coordinates
            self.process_single_polygon(polygon,x, y, xedges, yedges, diagonal_edges)

           # Collect z-coordinates from the single polygon 
            z = self.collect_z_coordinates([polygon])

            # Process z-coordinates and add them to the mesh_data
            self.process_z_coordinates(z, mesh_data)
            dirs = self.global_mesh_setup.get('dirs')
            # metal_edge_res = self.primitives_mesh_setup.get(polygon, {}).get('metal_edge_res') or \
            #                 self.properties_mesh_setup.get(polygon.GetProperty(), {}).get('metal_edge_res') or \
            #                 self.global_mesh_setup.get('metal_edge_res')

        # Get the mesh parameters
        self.mesh_res, self.min_cellsize, self.max_res, self.max_cellsize, self.num_lines = self.get_mesh_parameters()
        # mer = np.array([-1.0, 2.0]) / 3 * metal_edge_res if metal_edge_res else 0

        # Get unique vertical and horizontal edges
        unique_xedges, unique_yedges = self.get_unique_edges(xedges), self.get_unique_edges(yedges)

        self.found_circles = self.detect_all_circles_in_polygon(polygon)

        print ('min_cellsize:', self.min_cellsize)
        print ('mesh_res:', self.mesh_res)
        print('num_lines:', self.num_lines) 
        print ('max_res:', self.max_res)
        print ('max_cellsize:', self.max_cellsize)
        distance_smaller_than_min_cellsize = []
        min_cellsize_changed = False
        if self.global_mesh_setup.get('min_cellsize', None) is None:
            for i in range(len(unique_xedges) - 1):
                condition1 = abs(unique_xedges[i+1][0] - unique_xedges[i][0]) <= self.min_cellsize and abs(unique_xedges[i+1][0] - unique_xedges[i][0]) >= 1.5
                condition2 = abs(unique_xedges[i+1][0] - unique_xedges[i][0]) > self.min_cellsize and abs(unique_xedges[i+1][0] - unique_xedges[i][0]) < self.max_res
                if condition1 or condition2:
                    distance_smaller_than_min_cellsize.append([abs(unique_xedges[i+1][0] - unique_xedges[i][0]), unique_xedges[i][0], unique_xedges[i+1][0]])
        
            if distance_smaller_than_min_cellsize:
                n = min(distance_smaller_than_min_cellsize)
                lines = np.linspace(n[1], n[2], self.num_lines)
                ds = abs(np.min(np.diff(lines)))
                if ds < self.min_cellsize:
                    self.min_cellsize = ds
                    min_cellsize_changed = True
                    # insure that tight areas are not smaller mesh_resolution by multiplying with the number of cells between the two tight edges
                    self.mesh_res = round(ds * (self.num_lines -1))
                    self.max_res = self.min_cellsize + 0.25*self.min_cellsize
                    epsilon = None
                    for primitive in self.primitives_mesh_setup.keys():
                        if hasattr(primitive, 'GetProperty') and hasattr(primitive.GetProperty(), 'GetMaterialProperty') and primitive.GetProperty().GetMaterialProperty('epsilon') > 1:
                            current_epsilon = primitive.GetProperty().GetMaterialProperty('epsilon')
                            # self.primitives_with_epsilon[primitive] = current_epsilon
                            if epsilon is None or current_epsilon > epsilon:
                                epsilon = current_epsilon
                    if epsilon:
                        self.max_cellsize = self.max_cellsize / (epsilon ** 0.5)
            if self.max_cellsize == self.mesh_res:
                self.max_cellsize = self.mesh_res * 2

        self.check_max_resolution(diagonal_edges, unique_xedges, unique_yedges, self.mesh_res, self.max_res, mesh_data)

        print ('min_cellsize:', self.min_cellsize)
        print ('mesh_res:', self.mesh_res)
        print('num_lines:', self.num_lines) 
        print ('max_res:', self.max_res)
        print ('max_cellsize:', self.max_cellsize)

        # Sort x and y coordinates
        sorted_x, sorted_y = np.sort(x), np.sort(y)

        # Handle diagonal edges and add mesh lines for x and y directions
        self.handle_otheredges(diagonal_edges, unique_xedges, unique_yedges, self.mesh_res, self.max_res, mesh_data[0], 'x')
        self.handle_otheredges(diagonal_edges, unique_xedges, unique_yedges, self.mesh_res, self.max_res, mesh_data[1], 'y')

        # Refine mesh in small gaps for x and y edges
        self.mesh_small_gaps(xedges, self.mesh_res, self.max_res, self.num_lines, mesh_data, 'x')
        self.mesh_small_gaps(yedges, self.mesh_res, self.max_res, self.num_lines, mesh_data, 'y')
        self.mesh_small_gaps(z, self.mesh_res, self.max_res, self.num_lines, mesh_data, 'z')
        
        # Recalculate unique edges after refinement 
        unique_xedges, unique_yedges = self.get_unique_edges(xedges), self.get_unique_edges(yedges)
        
        # Sort edges by their starting coordinates
        xedges.sort(key=lambda edge: edge[0])
        yedges.sort(key=lambda edge: edge[0])   

        # Add missing mesh lines between points and edges for x and y directions
        self.add_missing_mesh_lines(unique_xedges, sorted_x, diagonal_edges, self.mesh_res, mesh_data[0], 'x')
        self.add_missing_mesh_lines(unique_yedges, sorted_y, diagonal_edges, self.mesh_res, mesh_data[1], 'y')
        
        # Add edges to the mesh mesh_data for x and y directions
        self.add_edges_to_mesh_mesh_data(mesh_data[0], xedges, self.mesh_res, self.min_cellsize, 'x')
        self.add_edges_to_mesh_mesh_data(mesh_data[1], yedges, self.mesh_res, self.min_cellsize, 'y')

        # Add graded meshlines at material transitions
        self.add_graded_mesh_lines_at_material_transitions(xedges,mesh_data[0], self.mesh_res)
        self.add_graded_mesh_lines_at_material_transitions(yedges,mesh_data[1], self.mesh_res)
        z = [(z[0], None, None, z[1]) for z in z]
        self.add_graded_mesh_lines_at_material_transitions(z,mesh_data[2], self.mesh_res)

        # Handle circular segments in the polygon
        self.handle_circular_segments(polygon, mesh_data)
        
        # self.metal_edge(xedges, polygon, self.mesh_res, mesh_data[0], dirs, metal_edge_res, 'x')
        # self.metal_edge(yedges, polygon, self.mesh_res, mesh_data[1], dirs, metal_edge_res, 'y')
    
        if list(self.mesh_data.values()):
            mesh_data[0].extend(list(self.mesh_data.values())[0][0][0])
            mesh_data[1].extend(list(self.mesh_data.values())[0][0][1])
            mesh_data[2].extend(list(self.mesh_data.values())[0][0][2])
        # Smooth the mesh lines for x, y, and z directions if edges_only is not set
        mesh_data[0] = sorted(mesh_data[0])
        mesh_data[1] = sorted(mesh_data[1])
        mesh_data[2] = sorted(mesh_data[2])
        mesh_with_max_cell_size = [[], [], []] 
        mesh_with_different_mesh_res = []

        if isinstance(polygon, list):
            # Check if any primitive has larger epsilon than 1
            if not min_cellsize_changed:
                for prim in polygon:
                    if hasattr(prim, 'GetProperty') and hasattr(prim.GetProperty(), 'GetMaterialProperty'):
                        if prim.GetProperty().GetMaterialProperty('epsilon') > 1:
                            epsilon = prim.GetProperty().GetMaterialProperty('epsilon')
                            tmp_mesh_res = self.mesh_res / (epsilon ** 0.5)
                            mesh_with_different_mesh_res.append([tmp_mesh_res, prim])
                if mesh_with_different_mesh_res:
                    if len(mesh_with_different_mesh_res) == 1:
                        same_prim_edges_x = [edge for edge in xedges if edge[3] == mesh_with_different_mesh_res[0][1]]
                        same_prim_edges_y = [edge for edge in yedges if edge[3] == mesh_with_different_mesh_res[0][1]]
                        same_prim_edges_z = [edge for edge in z if edge[3] == mesh_with_different_mesh_res[0][1]]
                        if same_prim_edges_x:
                            if len(same_prim_edges_x) > 1:
                                same_prim_edges_x.sort(key=lambda edge: edge[0])
                                for j in range(len(same_prim_edges_x) - 1):
                                    lines_in_range = [line for line in mesh_data[0] if same_prim_edges_x[j][0] <= line <= same_prim_edges_x[j + 1][0]]
                                    if lines_in_range:
                                        resolution = mesh_with_different_mesh_res[0][0]
                                        mesh_data[0] = SmoothMeshLines(lines_in_range, resolution).tolist()
                        if same_prim_edges_y:
                            if len(same_prim_edges_y) > 1:
                                same_prim_edges_y.sort(key=lambda edge: edge[0])
                                for j in range(len(same_prim_edges_y) - 1):
                                    lines_in_range = [line for line in mesh_data[1] if same_prim_edges_y[j][0] <= line <= same_prim_edges_y[j + 1][0]]
                                    if lines_in_range:
                                        resolution = mesh_with_different_mesh_res[0][0]
                                        mesh_data[1] = SmoothMeshLines(lines_in_range, resolution).tolist()
                        if same_prim_edges_z:
                            if len(same_prim_edges_z) > 1:
                                same_prim_edges_z.sort(key=lambda edge: edge[0])
                                for j in range(len(same_prim_edges_z) - 1):
                                    lines_in_range = [line for line in mesh_data[2] if same_prim_edges_z[j][0] <= line <= same_prim_edges_z[j + 1][0]]
                                    if lines_in_range:
                                        resolution = mesh_with_different_mesh_res[0][0]
                                        mesh_data[2] = SmoothMeshLines(lines_in_range, resolution).tolist()

                    if len(mesh_with_different_mesh_res) > 1:
                        mesh_with_different_mesh_res.sort(key=lambda x: x[0])
                        for i in range(len(mesh_with_different_mesh_res) - 1, -1, -1):
                            same_prim_edges_x = [edge for edge in xedges if edge[3] == mesh_with_different_mesh_res[i][1]]
                            same_prim_edges_y = [edge for edge in yedges if edge[3] == mesh_with_different_mesh_res[i][1]]
                            same_prim_edges_z = [edge for edge in z if edge[3] == mesh_with_different_mesh_res[i][1]]
                            if same_prim_edges_x:
                                if len(same_prim_edges_x) > 1:
                                    same_prim_edges_x.sort(key=lambda edge: edge[0])
                                    for j in range(len(same_prim_edges_x) - 1):
                                        lines_in_range = [line for line in mesh_data[0] if same_prim_edges_x[j][0] <= line <= same_prim_edges_x[j + 1][0]]
                                        if lines_in_range:
                                            resolution = mesh_with_different_mesh_res[i][0]
                                            mesh_data[0] = SmoothMeshLines(lines_in_range, resolution).tolist()
                            if same_prim_edges_y:
                                if len(same_prim_edges_y) > 1:
                                    same_prim_edges_y.sort(key=lambda edge: edge[0])
                                    for j in range(len(same_prim_edges_y) - 1):
                                        lines_in_range = [line for line in mesh_data[1] if same_prim_edges_y[j][0] <= line <= same_prim_edges_y[j + 1][0]]
                                        if lines_in_range:
                                            resolution = mesh_with_different_mesh_res[i][0]
                                            mesh_data[1] = SmoothMeshLines(lines_in_range, resolution).tolist()
                            if same_prim_edges_z:
                                if len(same_prim_edges_z) > 1:
                                    same_prim_edges_z.sort(key=lambda edge: edge[0])
                                    for j in range(len(same_prim_edges_z) - 1):
                                        lines_in_range = [line for line in mesh_data[2] if same_prim_edges_z[j][0] <= line <= same_prim_edges_z[j + 1][0]]
                                        if lines_in_range:
                                            resolution = mesh_with_different_mesh_res[i][0]
                                            mesh_data[2] = SmoothMeshLines(lines_in_range, resolution).tolist()

            if not any (self.primitives_mesh_setup.get(prim, {}).get('edges_only', False) for prim in polygon):
                for i in range(len(mesh_data[0]) - 1):
                    if mesh_data[0][i+1] - mesh_data[0][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[0].append((mesh_data[0][i], mesh_data[0][i+1]))
                for i in range(len(mesh_data[1]) - 1):
                    if mesh_data[1][i+1] - mesh_data[1][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[1].append((mesh_data[1][i], mesh_data[1][i+1]))
                for i in range(len(mesh_data[2]) - 1):
                    if mesh_data[2][i+1] - mesh_data[2][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[2].append((mesh_data[2][i], mesh_data[2][i+1]))

                mesh_data[0] = SmoothMeshLines(mesh_data[0], self.mesh_res).tolist()    
                mesh_data[1] = SmoothMeshLines(mesh_data[1], self.mesh_res).tolist()
                mesh_data[2] = SmoothMeshLines(mesh_data[2], self.mesh_res).tolist()

                for i in range(3):
                    for start, end in mesh_with_max_cell_size[i]:
                        mesh_data[i] = [line for line in mesh_data[i] if not (start < line < end)]

        else:
            if not self.primitives_mesh_setup.get(polygon, {}).get('edges_only', False):
                for i in range(len(mesh_data[0]) - 1):
                    if mesh_data[0][i+1] - mesh_data[0][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[0].append((mesh_data[0][i], mesh_data[0][i+1]))
                for i in range(len(mesh_data[1]) - 1):
                    if mesh_data[1][i+1] - mesh_data[1][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[1].append((mesh_data[1][i], mesh_data[1][i+1]))
                for i in range(len(mesh_data[2]) - 1):
                    if mesh_data[2][i+1] - mesh_data[2][i] > self.max_cellsize/2:
                        mesh_with_max_cell_size[2].append((mesh_data[2][i], mesh_data[2][i+1]))

                mesh_data[0] = SmoothMeshLines(mesh_data[0], self.mesh_res).tolist()    
                mesh_data[1] = SmoothMeshLines(mesh_data[1], self.mesh_res).tolist()
                mesh_data[2] = SmoothMeshLines(mesh_data[2], self.mesh_res).tolist()

                for i in range(3):
                    for start, end in mesh_with_max_cell_size[i]:
                        mesh_data[i] = [line for line in mesh_data[i] if not (start < line < end)]
  
        # Process the mesh_data to remove close points and refine the mesh        
        if self.global_mesh_setup.get('min_cellsize', None) is not None or min_cellsize_changed:
            mesh_data[0] = self.process_mesh_data(mesh_data[0], self.min_cellsize, unique_xedges)
            mesh_data[1] = self.process_mesh_data(mesh_data[1], self.min_cellsize, unique_yedges)
            mesh_data[2] = self.process_mesh_data(mesh_data[2], self.min_cellsize, z)
        
        # Add Ports to the mesh_data
        mesh_data[0] = self.add_ports_to_mesh_data(mesh_data[0], xedges, 'x')
        mesh_data[1] = self.add_ports_to_mesh_data(mesh_data[1], yedges, 'y')
        mesh_data[2] = self.add_ports_to_mesh_data(mesh_data[2], z, 'z')

        # Chech if all edges are in the mesh_data
        self.add_edges_to_mesh_mesh_data(mesh_data[0], xedges, self.mesh_res, self.min_cellsize, 'x')
        self.add_edges_to_mesh_mesh_data(mesh_data[1], yedges, self.mesh_res, self.min_cellsize, 'y')
        # self.add_edges_to_mesh_mesh_data(mesh_data[2], z, self.mesh_res, self.min_cellsize, 'z')

        # Check the lines between edges
        # self.check_lines_between_edges(mesh_data, xedges, yedges, z)

        # If no z-direction mesh_datas exist, set it to None
        if mesh_data[2] == []:
            mesh_data[2] = None

        # Initialize the final mesh_data with None for all directions
        realmesh_data = [None, None, None]

       # Assign mesh_datas to the appropriate directions based on global setup 
        if dirs is not None:
            for ny in GetMultiDirs(dirs):
                realmesh_data[ny] = mesh_data[ny] 

        metal_edge_res = None

        return (realmesh_data, dirs, metal_edge_res)
    
    def handle_otheredges(self, otheredges, unique_xedges, unique_yedges, mesh_res, max_res, mesh_data, direction, check_max_resolution=False):
        
        diagonal_edges = otheredges
        if not check_max_resolution:
            other_edges_in_range = []
            if direction == 'x':
                unique_edges = self.remove_close_unique_edges(unique_xedges, max_res)
                unique_edges = [unique_xedges[0] for unique_xedges in unique_edges]

            if direction == 'y':
                unique_edges = self.remove_close_unique_edges(unique_yedges, max_res)
                unique_edges = [unique_yedges[0] for unique_yedges in unique_edges]
            for edge in diagonal_edges:            
                if direction == 'x':
                    start , end = edge[0], edge[1]
                    other_edges_in_range = [other_edge for other_edge in diagonal_edges if (start <= other_edge[0] <= end or start <= other_edge[1] <= end or start >= other_edge[0] >= end or start >= other_edge[1] >= end)]
                if direction == 'y':
                    start , end = edge[2], edge[3]
                    other_edges_in_range = [other_edge for other_edge in diagonal_edges if (start <= other_edge[2] <= end or start <= other_edge[3] <= end or start >= other_edge[2] >= end or start >= other_edge[3] >= end)]
                x_start, x_end, y_start, y_end, prim = edge
                lines_in_range = [direction for direction in unique_edges if start <= direction <= end or start >= direction >= end]

                if not other_edges_in_range:
                    alpha = np.atan(abs((y_end-y_start))/abs((x_end-x_start)))
                    resolution = mesh_res * np.cos(alpha)
                    resolution = self.max_res if resolution < self.max_res else resolution
                    if not lines_in_range:
                        lines=SmoothMeshLines([start, end], resolution)    
                    if lines_in_range:
                        lines_in_range.extend([start, end])
                        lines=SmoothMeshLines(lines_in_range, resolution)
                    mesh_data.extend(lines)
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
                    lines_in_mesh_data_range = [direction for direction in mesh_data if min_line <= direction <= max_line] 
                    for line in lines_in_mesh_data_range:
                        mesh_data.remove(line)
                    if resolution < self.max_res:
                        resolution = self.max_res
                    lines_in_range = [direction for direction in unique_edges if min_line < direction < max_line]
                    if lines_in_range:
                        lines_in_range.extend([min_line, max_line])  
                        lines=SmoothMeshLines(lines_in_range, resolution)
                    if not lines_in_range:
                        lines=SmoothMeshLines([min_line,max_line], resolution)
                    mesh_data.extend(lines)
                    if direction == 'x':
                        for alpha_val, edge in alphas_in_range:
                            if alpha_val > alpha:
                                lines_in_mesh_data_range = [line for line in mesh_data if edge[0] <= line <= edge[1] or edge[0] >= line >= edge[1]]
                                for line in lines_in_mesh_data_range:
                                    mesh_data.remove(line)
                                resolution = mesh_res * np.cos(np.deg2rad(alpha_val))
                                resolution = self.max_res if resolution < self.max_res else resolution
                                xlines = SmoothMeshLines([edge[0], edge[1]], resolution)
                                mesh_data.extend(xlines)
                    if direction == 'y':
                        for alpha_val, edge in alphas_in_range:
                            if alpha_val < alpha:
                                lines_in_mesh_data_range = [line for line in mesh_data if edge[2] <= line <= edge[3] or edge[2] >= line >= edge[3]]
                                for line in lines_in_mesh_data_range:
                                    mesh_data.remove(line)
                                resolution = mesh_res * np.sin(np.deg2rad(alpha_val))
                                resolution = self.max_res if resolution < self.max_res else resolution
                                ylines = SmoothMeshLines([edge[2], edge[3]], abs(resolution))
                                mesh_data.extend(ylines)
        circ_segments = []
        do_sort = True
        for edge in diagonal_edges:
            circ_segments = self.detect_all_circles_in_polygon(edge[4])
            if circ_segments:
                do_sort = False
        if do_sort:
            if direction == 'x':
                diagonal_edges.sort(key=lambda edge: edge[0])
            if direction == 'y':
                diagonal_edges.sort(key=lambda edge: edge[2])
        dist = []
        for i in range(len(diagonal_edges)):
            for j in range(i + 1, len(diagonal_edges)):
                line1 = diagonal_edges[i]
                line2 = diagonal_edges[j]
                dot_product = np.dot([line1[1] - line1[0], line1[3] - line1[2]], [line2[1] - line2[0], line2[3] - line2[2]])
                norm_product = np.linalg.norm([line1[1] - line1[0], line1[3] - line1[2]]) * np.linalg.norm([line2[1] - line2[0], line2[3] - line2[2]])
                cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
                angle = np.round(np.rad2deg(np.arccos(cos_angle)), 2)
                if diagonal_edges[i][4].GetElevation() != diagonal_edges[j][4].GetElevation():
                    continue
                if diagonal_edges[i][4] == diagonal_edges[j][4] and (angle not in [0, 180]):
                    if not (np.isclose(angle, 0, atol=10) or np.isclose(angle, 180, atol=10)):
                        continue
                    else:
                        if direction == 'x':
                            if not (line2[0] < line1[0] < line2[1] or  line2[0] > line1[0] > line2[1]
                            or line2[0] < line1[1] < line2[1] or  line2[0] < line1[1] < line2[1]
                            or line1[0] < line2[0] < line1[1] or  line1[0] > line2[0] > line1[1]
                            or line1[0] < line2[1] < line1[1] or  line1[0] < line2[1] < line1[1]):
                                continue
                        if direction == 'y':
                            if not (line2[2] < line1[2] < line2[3] or  line2[2] > line1[2] > line2[3]
                            or line2[2] < line1[3] < line2[3] or  line2[2] < line1[3] < line2[3]
                            or line1[2] < line2[2] < line1[3] or  line1[2] > line2[2] > line1[3]
                            or line1[2] < line2[3] < line1[3] or  line1[2] < line2[3] < line1[3]):
                                continue
                            
                if np.isclose(angle, 90, atol=1e-2):
                    continue
                p1 = np.array([line1[0], line1[2]])  # (x1, y1)
                p2 = np.array([line1[1], line1[3]])  # (x2, y2)
                q1 = np.array([line2[0], line2[2]])  # (x1, y1)
                q2 = np.array([line2[1], line2[3]])  # (x2, y2)
                distance = self.distance_between_segments(p1, p2, q1, q2)
                distance = [small_dist for small_dist in distance if small_dist[0] <= mesh_res]
                if distance:
                    dist.append(distance)
                alpha = np.round(np.rad2deg(np.atan(abs((q2[1] - q1[1]) / abs((q2[0] - q1[0]))))), 2)
                if np.min(np.diff([line1[0:2], line2[0:2]])) > mesh_res or np.min(np.diff([line1[2:4], line2[2:4]])) > mesh_res:
                    continue
        if dist and check_max_resolution:
            for dist0 in dist:
                # discretize the distance with the number of lines and calculate the minimal distance between the lines for the maximum resolution
                x = np.round(np.diff(np.linspace(0, np.min([item[0] for item in dist0]), self.num_lines)),1)
                if np.min(x) < self.max_res and np.min(x) > 0:
                    self.max_res = np.min(x)

        if dist and not check_max_resolution:    
            for dist1 in dist:
                if direction == 'x':
                    coords_of_p = [item[1][0] for item in dist1]
                    resolution = mesh_res * np.cos(np.deg2rad(alpha))
                    # start_and_end_points = [line1[0], line1[1], line2[0], line2[1]]
                    start_and_end_points = [dist1[0][2][0], dist1[0][3][0], dist1[0][4][0], dist1[0][5][0]]
                if direction == 'y':
                    coords_of_p = [item[1][1] for item in dist1]
                    resolution = mesh_res * np.sin(np.deg2rad(alpha))
                    # start_and_end_points = [line1[2], line1[3], line2[2], line2[3]]
                    start_and_end_points = [dist1[0][2][1], dist1[0][3][1], dist1[0][4][1], dist1[0][5][1]]
                lines_in_range = [lines for lines in mesh_data if np.min(coords_of_p) <= lines <= np.max(coords_of_p)]
                for line in lines_in_range:
                    mesh_data.remove(line)
                lines_before_min = [line for line in mesh_data if line < np.min(coords_of_p) and abs(line - np.min(coords_of_p)) < max_res]
                lines_after_max = [line for line in mesh_data if line > np.max(coords_of_p) and abs(line - np.max(coords_of_p)) < max_res]

                if lines_before_min and lines_after_max:
                    min_line = min(min(lines_before_min), min(lines_after_max))
                    max_line = max(max(lines_before_min), max(lines_after_max))
                    lines = self.check_edges_in_range(unique_edges, min_line, max_line, start_and_end_points, mesh_data, max_res)
                    mesh_data.extend(lines)                        
                elif lines_before_min:
                    min_line = min(min(lines_before_min), np.min(coords_of_p))
                    max_line = max(max(lines_before_min), np.max(coords_of_p))
                    lines = self.check_edges_in_range(unique_edges, min_line, max_line, start_and_end_points, mesh_data, max_res)
                    mesh_data.extend(lines)
                elif lines_after_max:
                    min_line = min(min(lines_after_max), np.min(coords_of_p))
                    max_line = max(max(lines_after_max), np.max(coords_of_p))
                    lines = self.check_edges_in_range(unique_edges, min_line, max_line, start_and_end_points, mesh_data, max_res)
                    mesh_data.extend(lines)
                else:
                    min_line = np.min(coords_of_p)
                    max_line = np.max(coords_of_p)
                    lines = self.check_edges_in_range(unique_edges, min_line, max_line, start_and_end_points, mesh_data, max_res)
                    mesh_data.extend(lines)

    def check_edges_in_range(self, unique_edges, min_line, max_line, start_and_end_points, mesh_data, resolution):
        edges_in_range = [edge for edge in unique_edges if min_line < edge < max_line]
        if edges_in_range:
            lines_in_range = [line for line in mesh_data if min(min_line, max_line, min(start_and_end_points)) <= line <= max(min_line, max_line, max(start_and_end_points))]
            if lines_in_range:
                for line in lines_in_range:
                    mesh_data.remove(line)
            edges_in_range.extend([min_line, max_line])
            edges_in_range.extend(start_and_end_points)  
            lines=SmoothMeshLines(edges_in_range, resolution/1)
        if not edges_in_range:
            lines_in_range = [line for line in mesh_data if min(min_line, max_line) <= line <= max(min_line, max_line)]
            if lines_in_range:
                for line in lines_in_range:
                    mesh_data.remove(line)
            # edges_in_range.extend(start_and_end_points)
            edges_in_range.extend([min_line, max_line])  
            lines=SmoothMeshLines(edges_in_range, resolution/1)
        return lines
    
    def check_max_resolution(self, diagonal_edges, unique_xedges, unique_yedges, mesh_res, max_res, mesh_data):
        # x-direction
        self.handle_otheredges(diagonal_edges, unique_xedges, unique_yedges, mesh_res, max_res, mesh_data[0], 'x', check_max_resolution=True)
        # y-direction
        self.handle_otheredges(diagonal_edges, unique_xedges, unique_yedges, mesh_res, max_res, mesh_data[1], 'y', check_max_resolution=True)

    def handle_circular_segments(self, polygon, mesh_data):

        circ_segments = self.found_circles

        if circ_segments:
            print(f"Gefundene Kreisabschnitte: {len(circ_segments)}")
            for i, (x_seg, y_seg) in enumerate(circ_segments):
                print(f"  Kreis #{i+1}:")
                print("    X:", x_seg)
                print("    Y:", y_seg)
        else:
            print("Kein kreisförmiger Abschnitt gefunden.")

        # delete mesh lines that are inside the circle
        if circ_segments:
            for x_seg, y_seg in circ_segments:
                # Remove x lines inside the circle
                mesh_data[0] = [line for line in mesh_data[0] if not (min(x_seg) < line < max(x_seg))]
                # Add x lines inside the circle
                mesh_data[0].extend(SmoothMeshLines([min(x_seg), max(x_seg)], self.max_res))
                # Remove y lines inside the circle
                mesh_data[1] = [line for line in mesh_data[1] if not (min(y_seg) < line < max(y_seg))]
                # Add y lines inside the circle
                mesh_data[1].extend(SmoothMeshLines([min(y_seg), max(y_seg)], self.max_res))

    def add_ports_to_mesh_data(self, mesh_data, edges, direction):
        x, y = [], []
        zedges = []
        xedges, yedges = [], []
        diagonal_edges = []
        for prim in self.primitives_mesh_setup:
            if hasattr(prim, 'priority'):
                port_coords_x, port_coords_y, port_coords_z = self.transfer_port_to_polygon(prim.start, prim.stop)
                # if prim.measplane_shift:
                #     print('prim.meas_plane_shift:', prim, prim.measplane_shift)
                x.extend(port_coords_x)
                y.extend(port_coords_y)
                self.collect_edges(port_coords_x, port_coords_y, prim, xedges, yedges, diagonal_edges)
                z = [(z, None, None, prim) for z in port_coords_z]
                zedges.extend(z)
        if direction == 'x':
            if xedges:
                edges.extend(xedges)
        if direction == 'y':
            if yedges:
                edges.extend(yedges)
        if direction == 'z': 
            if zedges:
                edges.extend(zedges)
        for edge in edges:
            if hasattr(edge[3], 'priority'):
                mesh_data.append(edge[0])
                for line in mesh_data:
                    if abs(line - edge[0]) < self.min_cellsize/2:
                        if any([e[0] == line and hasattr(e[3], 'priority') for e in edges]):
                            continue
                        elif not any([e[0] == line and hasattr(e[3], 'priority') for e in edges]):
                            if line in mesh_data:
                                mesh_data.remove(line)
        return mesh_data
        
    def add_edges_to_mesh_mesh_data(self, mesh_data, edges, mesh_res, min_cellsize, direction):

        self.remove_close_edges(edges, min_cellsize, direction)
        for edge in edges:
            dirs = self.primitives_mesh_setup.get(edge[3], {}).get('dirs') or \
                    self.properties_mesh_setup.get(edge[3].GetProperty() if hasattr(edge[3], 'GetProperty') else None, {}).get('dirs') or \
                    self.global_mesh_setup.get('dirs')
            if direction == 'x':
                if 'x' in dirs:              
                    mesh_data.append(edge[0])
            if direction == 'y':
                if 'y' in dirs:
                    mesh_data.append(edge[0])
        # mesh_data.extend(edge[0] for edge in edges)

    def remove_close_edges(self, edges, min_cellsize, direction):

        edges_to_remove = []
        for i in range(len(edges) - 1):
            if abs(edges[i+1][0] - edges[i][0]) < min_cellsize and abs(edges[i+1][0] - edges[i][0]) > 0:
                if hasattr(edges[i][3], 'priority') and hasattr(edges[i + 1][3], 'priority'):
                    if edges[i][3].priority == edges[i + 1][3].priority:
                        continue
                elif (getattr(edges[i][3], 'GetPriority', lambda: None)() or edges[i][3].priority) > (getattr(edges[i + 1][3], 'GetPriority', lambda: None)() or edges[i + 1][3].priority):
                    edges_to_remove.append(edges[i + 1])
                elif (getattr(edges[i][3], 'GetPriority', lambda: None)() or edges[i][3].priority) < (getattr(edges[i + 1][3], 'GetPriority', lambda: None)() or edges[i + 1][3].priority):
                    edges_to_remove.append(edges[i])
                else:
                    print(f"\033[91mWarning: Detected closely spaced edges at ({direction} = {edges[i][0]} and {direction} = {edges[i+1][0]}, d{direction} = {abs(edges[i][0]-edges[i+1][0])}). This configuration may lead to prolonged simulation times. Consider assigning different priorities or modifying the structure to optimize performance.\033[0m")
                    continue
                    # if abs(edges[i][2]-edges[i][1]) < abs(edges[i+1][2]-edges[i+1][1]):
                    #     edges_to_remove.append(edges[i])
                    # else:
                    #     edges_to_remove.append(edges[i + 1])
        edges_to_remove = list({edge[0]: edge for edge in edges_to_remove}.values())
        if edges_to_remove:
            edges_to_remove_first_elements = {edge[0] for edge in edges_to_remove}
            edges[:] = [edge for edge in edges if edge[0] not in edges_to_remove_first_elements]

    def remove_close_unique_edges(self, unique_edges, min_cellsize):

        unique_edges_to_remove = []
        for i in range(len(unique_edges) - 1):
            if abs(unique_edges[i+1][0] - unique_edges[i][0]) < min_cellsize:
                if hasattr(unique_edges[i][1], 'priority') and hasattr(unique_edges[i + 1][1], 'priority'):
                    if unique_edges[i][1].priority == unique_edges[i + 1][1].priority:
                        continue
                elif (getattr(unique_edges[i][1], 'GetPriority', lambda: None)() or unique_edges[i][1].priority) > (getattr(unique_edges[i + 1][1], 'GetPriority', lambda: None)() or unique_edges[i + 1][1].priority):
                    unique_edges_to_remove.append(unique_edges[i + 1])
                elif (getattr(unique_edges[i][1], 'GetPriority', lambda: None)() or unique_edges[i][1].priority) < (getattr(unique_edges[i + 1][1], 'GetPriority', lambda: None)() or unique_edges[i + 1][1].priority):
                    unique_edges_to_remove.append(unique_edges[i])
                else:
                    if unique_edges[i][0] < unique_edges[i + 1][0]:
                        unique_edges_to_remove.append(unique_edges[i])
                    else:
                        unique_edges_to_remove.append(unique_edges[i+1])
        unique_edges_to_remove = list({edge[0]: edge for edge in unique_edges_to_remove}.values())
        if unique_edges_to_remove:
            unique_edges_to_remove_first_elements = {edge[0] for edge in unique_edges_to_remove}
            unique_edges[:] = [edge for edge in unique_edges if edge[0] not in unique_edges_to_remove_first_elements]
                               
        return unique_edges

    def mesh_small_gaps(self, unique_edges, mesh_res, max_res, num_lines, mesh_data, direction):
        unique_edges.sort(key = lambda x: x[0])
        min_cellsize = self.global_mesh_setup.get('min_cellsize', None)
        use_num_lines = True if min_cellsize is None else False
        max_res_list= []
        if direction == 'z':
            for i in range(len(unique_edges) - 1):
                if abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) <= mesh_res and abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) >= max_res:
                    z_in_range = [z for z in mesh_data[2] if unique_edges[i][0] <= z <= unique_edges[i + 1][0]]
                    for z in z_in_range:
                        mesh_data[2] = list(mesh_data[2])  
                        mesh_data[2].remove(z)
                    # if use_num_lines:
                    #     new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                    #     new_max_res = np.max(new_max_res)
                    #     # max_res_list.append(new_max_res)
                    # else:
                    #     new_max_res = max_res
                    zlines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], self.max_res)
                    # if len(zlines) <= 4:
                    #     mesh_data[2] = list(mesh_data[2]) 
                    #     mesh_data[2].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                    # else:
                    mesh_data[2].extend(zlines)
        else:
            for i in range(len(unique_edges) - 1):
                if abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) <= mesh_res and abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) >= max_res and abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) >= 1.5:
                    y1, y2 = unique_edges[i][1], unique_edges[i][2]
                    y1_next, y2_next = unique_edges[i + 1][1], unique_edges[i + 1][2]
                    if (y1 <= y1_next <= y2 or y1 >= y1_next >= y2 or
                        y1 <= y2_next <= y2 or y1 >= y2_next >= y2 or
                        y1_next <= y1 <= y2_next or y1_next >= y1 >= y2_next or
                        y1_next <= y2 <= y2_next or y1_next >= y2 >= y2_next):
                            if direction == 'x':
                                x_in_range = [x for x in mesh_data[0] if unique_edges[i][0] <= x <= unique_edges[i + 1][0]]
                                for x in x_in_range:
                                    mesh_data[0].remove(x)
                                if use_num_lines:
                                    new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                                    new_max_res = np.max(new_max_res)
                                    max_res_list.append(new_max_res)
                                else:
                                    new_max_res = max_res
                                xlines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], new_max_res)
                                if len(xlines) <= 4:
                                    mesh_data[0].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                                else:
                                    mesh_data[0].extend(xlines)
                            elif direction == 'y':
                                y_in_range = [y for y in mesh_data[1] if unique_edges[i][0] <= y <= unique_edges[i + 1][0]]
                                for y in y_in_range:
                                    mesh_data[1].remove(y)
                                if use_num_lines:
                                    new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                                    new_max_res = np.max(new_max_res)
                                    max_res_list.append(new_max_res)
                                else:
                                    new_max_res = max_res
                                ylines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], new_max_res)
                                if len(ylines) <= 4:
                                    mesh_data[1].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                                else:
                                    mesh_data[1].extend(ylines)
        # if max_res_list:
        #     self.min_cellsize = min(max_res_list) - 0.25*min(max_res_list)
        #     print('self.min_cellsize:', self.min_cellsize)
        #     self.max_res = min(max_res_list)

    def add_missing_mesh_lines(self, unique_edges, sorted_points, diagonal_edges, mesh_res, mesh_data, direction):
        'Check if the first and last point are x or y edges, if not it adds the missing mesh lines between the point and the edge'
        # if unique_edges.size > 0:
        if unique_edges:
            if unique_edges[-1][0] < sorted_points[-1]:
                for other_edge in diagonal_edges:
                    if direction == 'x':
                        start, end = other_edge[0], other_edge[1]
                    if direction == 'y':
                        start, end = other_edge[2], other_edge[3]
                    if start <= sorted_points[-1] <= end or end <= sorted_points[-1] <= start:
                        if abs(np.diff([unique_edges[-1][0], min(start, end)])) < mesh_res:
                            lines = np.linspace(unique_edges[-1][0], min(start, end), 5)[1:]
                        else:
                            lines = SmoothMeshLines([unique_edges[-1][0], min(start, end)], mesh_res)[1:]
                        mesh_data.extend(lines)
            if unique_edges[0][0] > sorted_points[0]:
                for other_edge in diagonal_edges:
                    if direction == 'x':
                        start, end = other_edge[0], other_edge[1]
                    if direction == 'y':
                        start, end = other_edge[2], other_edge[3]
                    if start <= sorted_points[0] <= end or end <= sorted_points[0] <= start:
                        if abs(np.diff([unique_edges[0][0], max(start, end)])) < mesh_res:
                            lines = np.linspace(unique_edges[0][0], max(start, end), 5)[1:]
                        else:
                            lines = SmoothMeshLines([max(start, end), unique_edges[0][0]], mesh_res)
                        mesh_data.extend(lines)         
    
    def process_mesh_lines(self, grid):

        x, y, z = grid.GetLines(0), grid.GetLines(1), grid.GetLines(2)

        x_mesh_data = list(self.mesh_data.values())[0][0]
        x_mesh_data = x_mesh_data[0]  
        y_mesh_data = list(self.mesh_data.values())[0][0]
        y_mesh_data = y_mesh_data[1] 

        xmax, xmin, ymax, ymin = max(x), min(x), max(y), min(y)

        if xmax in x_mesh_data:
            x= np.append(x, xmax+self.wave_length)
        if xmin in x_mesh_data:
            x= np.append(x, xmin-self.wave_length)
        if ymax in y_mesh_data:
            y= np.append(y, ymax+self.wave_length)
        if ymin in y_mesh_data:
            y= np.append(y, ymin-self.wave_length)

        polygon = list(self.primitives_mesh_setup.keys())
        diagonal_edges = []
        xedges, yedges = [], []
        xx, yy= [], []
        if isinstance(polygon, list):
            for prim in polygon:
                self.process_primitive(prim, xx, yy, xedges, yedges, diagonal_edges)
        else:
            self.process_primitive(polygon, xx, yy, xedges, yedges, diagonal_edges)

        xedges.sort(key=lambda edge: edge[0])
        yedges.sort(key=lambda edge: edge[0]) 

        # x, y, z = self.mesh_data.get('x', [x, None]), self.mesh_data.get('y', [y, None]), self.mesh_data.get('z', [z, None]) 
        # print('mesh_data:', [value[0][0] for value in list(self.mesh_data.values())])
        # x, y, z = [value[0][0] for value in list(self.mesh_data.values())], [value[0][1] for value in list(self.mesh_data.values())], [value[0][2] for value in list(self.mesh_data.values())]
        # # z = [item for sublist in z for item in sublist if sublist is not None]
        # x = [item for sublist in x for item in sublist if sublist is not None]
        # y = [item for sublist in y for item in sublist if sublist is not None]
        # if not x:
        #     x = grid.GetLines(0)
        # if not y:
        #     y = grid.GetLines(1)
        # if z[0] is None:
        #     z = grid.GetLines(2)
        # else:
        #     z = [item for sublist in z for item in sublist if sublist is not None]

        zz_tuples = [(z, None) for z in z]
        mesh_data = [[], [], z]
        # self.mesh_small_gaps(zz_tuples, self.mesh_res, self.max_res, self.num_lines, mesh_data, 'z')
        z = np.append(mesh_data[2], z)
        z = np.unique(z)
        lines = [[SmoothMeshLines(x, self.max_cellsize, 1.3)], [SmoothMeshLines(y, self.max_cellsize/2, 1.3)], [SmoothMeshLines(z, self.max_cellsize, 1.3)]]
        for i in range(1, len(np.diff(lines[2][0])) - 1):
            # check if the difference between two consecutive z values is greater than 2 times the difference between the next two consecutive z values
            if i + 1 < len(lines[2][0]) and np.round(np.diff(lines[2][0])[i] / np.diff(lines[2][0])[i + 1], 1) > 2 and np.diff(lines[2][0])[i] > self.min_cellsize:
                lines[2][0] = list(lines[2][0])  # Convert to list
                lines[2][0].extend(SmoothMeshLines([lines[2][0][i], lines[2][0][i + 1]], self.mesh_res/2, 1.3))
                # Check lines between x edges
        for i in range(len(xedges) - 1):
            if abs(xedges[i][0] - xedges[i + 1][0]) > self.mesh_res:
                lines_in_range = [line for line in lines[0][0] if xedges[i][0] < line < xedges[i + 1][0]]
                if not lines_in_range:
                    lines[0][0] = np.append(lines[0][0], np.linspace(xedges[i][0], xedges[i + 1][0], self.num_lines))
        # Check lines between y edges
        for i in range(len(yedges) - 1):
            if abs(yedges[i][0] - yedges[i + 1][0]) > self.mesh_res:
                lines_in_range = [line for line in lines[1][0] if yedges[i][0] < line < yedges[i + 1][0]]
                if not lines_in_range:
                    lines[1][0] = np.append(lines[1][0], np.linspace(yedges[i][0], yedges[i + 1][0], self.num_lines))

        z = [(z, None) for z in z]
        x = [(x, None ) for x in lines[0][0]]
        y = [(y, None) for y in y]
        # lines[0][0] = self.process_mesh_data(lines[0][0], self.min_cellsize, x)
        # lines[1][0] = self.process_mesh_data(lines[1][0], self.min_cellsize, y)
        # lines[2][0] = self.process_mesh_data(lines[2][0], self.min_cellsize, z)

        lines[2][0] = np.unique(lines[2][0])
        return lines 
    
    def tranfer_box_to_polygon(self, box):
        start = np.fmin(box.GetStart(), box.GetStop())
        stop = np.fmax(box.GetStart(), box.GetStop())
        x_coords = [start[0], stop[0], stop[0], start[0], start[0]]
        y_coords = [start[1], start[1], stop[1], stop[1], start[1]]
        z_coords = [float(start[2]), float(stop[2])]
        return x_coords, y_coords, z_coords
    
    def transfer_port_to_polygon(self, start, stop):
        port_coords_x = [start[0], stop[0], stop[0], start[0], start[0]]
        port_coords_y = [start[1], start[1], stop[1], stop[1], start[1]]
        port_coords_z = [start[2], stop[2]]
        return port_coords_x, port_coords_y, port_coords_z
    
    def process_primitive(self, prim, x, y, xedges, yedges, diagonal_edges):
        if not hasattr(prim, 'GetType'):
            port_coords_x, port_coords_y, port_coords_z = self.transfer_port_to_polygon(prim.start, prim.stop)
            x.extend(port_coords_x)
            y.extend(port_coords_y)
            self.collect_edges(port_coords_x, port_coords_y, prim, xedges, yedges, diagonal_edges)
        elif prim.GetType() == CSPrimitives.BOX:
            box_coords_x, box_coords_y, box_coords_z = self.tranfer_box_to_polygon(prim)
            x.extend(box_coords_x)
            y.extend(box_coords_y)
            self.collect_edges(box_coords_x, box_coords_y, prim, xedges, yedges, diagonal_edges)
        else:
            xx, yy = prim.GetCoords()[0], prim.GetCoords()[1]
            x.extend(xx)
            y.extend(yy)
            if xx[-1] != xx[0] or yy[-1] != yy[0]:
                xx = np.append(xx, xx[0])
                yy = np.append(yy, yy[0])
            self.collect_edges(xx, yy, prim, xedges, yedges, diagonal_edges)

    def collect_edges(self, x_coords, y_coords, prim, xedges, yedges, diagonal_edges):
        for i in range(len(x_coords) - 1):
            if x_coords[i] != x_coords[i + 1] and y_coords[i] != y_coords[i + 1]:
                diagonal_edges.append([x_coords[i], x_coords[i + 1], y_coords[i], y_coords[i + 1], prim])
            if x_coords[i] == x_coords[i + 1]:
                xedges.append([x_coords[i], y_coords[i], y_coords[i + 1], prim])
            if y_coords[i] == y_coords[i + 1]:
                yedges.append([y_coords[i], x_coords[i], x_coords[i + 1], prim])

    def collect_z_coordinates(self, polygon):
        z = [(prim.GetElevation(), prim) for prim in polygon if hasattr(prim, 'GetType') and prim.GetType() != CSPrimitives.BOX]
        z.extend((prim.GetElevation() + prim.GetLength(), prim) for prim in polygon if hasattr(prim, 'GetType') and prim.GetType() == CSPrimitives.LINPOLY)
        box_coords_z = [(self.tranfer_box_to_polygon(prim)[2][0], prim) for prim in polygon if hasattr(prim, 'GetType') and prim.GetType() == CSPrimitives.BOX]
        box_coords_z.extend((self.tranfer_box_to_polygon(prim)[2][1], prim) for prim in polygon if hasattr(prim, 'GetType') and prim.GetType() == CSPrimitives.BOX)
        z = list(set(z))
        z.sort(key=lambda x: x[0])
        z.extend(box_coords_z)
        return z

    def process_z_coordinates(self, z, mesh_data):
        for z_val, prim in z:
            dirs = self.primitives_mesh_setup.get(prim, {}).get('dirs') or \
            self.properties_mesh_setup.get(prim.GetProperty(), {}).get('dirs') or \
            self.global_mesh_setup.get('dirs')
            if dirs is not None and 'z' in dirs:
                mesh_data[2].append(z_val)            

    def process_single_polygon(self, polygon, x, y, xedges, yedges, diagonal_edges):
        xx, yy = polygon.GetCoords()[0], polygon.GetCoords()[1]

        x = np.append(x, xx)
        y = np.append(y, yy)
        for i in range(len(xx) - 1):
            if xx[i] != xx[i + 1] and yy[i] != yy[i + 1]:
                diagonal_edges.append([xx[i], xx[i + 1], yy[i], yy[i + 1], polygon])
            if xx[i] == xx[i + 1]:
                xedges.append([xx[i], yy[i], yy[i + 1], polygon])
            if yy[i] == yy[i + 1]:
                yedges.append([yy[i], xx[i], xx[i + 1], polygon])         

    def get_mesh_parameters(self):
        self.wave_length = None
        def get_mesh_res():

            fstart = self.global_mesh_setup.get('start_frequency', None)
            fstop = self.global_mesh_setup.get('stop_frequency', None)
            f0 = self.global_mesh_setup.get('f0', None)
            fc = self.global_mesh_setup.get('fc', None)
            unit = self.global_mesh_setup.get('drawing_unit', 1e-6)
            if fstart is not None and fstop is not None:
                self.wave_length = (C0/unit) / fstop
            elif f0 is not None and fc is not None:
                self.wave_length = (C0/unit) / (f0+fc)
            else:
                raise ValueError('Please provide start and stop frequency or f0 and fc in the global mesh setup')
            epsilon = 1
            # for primitive in self.primitives_mesh_setup.keys():
            #     if hasattr(primitive, 'GetProperty') and hasattr(primitive.GetProperty(), 'GetMaterialProperty') and primitive.GetProperty().GetMaterialProperty('epsilon') > 1:
            #         current_epsilon = primitive.GetProperty().GetMaterialProperty('epsilon')
            #         # self.primitives_with_epsilon[primitive] = current_epsilon
            #         if epsilon is None or current_epsilon > epsilon:
            #             epsilon = current_epsilon
            mesh_res = self.global_mesh_setup.get('mesh_resolution', 'medium')
            if mesh_res == 'low':
                mesh_res = self.wave_length / (15 * epsilon**0.5)
                num_lines = 4
            elif mesh_res == 'medium':
                mesh_res = self.wave_length / (20 * epsilon**0.5) 
                num_lines = 5
            elif mesh_res == 'high':
                mesh_res = self.wave_length / (25 * epsilon**0.5)
                num_lines = 6
            elif mesh_res == 'very_high':
                mesh_res = self.wave_length / (30 * epsilon**0.5)
                num_lines = 7
            else:
                mesh_res = self.wave_length / (20 * epsilon**0.5)
                num_lines = 5
            # print('primitives_with_epsilon:', self.primitives_with_epsilon)
            return mesh_res, num_lines
        
        mesh_res = self.global_mesh_setup.get('refined_cellsize', None)
        if mesh_res is not None:
            mesh_resolution = self.global_mesh_setup.get('mesh_resolution', None)
            if mesh_resolution is not None:
                num_lines = get_mesh_res()[1]
            else:
                num_lines = 5
        if mesh_res is None:
            mesh_res, num_lines = get_mesh_res()

        max_cellsize = self.global_mesh_setup.get('max_cellsize', get_mesh_res()[0])    
        min_cellsize = self.global_mesh_setup.get('min_cellsize', mesh_res / 4)
        max_res = min_cellsize + 0.25 * min_cellsize
        return mesh_res, min_cellsize, max_res, max_cellsize, num_lines               

    def process_mesh_data(self, mesh_data, min_cellsize, unique_edges):
        mesh_data = sorted(mesh_data)
        while True:  
            new_mesh_data = []
            skip_next = False
            changed = False 

            for i in range(len(mesh_data) - 1):
                if skip_next:
                    skip_next = False
                    continue

                if abs(mesh_data[i+1] - mesh_data[i]) < min_cellsize / 2:
                    changed = True  
                    if any(mesh_data[i] == edge[0] for edge in unique_edges) and not any(mesh_data[i+1] == edge[0] for edge in unique_edges):
                        new_mesh_data.append(mesh_data[i])
                        skip_next = True
                    elif any(mesh_data[i+1] == edge[0] for edge in unique_edges) and not any(mesh_data[i] == edge[0] for edge in unique_edges):
                        new_mesh_data.append(mesh_data[i+1])
                        skip_next = True
                    elif any(mesh_data[i] == edge[0] for edge in unique_edges) and any(mesh_data[i+1] == edge[0] for edge in unique_edges):
                        # new_mesh_data.append((mesh_data[i] + mesh_data[i+1]) / 2)
                        skip_next = False
                        continue
                    else:
                        new_mesh_data.append((mesh_data[i] + mesh_data[i+1]) / 2)
                        skip_next = True
                else:
                    new_mesh_data.append(mesh_data[i])

            if not skip_next and mesh_data:
                new_mesh_data.append(mesh_data[-1])

            if not changed:
                break  

            mesh_data = new_mesh_data  

        return mesh_data

    def get_unique_edges(self, edges):

        unique_edges = [(edge[0], edge[3]) for edge in edges]
        unique_edges = list(set(unique_edges))
        unique_edges = list({edge[0]: edge for edge in unique_edges}.values())
        unique_edges.sort(key=lambda x: x[0])
        return unique_edges      
          
    def metal_edge(self, edges, polygon, mesh_res, mesh_data, dirs, metal_edge_res, direction):
        'not ready yet'


        # if metal_edge_res is not None:
        #     if unique_xedges[0] <= sorted_x[0]:
        #         mesh_data_in_range =  [mesh_data for mesh_data in mesh_data[0] if unique_xedges[0]-mer[1] <= mesh_data <= unique_xedges[0]-mer[0]]
        #         if not mesh_data_in_range:
        #             mesh_data[0].append(unique_xedges[0]-mer[1])
        #             mesh_data[0].append(unique_xedges[0]-mer[0])
        #         else:
        #             mesh_data[0] = [h for h in mesh_data[0] if h not in mesh_data_in_range]
        #             mesh_data[0].append(unique_xedges[0]-mer[1])
        #             mesh_data[0].append(unique_xedges[0]-mer[0])
        #     if unique_xedges[-1] >= sorted_x[-1]:
        #         mesh_data_in_range =  [mesh_data for mesh_data in mesh_data[0] if unique_xedges[-1]+mer[0] <= mesh_data <= unique_xedges[-1]+mer[1]]
        #         if not mesh_data_in_range:
        #             mesh_data[0].append(unique_xedges[-1]+mer[0])
        #             mesh_data[0].append(unique_xedges[-1]+mer[1])
        #         else:
        #             mesh_data[0] = [h for h in mesh_data[0] if h not in mesh_data_in_range]
        #             mesh_data[0].append(unique_xedges[-1]+mer[0])
        #             mesh_data[0].append(unique_xedges[-1]+mer[1])
        #     if unique_yedges[0] <= sorted_y[0]:
        #         mesh_data_in_range =  [mesh_data for mesh_data in mesh_data[1] if unique_yedges[0]-mer[1] <= mesh_data <= unique_yedges[0]-mer[0]]
        #         if not mesh_data_in_range:
        #             mesh_data[1].append(unique_yedges[0]-mer[1])
        #             mesh_data[1].append(unique_yedges[0]-mer[0])
        #         else:
        #             mesh_data[1] = [h for h in mesh_data[1] if h not in mesh_data_in_range]
        #             mesh_data[1].append(unique_yedges[0]-mer[1])
        #             mesh_data[1].append(unique_yedges[0]-mer[0])
        #     if unique_yedges[-1] >= sorted_y[-1]:
        #         mesh_data_in_range =  [mesh_data for mesh_data in mesh_data[1] if unique_yedges[-1]+mer[0] <= mesh_data <= unique_yedges[-1]+mer[1]]
        #         if not mesh_data_in_range:
        #             mesh_data[1].append(unique_yedges[-1]+mer[0])
        #             mesh_data[1].append(unique_yedges[-1]+mer[1])
        #         else:
        #             mesh_data[1] = [h for h in mesh_data[1] if h not in mesh_data_in_range]
        #             mesh_data[1].append(unique_yedges[-1]+mer[0])
        #             mesh_data[1].append(unique_yedges[-1]+mer[1])
        
        if isinstance(polygon, list):
            coords = [prim.GetCoords() for prim in polygon if hasattr(prim, 'GetCoords')]
            x = np.concatenate([coord[0] for coord in coords])
            y = np.concatenate([coord[1] for coord in coords])
            coords = (x, y)
        else:
            coords = polygon.GetCoords()
            x = polygon.GetCoords()[0]
            y = polygon.GetCoords()[1] 
        mer = np.array([-1.0, 2.0]) / 3 * metal_edge_res if metal_edge_res else 0
        if direction == 'x':
            min_distance_x = self.calc_min_distance(x)
        if direction == 'y':
            min_distance_y = self.calc_min_distance(y)
        if dirs is not None:
            for i in range(len(edges) - 1):
                if metal_edge_res is not None:
                    if direction == 'x':
                        condition1 = self.yline_in_polygon(coords, edges[i][0]+min_distance_x/2, edges[i][1], edges[i][2]) and not self.yline_in_polygon(coords, edges[i][0]-min_distance_x/2, edges[i][1], edges[i][2])
                        condition2 = self.yline_in_polygon(coords, edges[i][0]-min_distance_x/2, edges[i][1], edges[i][2]) and self.yline_in_polygon(coords, edges[i][0]+min_distance_x/2, edges[i][1], edges[i][2])
                    if direction == 'y':
                        condition1 = self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]+min_distance_y/2) and not self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]-min_distance_y/2)
                        condition2 = self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]-min_distance_y/2) and self.xline_in_polygon(coords, edges[i][1], edges[i][2], edges[i][0]+min_distance_y/2)
                        if i > 0 and abs(edges[i][0] - edges[i + 1][0]) > mesh_res and abs(edges[i][0] - edges[i - 1][0]) > mesh_res:
                            if condition1:
                                mesh_data_in_range =  [mesh_data for mesh_data in mesh_data if edges[i][0]-mer[1] <= mesh_data <= edges[i][0]-mer[0]]
                                if not mesh_data_in_range:
                                    mesh_data.append(edges[i][0]-mer[1])
                                    mesh_data.append(edges[i][0]-mer[0])
                                else:
                                    mesh_data = [h for h in mesh_data if h not in mesh_data_in_range]
                                    mesh_data.append(edges[i][0]-mer[1])
                                    mesh_data.append(edges[i][0]-mer[0])
                            elif condition2:
                                continue
                            else:
                                mesh_data_in_range =  [mesh_data for mesh_data in mesh_data if edges[i][0]+mer[0] <= mesh_data <= edges[i][0]+mer[1]]
                                if not mesh_data_in_range:
                                    mesh_data.append(edges[i][0]+mer[0])
                                    mesh_data.append(edges[i][0]+mer[1])
                                else:
                                    mesh_data = [h for h in mesh_data if h not in mesh_data_in_range]
                                    mesh_data.append(edges[i][0]+mer[0])
                                    mesh_data.append(edges[i][0]+mer[1])

    def distance_between_segments(self, p1, p2, q1, q2):
        p = np.linspace(p1, p2, 10)
        def point_to_line_distance(p, a, b):
            # Projektion des Punktes p auf die Linie a-b
            ap = p - a
            ab = b - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            # t = np.clip(t, 0, 1)  # Projektion auf das Segment beschränken
            closest_point = a + t * ab
            return np.linalg.norm(p - closest_point)
        distances = []
        for p_point in p:
            distances.append((point_to_line_distance(p_point, q1, q2), p_point, p1, p2, q1, q2))
        return distances
    def add_graded_mesh_lines(self, start, end, growth):

        lines = []
        if start < end:
            current = start
            lines.append(current)
            temp_cellsize = self.min_cellsize
            while current + temp_cellsize < end:
                current += temp_cellsize
                lines.append(current)
                temp_cellsize *= growth

        else:
            current = start
            lines.append(current)
            temp_cellsize = self.min_cellsize
            while current - temp_cellsize > end:
                current -= temp_cellsize
                lines.append(current)
                temp_cellsize *= growth

        return lines

    def add_graded_mesh_lines_at_material_transitions(self, edges, mesh_data, mesh_res):

        for i in range(len(edges) - 1):
            if abs(np.diff([edges[i][0], edges[i + 1][0]])) == 0 and edges[i][0] > np.min(edges[0][0]) and edges[i+1][0] < np.max(edges[-1][0]):
                if hasattr(edges[i][3], 'GetProperty') and hasattr(edges[i + 1][3], 'GetProperty'):
                    if edges[i][3].GetProperty()!= edges[i + 1][3].GetProperty():
                        if hasattr(edges[i][3].GetProperty(),'GetMaterialProperty') and not hasattr(edges[i + 1][3].GetProperty(), 'GetMaterialProperty') or \
                            hasattr(edges[i + 1][3].GetProperty(),'GetMaterialProperty') and not hasattr(edges[i][3].GetProperty(), 'GetMaterialProperty'):
                                lines = self.add_graded_mesh_lines(edges[i][0], edges[i][0]-mesh_res,1.3)
                                mesh_data.extend(lines)
                                lines = self.add_graded_mesh_lines(edges[i][0], edges[i][0]+mesh_res,1.3)
                                mesh_data.extend(lines)

                        if hasattr(edges[i][3].GetProperty(),'GetMaterialProperty') and hasattr(edges[i + 1][3].GetProperty(), 'GetMaterialProperty'):
                            if edges[i][3].GetProperty().GetMaterialProperty('epsilon') != edges[i + 1][3].GetProperty().GetMaterialProperty('epsilon'):
                                lines = self.add_graded_mesh_lines(edges[i][0], edges[i][0]-mesh_res,1.1)
                                lines = self.add_graded_mesh_lines(edges[i][0], edges[i][0]+mesh_res,1.1)
                                mesh_data.extend(lines)

    def is_circle_in_polygon(self, polygon, min_points=8, tolerance=0.01, return_segment=False):
        coords = polygon.GetCoords()
        x_coords, y_coords = np.array(coords[0]), np.array(coords[1])
        N = len(x_coords)
        for start in range(N - min_points + 1):
            for length in range(min_points, N - start + 1):
                sub_x = x_coords[start:start+length]
                sub_y = y_coords[start:start+length]
                if len(sub_x) < 3:
                    continue
                x0, y0 = np.mean(sub_x), np.mean(sub_y)
                radii = np.sqrt((sub_x - x0) ** 2 + (sub_y - y0) ** 2)
                mean_radius = np.mean(radii)
                deviation = np.abs(radii - mean_radius) / mean_radius
                if np.all(deviation < tolerance):
                    if return_segment:
                        return True, sub_x, sub_y
                    return True
        return (False, None, None) if return_segment else False


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
    
    def detect_all_circles_in_polygon(self, polygon, min_points=20, tolerance=0.01):
        """
        Findet alle kreisähnlichen Abschnitte im Polygon.

        Args:
            polygon: Das Polygon-Objekt mit GetCoords()
            min_points: Mindestanzahl von Punkten, um als Kreis zu gelten
            tolerance: Relative Toleranz für den Radiusvergleich

        Returns:
            Liste von Tupeln: [(sub_x1, sub_y1), (sub_x2, sub_y2), ...] für jeden gefundenen Kreisabschnitt
        """
        if self.global_mesh_setup.get('use_circle_detection', False) is False:
            return []
        if isinstance(polygon, list):
            coords = [prim.GetCoords() for prim in polygon if hasattr(prim, 'GetCoords')]
            x_coords = np.concatenate([coord[0] for coord in coords])
            y_coords = np.concatenate([coord[1] for coord in coords])
        else:                
            coords = polygon.GetCoords()
            x_coords = np.array(coords[0])
            y_coords = np.array(coords[1])
        N = len(x_coords)

        found_segments = []
        used_indices = set()

        for start in range(N - min_points + 1):
            for length in range(min_points, N - start + 1):
                indices = tuple(range(start, start + length))

                # Falls diese Punkte schon Teil eines gefundenen Kreises sind, überspringen
                if any(i in used_indices for i in indices):
                    continue

                sub_x = x_coords[list(indices)]
                sub_y = y_coords[list(indices)]

                # Mittelpunkt-Näherung
                x0, y0 = np.mean(sub_x), np.mean(sub_y)
                radii = np.sqrt((sub_x - x0)**2 + (sub_y - y0)**2)
                mean_radius = np.mean(radii)
                deviation = np.abs(radii - mean_radius) / mean_radius

                if np.all(deviation < tolerance):
                    found_segments.append((sub_x, sub_y))
                    used_indices.update(indices)
                    break  # Nicht überlappend: nächster Startpunkt
                
        return found_segments

