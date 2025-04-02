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
        # self.ports_mesh_setup = {}
        self.mesh_data = {}

    def GenMesh(self, CSX, global_mesh_setup, primitives_mesh_setup, properties_mesh_setup, **kw):

        self.properties_mesh_setup = properties_mesh_setup
        self.primitives_mesh_setup = primitives_mesh_setup
        self.global_mesh_setup = global_mesh_setup
        # self.ports_mesh_setup = ports_mesh_setup

        csx = CSX
        grid = csx.GetGrid()

        unique_primitives = list(self.primitives_mesh_setup.keys())
        unique_properties = list(self.properties_mesh_setup.keys())
        only_edges = []
        for primitive, mesh_hint in self.primitives_mesh_setup.items():
            if mesh_hint.get('edges_only', False):
                unique_primitives.remove(primitive)
                only_edges.append(primitive)
        if only_edges:
            if len(only_edges) == 1:
                self.collect_mesh_data(only_edges[0], csx, **kw)
            else:
                self.collect_mesh_data_for_multiple_primitives(only_edges, csx, **kw)

        if len(unique_primitives) == 1:
            self.collect_mesh_data(unique_primitives[0], csx, **kw)
        else:
            self.collect_mesh_data_for_multiple_primitives(unique_primitives, csx, **kw)


        self.create_mesh_lines(grid)
        lines = self.process_mesh_lines(grid)

        for i in range(3):
            grid.AddLine(i, lines[i][0])

    def collect_mesh_data_for_multiple_primitives(self, primitives, csx, **kw):
        hint = None
        (hint,dirs,metal_edge_res) = self.mesh_hint_from_primitives(primitives, csx,  **kw)
        if hint is not None:
            self.mesh_data[tuple(primitives)] = (hint,dirs,metal_edge_res)

    def collect_mesh_data(self, primitive, csx, **kw):
        hint = None
        if primitive.GetType() == CSPrimitives.POINT:
            hint = self.mesh_hint_from_point(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.BOX:
            hint = self.mesh_hint_from_box(primitive, **kw)
        elif primitive.GetType() == CSPrimitives.POLYGON:
            (hint,dirs,metal_edge_res) = self.mesh_hint_from_primitives(primitive,  csx, **kw)
        elif primitive.GetType() == CSPrimitives.LINPOLY:
            (hint,dirs,metal_edge_res) = self.mesh_hint_from_primitives(primitive,  csx, **kw)
        
        if hint is not None:
            self.mesh_data[primitive] = (hint,dirs,metal_edge_res)

    def create_mesh_lines(self, grid):
        for primitive in self.mesh_data:
            for n in range(3):
                if self.mesh_data[primitive][0][n]:
                    grid.AddLine(n, self.mesh_data[primitive][0][n])

    def mesh_hint_from_primitives(self, polygon, csx, **kw):

        # Initialize the hint list for x, y, and z directions
        hint = [[], [], []]

        # Initialize lists to store edges and coordinates
        otheredges = []
        xedges, yedges = [], []
        x, y= [], []

        # Check if the input is a list of primitives
        if isinstance(polygon, list):
            # Process each primitive
            for prim in polygon:
                self.process_primitive(prim, x, y, xedges, yedges, otheredges)

            # Collect z-coordinates from the polygon
            z = self.collect_z_coordinates(polygon)

            # Process z-coordinates and add them to the hint list
            self.process_z_coordinates(z, hint)

            # Get the mesh setup parameters
            dirs = self.global_mesh_setup.get('dirs')
            # metal_edge_res = self.primitives_mesh_setup.get(polygon[0], {}).get('metal_edge_res') or \
                            # self.properties_mesh_setup.get(polygon[0].GetProperty(), {}).get('metal_edge_res') or \
                            # self.global_mesh_setup.get('metal_edge_res')
        else:
            # If the polygon is a single primitive, get its x and y coordinates
            x, y = polygon.GetCoords()[0], polygon.GetCoords()[1]

            # Process the single polygon to extract edges and coordinates
            self.process_single_polygon(polygon,x, y, xedges, yedges, otheredges)

           # Collect z-coordinates from the single polygon 
            z = self.collect_z_coordinates([polygon])

            # Process z-coordinates and add them to the hint
            self.process_z_coordinates(z, hint)
            dirs = self.global_mesh_setup.get('dirs')
            # metal_edge_res = self.primitives_mesh_setup.get(polygon, {}).get('metal_edge_res') or \
            #                 self.properties_mesh_setup.get(polygon.GetProperty(), {}).get('metal_edge_res') or \
            #                 self.global_mesh_setup.get('metal_edge_res')

        # Get the mesh parameters
        mesh_res, min_cellsize, max_res, max_cellsize, num_lines = self.get_mesh_parameters()
        # mer = np.array([-1.0, 2.0]) / 3 * metal_edge_res if metal_edge_res else 0

        # Get unique vertical and horizontal edges
        unique_xedges, unique_yedges = self.get_unique_edges(xedges), self.get_unique_edges(yedges)
        
        # Sort x and y coordinates
        sorted_x, sorted_y = np.sort(x), np.sort(y)

        # Handle diagonal edges and add mesh lines for x and y directions
        self.handle_otheredges(otheredges, unique_xedges, unique_yedges, mesh_res, max_res, hint[0], 'x')
        self.handle_otheredges(otheredges, unique_xedges, unique_yedges, mesh_res, max_res, hint[1], 'y')
        
        # Refine mesh in tight areas for x and y edges
        self.mesh_tight_areas(xedges, mesh_res, max_res, num_lines, hint, 'x')
        self.mesh_tight_areas(yedges, mesh_res, max_res, num_lines, hint, 'y')
        # self.mesh_tight_areas(z, mesh_res, max_res,num_lines, hint, 'z')
        
        # Recalculate unique edges after refinement 
        unique_xedges, unique_yedges = self.get_unique_edges(xedges), self.get_unique_edges(yedges)
        
        # Sort edges by their starting coordinates
        xedges.sort(key=lambda edge: edge[0])
        yedges.sort(key=lambda edge: edge[0])   

        # Add missing mesh lines between points and edges for x and y directions
        self.add_missing_mesh_lines(unique_xedges, sorted_x, otheredges, mesh_res, hint[0], 'x')
        self.add_missing_mesh_lines(unique_yedges, sorted_y, otheredges, mesh_res, hint[1], 'y')
        
        # Add edges to the mesh hint for x and y directions
        self.add_edges_to_mesh_hint(hint[0], xedges, mesh_res, min_cellsize, 'x')
        self.add_edges_to_mesh_hint(hint[1], yedges, mesh_res, min_cellsize, 'y')

        # self.metal_edge(xedges, polygon, mesh_res, hint[0], dirs, metal_edge_res, 'x')
        # self.metal_edge(yedges, polygon, mesh_res, hint[1], dirs, metal_edge_res, 'y')
    
        # if metal_edge_res is not None:
        #     if unique_xedges[0] <= sorted_x[0]:
        #         hint_in_range =  [hint for hint in hint[0] if unique_xedges[0]-mer[1] <= hint <= unique_xedges[0]-mer[0]]
        #         if not hint_in_range:
        #             hint[0].append(unique_xedges[0]-mer[1])
        #             hint[0].append(unique_xedges[0]-mer[0])
        #         else:
        #             hint[0] = [h for h in hint[0] if h not in hint_in_range]
        #             hint[0].append(unique_xedges[0]-mer[1])
        #             hint[0].append(unique_xedges[0]-mer[0])
        #     if unique_xedges[-1] >= sorted_x[-1]:
        #         hint_in_range =  [hint for hint in hint[0] if unique_xedges[-1]+mer[0] <= hint <= unique_xedges[-1]+mer[1]]
        #         if not hint_in_range:
        #             hint[0].append(unique_xedges[-1]+mer[0])
        #             hint[0].append(unique_xedges[-1]+mer[1])
        #         else:
        #             hint[0] = [h for h in hint[0] if h not in hint_in_range]
        #             hint[0].append(unique_xedges[-1]+mer[0])
        #             hint[0].append(unique_xedges[-1]+mer[1])
        #     if unique_yedges[0] <= sorted_y[0]:
        #         hint_in_range =  [hint for hint in hint[1] if unique_yedges[0]-mer[1] <= hint <= unique_yedges[0]-mer[0]]
        #         if not hint_in_range:
        #             hint[1].append(unique_yedges[0]-mer[1])
        #             hint[1].append(unique_yedges[0]-mer[0])
        #         else:
        #             hint[1] = [h for h in hint[1] if h not in hint_in_range]
        #             hint[1].append(unique_yedges[0]-mer[1])
        #             hint[1].append(unique_yedges[0]-mer[0])
        #     if unique_yedges[-1] >= sorted_y[-1]:
        #         hint_in_range =  [hint for hint in hint[1] if unique_yedges[-1]+mer[0] <= hint <= unique_yedges[-1]+mer[1]]
        #         if not hint_in_range:
        #             hint[1].append(unique_yedges[-1]+mer[0])
        #             hint[1].append(unique_yedges[-1]+mer[1])
        #         else:
        #             hint[1] = [h for h in hint[1] if h not in hint_in_range]
        #             hint[1].append(unique_yedges[-1]+mer[0])
        #             hint[1].append(unique_yedges[-1]+mer[1])
        if list(self.mesh_data.values()):
            hint[0].extend(list(self.mesh_data.values())[0][0][0])
            hint[1].extend(list(self.mesh_data.values())[0][0][1])
            hint[2].extend(list(self.mesh_data.values())[0][0][2])
        # Smooth the mesh lines for x, y, and z directions if edges_only is not set
        hint[0] = sorted(hint[0])
        hint[1] = sorted(hint[1])
        hint[2] = sorted(hint[2])
        mesh_with_max_resolution = [[], [], []] 

        if isinstance(polygon, list):
            if not any (self.primitives_mesh_setup.get(prim, {}).get('edges_only', False) for prim in polygon):
                for i in range(len(hint[0]) - 1):
                    if hint[0][i+1] - hint[0][i] > max_cellsize:
                        mesh_with_max_resolution[0].append((hint[0][i], hint[0][i+1]))
                for i in range(len(hint[1]) - 1):
                    if hint[1][i+1] - hint[1][i] > max_cellsize:
                        mesh_with_max_resolution[1].append((hint[1][i], hint[1][i+1]))
                for i in range(len(hint[2]) - 1):
                    if hint[2][i+1] - hint[2][i] > max_cellsize:
                        mesh_with_max_resolution[2].append((hint[2][i], hint[2][i+1]))

                hint[0] = SmoothMeshLines(hint[0], mesh_res).tolist()    
                hint[1] = SmoothMeshLines(hint[1], mesh_res).tolist()
                hint[2] = SmoothMeshLines(hint[2], mesh_res).tolist()

                for i in range(3):
                    for start, end in mesh_with_max_resolution[i]:
                        hint[i] = [line for line in hint[i] if not (start <= line <= end)]

        else:
            if not self.primitives_mesh_setup.get(polygon, {}).get('edges_only', False):
                hint[0] = SmoothMeshLines(hint[0], mesh_res).tolist()    
                hint[1] = SmoothMeshLines(hint[1], mesh_res).tolist()
                hint[2] = SmoothMeshLines(hint[2], mesh_res).tolist()

        # Process the hint to remove close points and refine the mesh        
        self.process_hint(hint[0], min_cellsize, unique_xedges)
        self.process_hint(hint[1], min_cellsize, unique_yedges)
        self.process_hint(hint[2], min_cellsize, z)

        # If no z-direction hints exist, set it to None
        if hint[2] == []:
            hint[2] = None

        # Initialize the final hint with None for all directions
        realhint = [None, None, None]

       # Assign hints to the appropriate directions based on global setup 
        if dirs is not None:
            for ny in GetMultiDirs(dirs):
                realhint[ny] = hint[ny] 

        metal_edge_res = None

        return (realhint, dirs, metal_edge_res)
    
    def handle_otheredges(self, otheredges, unique_xedges, unique_yedges, mesh_res, max_res, hint, direction):
        other_edges_in_range = []
        if direction == 'x':
            unique_edges = self.remove_close_unique_edges(unique_xedges, max_res)
            unique_edges = [unique_xedges[0] for unique_xedges in unique_edges]

        if direction == 'y':
            unique_edges = self.remove_close_unique_edges(unique_yedges, max_res)
            unique_edges = [unique_yedges[0] for unique_yedges in unique_edges]
        for edge in otheredges:            
            if direction == 'x':
                start , end = edge[0], edge[1]
                other_edges_in_range = [other_edge for other_edge in otheredges if (start <= other_edge[0] <= end or start <= other_edge[1] <= end or start >= other_edge[0] >= end or start >= other_edge[1] >= end)]
            if direction == 'y':
                start , end = edge[2], edge[3]
                other_edges_in_range = [other_edge for other_edge in otheredges if (start <= other_edge[2] <= end or start <= other_edge[3] <= end or start >= other_edge[2] >= end or start >= other_edge[3] >= end)]
            x_start, x_end, y_start, y_end, prim = edge
            lines_in_range = [direction for direction in unique_edges if start <= direction <= end or start >= direction >= end]

            if not other_edges_in_range:
                alpha = np.atan(abs((y_end-y_start))/abs((x_end-x_start)))
                resolution = mesh_res * np.cos(alpha)
                if not lines_in_range:
                    lines=SmoothMeshLines([start, end], resolution)    
                if lines_in_range:
                    lines_in_range.extend([start, end])
                    lines=SmoothMeshLines(lines_in_range, resolution)
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
                if not lines_in_range:
                    lines=SmoothMeshLines([min_line,max_line], resolution)
                hint.extend(lines)
                if direction == 'x':
                    for alpha_val, edge in alphas_in_range:
                        if alpha_val > alpha:
                            lines_in_hint_range = [line for line in hint if edge[0] <= line <= edge[1] or edge[0] >= line >= edge[1]]
                            for line in lines_in_hint_range:
                                hint.remove(line)
                            resolution = mesh_res * np.cos(np.deg2rad(alpha_val))
                            xlines = SmoothMeshLines([edge[0], edge[1]], resolution)
                            hint.extend(xlines)
                if direction == 'y':
                    for alpha_val, edge in alphas_in_range:
                        if alpha_val < alpha:
                            lines_in_hint_range = [line for line in hint if edge[2] <= line <= edge[3] or edge[2] >= line >= edge[3]]
                            for line in lines_in_hint_range:
                                hint.remove(line)
                            resolution = mesh_res * np.sin(np.deg2rad(alpha_val))
                            print ('resolution:', resolution)
                            ylines = SmoothMeshLines([edge[2], edge[3]], abs(resolution))
                            hint.extend(ylines)
        if direction == 'x':
            otheredges.sort(key=lambda edge: edge[0])
        if direction == 'y':
            otheredges.sort(key=lambda edge: edge[2])
        for i in range(len(otheredges)):
            for j in range(i + 1, len(otheredges)):
                line1 = otheredges[i]
                line2 = otheredges[j]
                dot_product = np.dot([line1[1] - line1[0], line1[3] - line1[2]], [line2[1] - line2[0], line2[3] - line2[2]])
                norm_product = np.linalg.norm([line1[1] - line1[0], line1[3] - line1[2]]) * np.linalg.norm([line2[1] - line2[0], line2[3] - line2[2]])
                cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
                angle = np.round(np.rad2deg(np.arccos(cos_angle)), 2)
                if otheredges[i][4].GetElevation() != otheredges[j][4].GetElevation():
                    continue
                if np.isclose(angle, 90, atol=1e-2):
                    continue
                p1 = np.array([line1[0], line1[2]])  # (x1, y1)
                p2 = np.array([line1[1], line1[3]])  # (x2, y2)
                q1 = np.array([line2[0], line2[2]])  # (x1, y1)
                q2 = np.array([line2[1], line2[3]])  # (x2, y2)
                dist = self.distance_between_segments(p1, p2, q1, q2)
                dist = [small_dist for small_dist in dist if small_dist[0] <= mesh_res]
                alpha = np.round(np.rad2deg(np.atan(abs((q2[1] - q1[1]) / abs((q2[0] - q1[0]))))), 2)
                if np.min(np.diff([line1[0:2], line2[0:2]])) > mesh_res or np.min(np.diff([line1[2:4], line2[2:4]])) > mesh_res:
                    continue
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
                    lines_before_min = [line for line in hint if line < np.min(coords_of_p) and abs(line - np.min(coords_of_p)) < max_res]
                    lines_after_max = [line for line in hint if line > np.max(coords_of_p) and abs(line - np.max(coords_of_p)) < max_res]

                    if lines_before_min and lines_after_max:
                        min_line = min(min(lines_before_min), min(lines_after_max))
                        max_line = max(max(lines_before_min), max(lines_after_max))
                        lines = self.check_edges_in_range(unique_edges, min_line, max_line, max_res)
                        hint.extend(lines)                        
                    elif lines_before_min:
                        min_line = min(min(lines_before_min), np.min(coords_of_p))
                        max_line = max(max(lines_before_min), np.max(coords_of_p))
                        lines = self.check_edges_in_range(unique_edges, min_line, max_line, max_res)
                        hint.extend(lines)
                    elif lines_after_max:
                        min_line = min(min(lines_after_max), np.min(coords_of_p))
                        max_line = max(max(lines_after_max), np.max(coords_of_p))
                        lines = self.check_edges_in_range(unique_edges, min_line, max_line, max_res)
                        hint.extend(lines)
                    else:
                        min_line = np.min(coords_of_p)
                        max_line = np.max(coords_of_p)
                        lines = self.check_edges_in_range(unique_edges, min_line, max_line, max_res)
                        hint.extend(lines)

    def check_edges_in_range(self, unique_edges, min_line, max_line, resolution):
        lines_in_range = [edge for edge in unique_edges if min_line < edge < max_line]
        if lines_in_range:
            lines_in_range.extend([min_line, max_line])  
            lines=SmoothMeshLines(lines_in_range, resolution)
        if not lines_in_range:
            lines=SmoothMeshLines([min_line,max_line], resolution)
        return lines
    
    def add_edges_to_mesh_hint(self, hint, edges, mesh_res, min_cellsize, direction):
    
        self.remove_close_edges(edges, min_cellsize)
        for edge in edges:
            dirs = self.primitives_mesh_setup.get(edge[3], {}).get('dirs') or \
                    self.properties_mesh_setup.get(edge[3].GetProperty() if hasattr(edge[3], 'GetProperty') else None, {}).get('dirs') or \
                    self.global_mesh_setup.get('dirs')
            if direction == 'x':
                if 'x' in dirs:              
                    hint.append(edge[0])
            if direction == 'y':
                if 'y' in dirs:
                    hint.append(edge[0])
        # hint.extend(edge[0] for edge in edges)

    def remove_close_edges(self, edges, min_cellsize):
        edges_to_remove = []
        for i in range(len(edges) - 1):
            if abs(edges[i+1][0] - edges[i][0]) < min_cellsize and abs(edges[i+1][0] - edges[i][0]) > 0:
                if (getattr(edges[i][3], 'GetPriority', lambda: None)() or edges[i][3].priority) > (getattr(edges[i + 1][3], 'GetPriority', lambda: None)() or edges[i + 1][3].priority):
                    edges_to_remove.append(edges[i + 1])
                elif (getattr(edges[i][3], 'GetPriority', lambda: None)() or edges[i][3].priority) < (getattr(edges[i + 1][3], 'GetPriority', lambda: None)() or edges[i + 1][3].priority):
                    edges_to_remove.append(edges[i])
                else:
                    if abs(edges[i][2]-edges[i][1]) < abs(edges[i+1][2]-edges[i+1][1]):
                        edges_to_remove.append(edges[i])
                    else:
                        edges_to_remove.append(edges[i + 1])
        edges_to_remove = list({edge[0]: edge for edge in edges_to_remove}.values())
        if edges_to_remove:
            edges_to_remove_first_elements = {edge[0] for edge in edges_to_remove}
            edges[:] = [edge for edge in edges if edge[0] not in edges_to_remove_first_elements]

    def remove_close_unique_edges(self, unique_edges, min_cellsize):
        unique_edges_to_remove = []
        for i in range(len(unique_edges) - 1):
            if abs(unique_edges[i+1][0] - unique_edges[i][0]) < min_cellsize:
                if (getattr(unique_edges[i][1], 'GetPriority', lambda: None)() or unique_edges[i][1].priority) > (getattr(unique_edges[i + 1][1], 'GetPriority', lambda: None)() or unique_edges[i + 1][1].priority):
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

    def mesh_tight_areas(self, unique_edges, mesh_res, max_res, num_lines, hint, direction):
        unique_edges.sort(key = lambda x: x[0])
        min_cellsize = self.global_mesh_setup.get('min_cellsize', None)
        use_num_lines = True if min_cellsize is None else False
        if direction == 'z':
            for i in range(len(unique_edges) - 1):
                if abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) < mesh_res and abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) > max_res:
                    z_in_range = [z for z in hint[2] if unique_edges[i][0] <= z <= unique_edges[i + 1][0]]
                    for z in z_in_range:
                        hint[2] = list(hint[2])  
                        hint[2].remove(z)
                    if use_num_lines:
                        new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                        new_max_res = np.max(new_max_res)
                    else:
                        new_max_res = max_res
                    zlines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], new_max_res)
                    # if len(zlines) <= 4:
                    #     hint[2] = list(hint[2]) 
                    #     hint[2].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                    # else:
                    hint[2].extend(zlines)
        else:
            for i in range(len(unique_edges) - 1):
                if abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) < mesh_res and abs(np.diff([unique_edges[i][0], unique_edges[i + 1][0]])) > max_res:
                    y1, y2 = unique_edges[i][1], unique_edges[i][2]
                    y1_next, y2_next = unique_edges[i + 1][1], unique_edges[i + 1][2]
                    if (y1 <= y1_next <= y2 or y1 >= y1_next >= y2 or
                        y1 <= y2_next <= y2 or y1 >= y2_next >= y2 or
                        y1_next <= y1 <= y2_next or y1_next >= y1 >= y2_next or
                        y1_next <= y2 <= y2_next or y1_next >= y2 >= y2_next):
                            if direction == 'x':
                                x_in_range = [x for x in hint[0] if unique_edges[i][0] <= x <= unique_edges[i + 1][0]]
                                for x in x_in_range:
                                    hint[0].remove(x)
                                if use_num_lines:
                                    new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                                    new_max_res = np.max(new_max_res)
                                else:
                                    new_max_res = max_res
                                xlines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], new_max_res)
                                if len(xlines) <= 4:
                                    hint[0].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                                else:
                                    hint[0].extend(xlines)
                            elif direction == 'y':
                                y_in_range = [y for y in hint[1] if unique_edges[i][0] <= y <= unique_edges[i + 1][0]]
                                for y in y_in_range:
                                    hint[1].remove(y)
                                if use_num_lines:
                                    new_max_res = np.diff(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], num_lines))
                                    new_max_res = np.max(new_max_res)
                                else:
                                    new_max_res = max_res
                                ylines = SmoothMeshLines([unique_edges[i][0], unique_edges[i + 1][0]], new_max_res)
                                if len(ylines) <= 4:
                                    hint[1].extend(np.linspace(unique_edges[i][0], unique_edges[i + 1][0], 4))
                                else:
                                    hint[1].extend(ylines)

    def add_missing_mesh_lines(self, unique_edges, sorted_points, otheredges, mesh_res, hint, direction):
        'Check if the first and last point are x or y edges, if not it adds the missing mesh lines between the point and the edge'
        # if unique_edges.size > 0:
        if unique_edges:
            if unique_edges[-1][0] < sorted_points[-1]:
                for other_edge in otheredges:
                    if direction == 'x':
                        start, end = other_edge[0], other_edge[1]
                    if direction == 'y':
                        start, end = other_edge[2], other_edge[3]
                    if start <= sorted_points[-1] <= end or end <= sorted_points[-1] <= start:
                        if abs(np.diff([unique_edges[-1][0], min(start, end)])) < mesh_res:
                            lines = np.linspace(unique_edges[-1][0], min(start, end), 5)[1:]
                        else:
                            lines = SmoothMeshLines([unique_edges[-1][0], min(start, end)], mesh_res)[1:]
                        hint.extend(lines)
            if unique_edges[0][0] > sorted_points[0]:
                for other_edge in otheredges:
                    if direction == 'x':
                        start, end = other_edge[0], other_edge[1]
                    if direction == 'y':
                        start, end = other_edge[2], other_edge[3]
                    if start <= sorted_points[0] <= end or end <= sorted_points[0] <= start:
                        if abs(np.diff([unique_edges[0][0], max(start, end)])) < mesh_res:
                            lines = np.linspace(unique_edges[0][0], max(start, end), 5)[1:]
                        else:
                            lines = SmoothMeshLines([max(start, end), unique_edges[0][0]], mesh_res)
                        hint.extend(lines)         
    
    def process_mesh_lines(self, grid):

        mesh_res, min_cellsize, max_res, max_cellsize, num_lines = self.get_mesh_parameters()
        x, y, z = grid.GetLines(0), grid.GetLines(1), grid.GetLines(2)
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
        hint = [[], [], z]
        self.mesh_tight_areas(zz_tuples, mesh_res, max_res, num_lines, hint, 'z')
        z = np.append(hint[2], z)
        z = np.unique(z)
        lines = [[SmoothMeshLines(x, max_cellsize, 1.3)], [SmoothMeshLines(y, max_cellsize, 1.3)], [SmoothMeshLines(z, max_cellsize, 1.3)]]
        for i in range(1, len(np.diff(lines[2][0])) - 1):
            # check if the difference between two consecutive z values is greater than 2 times the difference between the next two consecutive z values
            if i + 1 < len(lines[2][0]) and np.round(np.diff(lines[2][0])[i] / np.diff(lines[2][0])[i + 1], 1) > 2 and np.diff(lines[2][0])[i] > min_cellsize:
                lines[2][0] = list(lines[2][0])  # Convert to list
                lines[2][0].extend(SmoothMeshLines([lines[2][0][i], lines[2][0][i + 1]], mesh_res, 1.3))

        z = [(z, None) for z in z]
        x = [(x, None ) for x in lines[0][0]]
        y = [(y, None) for y in y]
        self.process_hint(lines[2][0], min_cellsize, z)
        self.process_hint(lines[0][0], min_cellsize, x)
        self.process_hint(lines[1][0], min_cellsize, y)

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
    
    def process_primitive(self, prim, x, y, xedges, yedges, otheredges):
        if not hasattr(prim, 'GetType'):
            port_coords_x, port_coords_y, port_coords_z = self.transfer_port_to_polygon(prim.start, prim.stop)
            x.extend(port_coords_x)
            y.extend(port_coords_y)
            self.collect_edges(port_coords_x, port_coords_y, prim, xedges, yedges, otheredges)
        elif prim.GetType() == CSPrimitives.BOX:
            box_coords_x, box_coords_y, box_coords_z = self.tranfer_box_to_polygon(prim)
            x.extend(box_coords_x)
            y.extend(box_coords_y)
            self.collect_edges(box_coords_x, box_coords_y, prim, xedges, yedges, otheredges)
        else:
            xx, yy = prim.GetCoords()[0], prim.GetCoords()[1]
            x.extend(xx)
            y.extend(yy)
            if xx[-1] != xx[0] or yy[-1] != yy[0]:
                xx = np.append(xx, xx[0])
                yy = np.append(yy, yy[0])
            self.collect_edges(xx, yy, prim, xedges, yedges, otheredges)

    def collect_edges(self, x_coords, y_coords, prim, xedges, yedges, otheredges):
        for i in range(len(x_coords) - 1):
            if x_coords[i] != x_coords[i + 1] and y_coords[i] != y_coords[i + 1]:
                otheredges.append([x_coords[i], x_coords[i + 1], y_coords[i], y_coords[i + 1], prim])
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

    def process_z_coordinates(self, z, hint):
        for z_val, prim in z:
            dirs = self.primitives_mesh_setup.get(prim, {}).get('dirs') or \
            self.properties_mesh_setup.get(prim.GetProperty(), {}).get('dirs') or \
            self.global_mesh_setup.get('dirs')
            if dirs is not None and 'z' in dirs:
                hint[2].append(z_val)            

    def process_single_polygon(self, polygon, x, y, xedges, yedges, otheredges):
        xx, yy = polygon.GetCoords()[0], polygon.GetCoords()[1]

        x = np.append(x, xx)
        y = np.append(y, yy)
        for i in range(len(xx) - 1):
            if xx[i] != xx[i + 1] and yy[i] != yy[i + 1]:
                otheredges.append([xx[i], xx[i + 1], yy[i], yy[i + 1], polygon])
            if xx[i] == xx[i + 1]:
                xedges.append([xx[i], yy[i], yy[i + 1], polygon])
            if yy[i] == yy[i + 1]:
                yedges.append([yy[i], xx[i], xx[i + 1], polygon])         

    def get_mesh_parameters(self):

        def get_mesh_res():

            fstart = self.global_mesh_setup.get('start_frequency', None)
            fstop = self.global_mesh_setup.get('stop_frequency', None)
            f0 = self.global_mesh_setup.get('f0', None)
            fc = self.global_mesh_setup.get('fc', None)
            unit = self.global_mesh_setup.get('drawing_unit', 1e-6)
            if fstart is not None and fstop is not None:
                wave_length = (C0/unit) / fstop
            elif f0 is not None and fc is not None:
                wave_length = (C0/unit) / (f0+fc)
            else:
                raise ValueError('Please provide start and stop frequency or f0 and fc in the global mesh setup')
            epsilon = 1
            for primitive in self.primitives_mesh_setup.keys():
                if hasattr(primitive, 'GetProperty') and hasattr(primitive.GetProperty(), 'GetMaterialProperty') and primitive.GetProperty().GetMaterialProperty('epsilon') > 1:
                    current_epsilon = primitive.GetProperty().GetMaterialProperty('epsilon')
                    if epsilon is None or current_epsilon > epsilon:
                        epsilon = current_epsilon
            mesh_res = self.global_mesh_setup.get('mesh_resolution', 'medium')
            if mesh_res == 'low':
                mesh_res = wave_length / (15 * epsilon**0.5)
                num_lines = 4
            elif mesh_res == 'medium':
                mesh_res = wave_length / (20 * epsilon**0.5) 
                num_lines = 5
            elif mesh_res == 'high':
                mesh_res = wave_length / (25 * epsilon**0.5)
                num_lines = 6
            elif mesh_res == 'very_high':
                mesh_res = wave_length / (30 * epsilon**0.5)
                num_lines = 7
            else:
                mesh_res = wave_length / (20 * epsilon**0.5)
                num_lines = 5
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

    def process_hint(self, hint, min_cellsize, unique_edges):

        hint = list(hint)  # Convert to a Python list for element deletion
        i = 0
        while i < len(hint) - 1:
            if abs(hint[i] - hint[i+1]) < min_cellsize / 2:
                if any(hint[i] == edge[0] for edge in unique_edges) and not any(hint[i+1] == edge[0] for edge in unique_edges):
                    del hint[i+1]
                elif any(hint[i+1] == edge[0] for edge in unique_edges) and not any(hint[i] == edge[0] for edge in unique_edges):
                    del hint[i]
                elif any(hint[i] == edge[0] for edge in unique_edges) and any(hint[i+1] == edge[0] for edge in unique_edges):
                    hint[i] = (hint[i] + hint[i+1]) / 2
                    del hint[i+1]
                else:
                    hint[i] = (hint[i] + hint[i+1]) / 2
                    del hint[i+1]
            else:
                i += 1
        hint = np.array(hint)  # Convert back to a NumPy array if needed

    def get_unique_edges(self, edges):

        unique_edges = [(edge[0], edge[3]) for edge in edges]
        unique_edges = list(set(unique_edges))
        unique_edges = list({edge[0]: edge for edge in unique_edges}.values())
        unique_edges.sort(key=lambda x: x[0])
        return unique_edges      
          
    def metal_edge(self, edges, polygon, mesh_res, hint, dirs, metal_edge_res, direction):
        'not ready yet'
        if isinstance(polygon, list):
            coords = [prim.GetCoords() for prim in polygon]
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
                                hint_in_range =  [hint for hint in hint if edges[i][0]-mer[1] <= hint <= edges[i][0]-mer[0]]
                                if not hint_in_range:
                                    hint.append(edges[i][0]-mer[1])
                                    hint.append(edges[i][0]-mer[0])
                                else:
                                    hint = [h for h in hint if h not in hint_in_range]
                                    hint.append(edges[i][0]-mer[1])
                                    hint.append(edges[i][0]-mer[0])
                            elif condition2:
                                continue
                            else:
                                hint_in_range =  [hint for hint in hint if edges[i][0]+mer[0] <= hint <= edges[i][0]+mer[1]]
                                if not hint_in_range:
                                    hint.append(edges[i][0]+mer[0])
                                    hint.append(edges[i][0]+mer[1])
                                else:
                                    hint = [h for h in hint if h not in hint_in_range]
                                    hint.append(edges[i][0]+mer[0])
                                    hint.append(edges[i][0]+mer[1])

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
    
