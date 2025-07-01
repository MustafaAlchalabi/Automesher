from CSXCAD.SmoothMeshLines import SmoothMeshLines
import numpy as np
import geometry_utils, meshing_utils

def smooth_and_process_mesh_lines(automesher, mesh_data, polygon, x_edges, y_edges, z, unique_xedges, unique_yedges, z_coords, mesh_map):
    
    if list(automesher.mesh_data.values()):
        mesh_data[0].extend(list(automesher.mesh_data.values())[0][0][0])
        mesh_data[1].extend(list(automesher.mesh_data.values())[0][0][1])
        mesh_data[2].extend(list(automesher.mesh_data.values())[0][0][2])

    mesh_data[0] = sorted(mesh_data[0])
    mesh_data[1] = sorted(mesh_data[1])
    mesh_data[2] = sorted(mesh_data[2])
    automesher.mesh_with_max_cell_size = [[], [], []]
    mesh_with_different_mesh_res = []

    if isinstance(polygon, list):
        if automesher.min_cellsize_changed:
        #     for prim in polygon:
        #         if hasattr(prim, 'GetProperty') and hasattr(prim.GetProperty(), 'GetMaterialProperty'):
        #             if prim.GetProperty().GetMaterialProperty('epsilon') > 1:
        #                 epsilon = prim.GetProperty().GetMaterialProperty('epsilon')
        #                 tmp_mesh_res = automesher.mesh_res / (epsilon ** 0.5)
        #                 mesh_with_different_mesh_res.append([tmp_mesh_res, prim])
        #     if mesh_with_different_mesh_res:
        #         mesh_with_different_mesh_res.sort(key=lambda x: x[0])
        #         for res, prim in reversed(mesh_with_different_mesh_res):
        #             same_prim_edges_x = [edge for edge in x_edges if edge[3] == prim]
        #             same_prim_edges_y = [edge for edge in y_edges if edge[3] == prim]
        #             same_prim_edges_z = [edge for edge in z if edge[3] == prim]
        #             for edges, idx in zip([same_prim_edges_x, same_prim_edges_y, same_prim_edges_z], range(3)):
        #                 if len(edges) > 1:
        #                     edges.sort(key=lambda edge: edge[0])
        #                     for j in range(len(edges) - 1):
        #                         lines_in_range = [line for line in mesh_data[idx] if edges[j][0] <= line <= edges[j + 1][0]]
        #                         if lines_in_range:
        #                             mesh_data[idx] = SmoothMeshLines(lines_in_range, res).tolist()

            if not any(automesher.primitives_mesh_setup.get(prim, {}).get('edges_only', False) for prim in polygon):
                for i in range(len(mesh_data[0]) - 1):
                    if mesh_data[0][i + 1] - mesh_data[0][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[0].append((mesh_data[0][i], mesh_data[0][i + 1]))
                for i in range(len(mesh_data[1]) - 1):
                    if mesh_data[1][i + 1] - mesh_data[1][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[1].append((mesh_data[1][i], mesh_data[1][i + 1]))
                for i in range(len(mesh_data[2]) - 1):
                    if mesh_data[2][i + 1] - mesh_data[2][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[2].append((mesh_data[2][i], mesh_data[2][i + 1]))

                for idx in range(3):
                    mesh_data[idx] = SmoothMeshLines(mesh_data[idx], automesher.mesh_res).tolist()
                    for start, end in automesher.mesh_with_max_cell_size[idx]:
                        mesh_data[idx] = [line for line in mesh_data[idx] if not (start < line < end)]

    else:
        if automesher.min_cellsize_changed:
            if not automesher.primitives_mesh_setup.get(polygon, {}).get('edges_only', False):
                for i in range(len(mesh_data[0]) - 1):
                    if mesh_data[0][i + 1] - mesh_data[0][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[0].append((mesh_data[0][i], mesh_data[0][i + 1]))
                for i in range(len(mesh_data[1]) - 1):
                    if mesh_data[1][i + 1] - mesh_data[1][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[1].append((mesh_data[1][i], mesh_data[1][i + 1]))
                for i in range(len(mesh_data[2]) - 1):
                    if mesh_data[2][i + 1] - mesh_data[2][i] > automesher.max_cellsize / 2:
                        automesher.mesh_with_max_cell_size[2].append((mesh_data[2][i], mesh_data[2][i + 1]))

                for idx in range(3):
                    mesh_data[idx] = SmoothMeshLines(mesh_data[idx], automesher.mesh_res).tolist()
                    for start, end in automesher.mesh_with_max_cell_size[idx]:
                        mesh_data[idx] = [line for line in mesh_data[idx] if not (start < line < end)]

    if mesh_map[0]:
        mesh_map[0].sort(key=lambda epsilon: epsilon[2], reverse=True)
    if mesh_map[1]:
        mesh_map[1].sort(key=lambda epsilon: epsilon[2], reverse=True)
    if mesh_map[2]:
        mesh_map[2].sort(key=lambda epsilon: epsilon[2], reverse=True)

    lines_to_be_smoothed = [[], [], []]
    for map in mesh_map[0]:
        max_cellsize = automesher.max_cellsize_air / map[2]**0.5
        lines_to_be_smoothed = [line for line in mesh_data[0] if map[0] <= line <= map[1]]
        if lines_to_be_smoothed:
            mesh_data[0].extend(SmoothMeshLines(lines_to_be_smoothed, max_cellsize).tolist())
    for map in mesh_map[1]:
        max_cellsize = automesher.max_cellsize_air / map[2]**0.5
        lines_to_be_smoothed = [line for line in mesh_data[1] if map[0] <= line <= map[1]]
        if lines_to_be_smoothed:
            mesh_data[1].extend(SmoothMeshLines(lines_to_be_smoothed, max_cellsize).tolist())
    for map in mesh_map[2]:
        max_cellsize = automesher.max_cellsize_air / map[2]**0.5
        lines_to_be_smoothed = [line for line in mesh_data[2] if map[0] <= line <= map[1]]
        if lines_to_be_smoothed:
            mesh_data[2].extend(SmoothMeshLines(lines_to_be_smoothed, max_cellsize).tolist())


    # if automesher.global_mesh_setup.get('min_cellsize', None) is not None or automesher.min_cellsize_changed:
    mesh_data[0] = process_mesh_data(mesh_data[0], automesher.min_cellsize, unique_xedges)
    mesh_data[1] = process_mesh_data(mesh_data[1], automesher.min_cellsize, unique_yedges)
    mesh_data[2] = process_mesh_data(mesh_data[2], automesher.min_cellsize, z_coords)

def process_mesh_data(mesh_data, min_cellsize, unique_edges):
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


def process_mesh_lines(automesher, grid):

    x, y, z = grid.GetLines(0), grid.GetLines(1), grid.GetLines(2)

    x_mesh_data = list(automesher.mesh_data.values())[0][0]
    x_mesh_data = x_mesh_data[0]  
    y_mesh_data = list(automesher.mesh_data.values())[0][0]
    y_mesh_data = y_mesh_data[1] 
    z_mesh_data = list(automesher.mesh_data.values())[0][0]
    z_mesh_data = z_mesh_data[2]


    xmax, xmin, ymax, ymin, zmax, zmin = max(x), min(x), max(y), min(y), max(z), min(z)

    polygon = list(automesher.primitives_mesh_setup.keys())
    diagonal_edges = []
    x_edges, y_edges = [], []
    xx, yy= [], []
    if isinstance(polygon, list):
        for prim in polygon:
            geometry_utils.process_primitive(prim, xx, yy, x_edges, y_edges, diagonal_edges)
    else:
        geometry_utils.process_primitive(polygon, xx, yy, x_edges, y_edges, diagonal_edges)

    x_edges.sort(key=lambda edge: edge[0])
    y_edges.sort(key=lambda edge: edge[0]) 


    zz_tuples = [(z, None) for z in z]
    mesh_data = [[], [], z]
    # self.mesh_small_gaps(zz_tuples, automesher.mesh_res, automesher.max_res, automesher.num_lines, mesh_data, 'z')
    z = np.append(mesh_data[2], z)
    z = np.unique(z)
    lines = [SmoothMeshLines(x, automesher.max_cellsize/2, 1.3), SmoothMeshLines(y, automesher.max_cellsize/2, 1.3), SmoothMeshLines(z, automesher.max_cellsize/2, 1.3)]
    # for i in range(1, len(np.diff(lines[2])) - 1):
    #     # check if the difference between two consecutive z values is greater than 2 times the difference between the next two consecutive z values
    #     if i + 1 < len(lines[2][0]) and np.round(np.diff(lines[2][0])[i] / np.diff(lines[2][0])[i + 1], 1) > 2 and np.diff(lines[2][0])[i] > automesher.min_cellsize:
    #         lines[2][0] = list(lines[2][0])  # Convert to list
    #         lines[2][0].extend(SmoothMeshLines([lines[2][0][i], lines[2][0][i + 1]], automesher.mesh_res/2, 1.3))

    # Check lines between x edges
    for i in range(len(x_edges) - 1):
        if abs(x_edges[i][0] - x_edges[i + 1][0]) > automesher.mesh_res:
            lines_in_range = [line for line in lines[0] if x_edges[i][0] < line < x_edges[i + 1][0]]
            if not lines_in_range:
                lines[0] = np.append(lines[0], np.linspace(x_edges[i][0], x_edges[i + 1][0], automesher.num_lines))
    # Check lines between y edges
    for i in range(len(y_edges) - 1):
        if abs(y_edges[i][0] - y_edges[i + 1][0]) > automesher.mesh_res:
            lines_in_range = [line for line in lines[1] if y_edges[i][0] < line < y_edges[i + 1][0]]
            if not lines_in_range:
                lines[1] = np.append(lines[1], np.linspace(y_edges[i][0], y_edges[i + 1][0], automesher.num_lines))

#     automesher.global_mesh_setup:'boundary_distance': [ 1000, 1000, 1000, 1000, 1000, 1000 ], # value, auto or None
    graded_lines_y = []
    graded_lines_x = []
    graded_lines_z = []

    distance = automesher.global_mesh_setup.get('boundary_distance', [0, 0, 0, 0, 0, 0])
    for i in range(len(distance)):
        if distance[i] == 'auto':
            distance[i] = automesher.wave_length
        elif distance[i] is None:
            distance[i] = 0

    if xmax in x_mesh_data:
        x= np.append(x, xmax+distance[0])
        graded_lines_x.extend(meshing_utils.add_graded_mesh_lines(np.max(lines[0]), xmax+distance[0], abs(np.max(lines[0])- lines[0][np.argmax(lines[0]) - 1]), automesher.max_cellsize_air, 1.3))
    if xmin in x_mesh_data:
        x= np.append(x, xmin-distance[1])
        graded_lines_x.extend(meshing_utils.add_graded_mesh_lines(np.min(lines[0]), xmin-distance[1], abs(np.min(lines[0]) - lines[0][np.argmin(lines[0]) + 1]), automesher.max_cellsize_air, 1.3))
    if ymax in y_mesh_data:
        y= np.append(y, ymax+distance[2])
        graded_lines_y.extend(meshing_utils.add_graded_mesh_lines(np.max(lines[1]), ymax+distance[2], abs(np.max(lines[1])- lines[1][np.argmax(lines[1]) - 1]), automesher.max_cellsize_air, 1.3))
    if ymin in y_mesh_data:
        y= np.append(y, ymin-distance[3])
        graded_lines_y.extend(meshing_utils.add_graded_mesh_lines(np.min(lines[1]), ymin-distance[3], abs(np.min(lines[1]) - lines[1][np.argmin(lines[1]) + 1]), automesher.max_cellsize_air, 1.3))
    if z_mesh_data and zmax in z_mesh_data:
        z= np.append(z, zmax+distance[4])
        graded_lines_z.extend(meshing_utils.add_graded_mesh_lines(np.max(lines[2]), zmax+distance[4], abs(np.max(lines[2])- lines[2][np.argmax(lines[2]) - 1]), automesher.max_cellsize_air, 1.3))
    if z_mesh_data and zmin in z_mesh_data:
        z= np.append(z, zmin-distance[5])
        graded_lines_z.extend(meshing_utils.add_graded_mesh_lines(np.min(lines[2]), zmin-distance[5], abs(np.min(lines[2]) - lines[2][np.argmin(lines[2]) + 1]), automesher.max_cellsize_air, 1.3))

    # add  graded lines to lines list
    lines[0] = np.append(lines[0], graded_lines_x)
    lines[1] = np.append(lines[1], graded_lines_y)
    lines[2] = np.append(lines[2], graded_lines_z)
    # add z lines to lines list
    z = [(z, None) for z in z]
    x = [(x, None ) for x in lines[0]]
    y = [(y, None) for y in y]

    lines[0] = [line for line in lines[0] if x_mesh_data and all(abs(line - x) > 0.1 for x in x_mesh_data)]
    lines[1] = [line for line in lines[1] if y_mesh_data and all(abs(line - y) > 0.1 for y in y_mesh_data)]
    lines[2] = [line for line in lines[2] if z_mesh_data and all(abs(line - z) > 0.1 for z in z_mesh_data)]

    return lines 