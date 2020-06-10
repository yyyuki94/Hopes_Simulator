import random

import numpy as np
import matplotlib.pyplot as plt

# 経路コスト計算
def calc_cost(A, B):
    return np.linalg.norm(B - A)

# サブゴール生成
def generate_points(N):
    return 2 * np.random.rand(N, 2) - 1

# 経路計算
def generate_distance_matrix(start, sub_goals, goal=(0,0)):
    points = np.block([[start], [sub_goals], [np.array(goal)]])
    
    distances = np.zeros((points.shape[0], points.shape[0]), dtype=np.float32)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i, j] = calc_cost(points[i], points[j])
            
    return distances

# Nearest Neighbor法
def route_nn(start, selected_points, selected_sub, distance_matrix):
#     distances_tmp = np.zeros((route_tmp.shape[0], route_tmp.shape[0]))

#     for i in range(route_tmp.shape[0]):
#         for j in range(route_tmp.shape[0]):
#             distances_tmp[i, j] = calc_cost(route_tmp[i], route_tmp[j])

#     distances_start = np.zeros(route_tmp.shape[0])
#     for i in range(route_tmp.shape[0]):
#         distances_start[i] = calc_cost(start, route_tmp[i])
    distances_tmp = distance_matrix[selected_sub, :]
    distances_tmp = distances_tmp[:, selected_sub]
    distances_start = distance_matrix[0, selected_sub]
    
    route_nst = []
    start_point = np.argsort(distances_start)[1]
    while True:
        route_nst.append(start_point)

        if len(route_nst) == selected_points.shape[0]:
            break

        remain = list(set(range(selected_points.shape[0])) - set(route_nst))

        dist = distances_tmp[start_point]
        dist_idx = np.array(range(len(dist)))[remain]
        dist_sort = np.argsort(dist[remain])
        start_point = dist_idx[dist_sort[0]]
        
    return selected_points[route_nst], np.array(route_nst)

# 経路生成
def generate_route(start, sub_goals, distance_matrix, num_sub=None, target=None, goal=(0,0)):
    if (num_sub is None) and (target is None):
        raise ValueError(f"num_sub or target must not be None.")
    goal = np.array(goal)
    if target is None:
        selected_sub = np.random.choice(range(len(sub_goals)), num_sub, replace=False)
    else:
        selected_sub = np.array(range(len(sub_goals)))[target]
        
    route_sub, selected_sub = route_nn(start, sub_goals[selected_sub], selected_sub, distance_matrix)
    route = np.block([[start], [route_sub], [goal]])
    selected_tmp = np.empty(selected_sub.shape[0] + 2, dtype=int)
    selected_tmp[0] = 0
    selected_tmp[-1] = distance_matrix.shape[0] - 1
    selected_tmp[1:-1] = selected_sub
    return route, selected_tmp

# マップ表示
def view_map(start, sub_goals, route, goal=(0,0), weights=None):
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.3, 0.8, 0.8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.grid(True)
    ax.plot(start[0], start[1], marker="d", color="blue", markersize=10)
    ax.plot(goal[0], goal[1], marker="*", color="m", markersize=10)
    
    if weights is None:
        ax.scatter(sub_goals[:, 0], sub_goals[:, 1])
    else:
        maximum, minimum = weights.max(), weights.min()
        cmap = plt.get_cmap("hot", maximum - minimum + 1)
        gradient = np.linspace(0, 1, cmap.N)
        gradient_array = np.vstack((gradient, gradient))
        ax2 = fig.add_axes((0.1,0.1,0.8,0.05))
        ax2.imshow(gradient_array, aspect='auto', cmap=cmap)
        ax2.set_axis_off()
        for i in range(minimum, maximum+1):
            mask = weights == i
            if mask.any() == False:
                continue
            ax.scatter(sub_goals[mask, 0], sub_goals[mask, 1], color=cmap(i))
    ax.plot(route[:, 0], route[:, 1], color="green")
    plt.show()

# 2-opt法
def optimize_with_2opt(neighbor, route_list, point_idxs, distance_matrix):
    def gen_point_group(num_route):
        point_group = list(range(0, num_route-2))
        random.shuffle(point_group)
        return point_group
    
    route = route_list.copy()
    point_group = gen_point_group(route.shape[0])
    
#     distances = np.zeros((route.shape[0], route.shape[0]))

#     for i in range(route.shape[0]):
#         for j in range(route.shape[0]):
#             distances[i, j] = calc_cost(route[i], route[j])
    distances = distance_matrix[point_idxs, :].copy()
    distances = distances[:, point_idxs]
    
    while len(point_group) != 0:
        point_a = point_group.pop()
        point_b = point_a + 1

        tmp = np.array(range(route.shape[0]))
        tmp = np.delete(tmp, np.where((tmp == point_a) | (tmp == point_b) | (tmp == route.shape[0] - 1) | (tmp == 0)))
        tmp = np.delete(tmp, np.where((tmp > point_a + neighbor) | (tmp < point_a - neighbor)))

        point_c = np.random.choice(tmp)
        point_d = point_c + 1

        cost_ab = distances[point_a, point_b]
        cost_cd = distances[point_c, point_d]
        cost_ac = distances[point_a, point_c]
        cost_bd = distances[point_b, point_d]

        if cost_ab + cost_cd > cost_ac + cost_bd:
            route[point_b], route[point_c] = route[point_c].copy(), route[point_b].copy()
            distances[point_b, :], distances[point_c, :] = distances[point_c, :].copy(), distances[point_b, :].copy()
            distances[:, point_b], distances[:, point_c] = distances[:, point_c].copy(), distances[:, point_b].copy()
            point_group = gen_point_group(route.shape[0])

    return route