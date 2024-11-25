import math
import time
import pulp
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

def calculate_deviation(obtained, optimal):
    return ((obtained - optimal) / optimal) * 100

import re

def read_vrp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_coord_section = False
    demand_section = False
    depot_section = False
    
    coordinates = {}
    demands = {}
    depot = None
    capacity = None
    num_vehicles = None  # Нова змінна для кількості транспортних засобів
    
    for line in lines:
        line = line.strip()
        if line.startswith('COMMENT'):
            # Пошук кількості транспортних засобів у коментарі
            match = re.search(r'No of trucks:\s*(\d+)', line)
            if match:
                num_vehicles = int(match.group(1))
        elif line.startswith('CAPACITY'):
            capacity = int(line.split()[-1])
        elif line.startswith('NODE_COORD_SECTION'):
            node_coord_section = True
            demand_section = False
            depot_section = False
            continue
        elif line.startswith('DEMAND_SECTION'):
            node_coord_section = False
            demand_section = True
            depot_section = False
            continue
        elif line.startswith('DEPOT_SECTION'):
            node_coord_section = False
            demand_section = False
            depot_section = True
            continue
        elif line.startswith('EOF'):
            break
        
        if node_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates[idx] = (x, y)
        elif demand_section:
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                demand = int(parts[1])
                demands[idx] = demand
        elif depot_section:
            depot_idx = int(line)
            if depot_idx != -1:
                depot = depot_idx
    
    return coordinates, demands, depot, capacity, num_vehicles



def cvrp_pulp(coordinates, demands, depot, capacity, num_vehicles):
    nodes = list(coordinates.keys())
    customers = [n for n in nodes if n != depot]
    
    # Обчислення відстаней між вузлами
    distances = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                xi, yi = coordinates[i]
                xj, yj = coordinates[j]
                distances[i, j] = math.hypot(xi - xj, yi - yj)
    
    # Ініціалізація проблеми
    problem = pulp.LpProblem("CVRP", pulp.LpMinimize)
    
    # Змінні
    x = pulp.LpVariable.dicts('x', (nodes, nodes), cat='Binary')
    u = pulp.LpVariable.dicts('u', customers, lowBound=0, upBound=capacity, cat='Continuous')
    
    # Цільова функція
    problem += pulp.lpSum([distances[i, j] * x[i][j] for i in nodes for j in nodes if i != j])
    
    # Обмеження
    for j in customers:
        problem += pulp.lpSum([x[i][j] for i in nodes if i != j]) == 1  # Кожен клієнт відвідується один раз
    for i in customers:
        problem += pulp.lpSum([x[i][j] for j in nodes if i != j]) == 1
    for i in nodes:
        problem += x[i][i] == 0  # Заборона петльових маршрутів
    
    # Обмеження на кількість транспортних засобів
    problem += pulp.lpSum([x[depot][j] for j in nodes if j != depot]) == num_vehicles  # Вихід з депо
    problem += pulp.lpSum([x[i][depot] for i in nodes if i != depot]) == num_vehicles  # Повернення до депо
    
    # Обмеження на місткість та під-тури (MTZ обмеження)
    for i in customers:
        problem += u[i] >= demands[i]
        problem += u[i] <= capacity
    for i in customers:
        for j in customers:
            if i != j:
                problem += u[i] - u[j] + capacity * x[i][j] <= capacity - demands[j]
    
    # Розв'язання
    solver = pulp.PULP_CBC_CMD(timeLimit=600)
    problem.solve(solver)
    
    # Перевірка статусу
    print("Статус рішення:", pulp.LpStatus[problem.status])
    print("Об'єктивна функція:", pulp.value(problem.objective))
    
    if problem.status == pulp.LpStatusInfeasible or problem.status == pulp.LpStatusUnbounded:
        print("Рішення не знайдено або задача не обмежена")
        return None, None
    
    # Отримання маршрутів
    edges = [(i, j) for i in nodes for j in nodes if i != j and pulp.value(x[i][j]) > 0.5]
    
    print("Побудова графа маршрутів...")
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Пошук маршрутів
    routes = []
    for component in nx.weakly_connected_components(G):
        subgraph = G.subgraph(component)
        cycle = list(nx.simple_cycles(subgraph))
        if cycle:
            route = cycle[0]
            route.append(route[0])  # Додаємо повернення до початкового вузла
            routes.append(route)
        else:
            path = list(nx.dfs_edges(subgraph, source=depot))
            route = [depot] + [v for u, v in path]
            if route[-1] != depot:
                route.append(depot)
            routes.append(route)
    
    total_distance = pulp.value(problem.objective)
    return routes, total_distance


def cvrp_nearest_neighbor(coordinates, demands, depot, capacity, num_vehicles):
    nodes = list(coordinates.keys())
    customers = [n for n in nodes if n != depot]
    unvisited = set(customers)
    routes = []
    total_distance = 0
    
    while unvisited and len(routes) < num_vehicles:
        route = [depot]
        load = 0
        current_node = depot
        while True:
            candidates = [node for node in unvisited if load + demands[node] <= capacity]
            if not candidates:
                break
            # Знаходимо найближчого сусіда
            next_node = min(candidates, key=lambda node: math.hypot(
                coordinates[current_node][0] - coordinates[node][0],
                coordinates[current_node][1] - coordinates[node][1]))
            route.append(next_node)
            load += demands[next_node]
            unvisited.remove(next_node)
            total_distance += math.hypot(
                coordinates[current_node][0] - coordinates[next_node][0],
                coordinates[current_node][1] - coordinates[next_node][1])
            current_node = next_node
        # Повернення до депо
        total_distance += math.hypot(
            coordinates[current_node][0] - coordinates[depot][0],
            coordinates[current_node][1] - coordinates[depot][1])
        route.append(depot)
        routes.append(route)
    
    # Якщо залишилися невідвідані клієнти після використання всіх транспортних засобів
    if unvisited:
        print("Не вдалося обслугувати всіх клієнтів з заданою кількістю транспортних засобів.")
        return routes, None  # Або підніміть виняток
    
    return routes, total_distance



def save_results(routes, total_distance, execution_time, num_vehicles, filename):
    with open(filename, 'w') as f:
        f.write(f'Total Distance: {total_distance}\n')
        f.write(f'Execution Time: {execution_time} seconds\n')
        f.write(f'Number of Vehicles: {num_vehicles}\n')
        for i, route in enumerate(routes):
            f.write(f'Route {i+1}: {route}\n')


def plot_routes(coordinates, routes, depot, title='Маршрути CVRP'):
    plt.figure(figsize=(10, 8))
    # Відображення всіх вузлів
    for node, (x, y) in coordinates.items():
        plt.plot(x, y, 'ko')
        plt.text(x + 0.5, y + 0.5, str(node), fontsize=9)
    # Відображення маршрутів
    colors = plt.get_cmap('tab20')
    for idx, route in enumerate(routes):
        x_coords = [coordinates[node][0] for node in route]
        y_coords = [coordinates[node][1] for node in route]
        plt.plot(x_coords, y_coords, marker='o', color=colors(idx % 20), label=f'Маршрут {idx+1}')
    plt.plot(coordinates[depot][0], coordinates[depot][1], marker='s', markersize=10, color='red', label='Депо')
    plt.title(title)
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True)
    plt.show()



def run_and_evaluate(algorithm, coordinates, demands, depot, capacity):
    start_time = time.time()
    routes, total_distance = algorithm(coordinates, demands, depot, capacity)
    end_time = time.time()
    execution_time = end_time - start_time
    num_vehicles = len(routes) if routes else 0
    return {
        'routes': routes,
        'total_distance': total_distance,
        'execution_time': execution_time,
        'num_vehicles': num_vehicles
    }
    
    
def main():
    # Замість 'P-n20-k2.vrp' використайте ваш файл
    file_path = 'A-n32-k5.vrp'
    coordinates, demands, depot, capacity, num_vehicles = read_vrp_file(file_path)
    
    if num_vehicles is None:
        print("Кількість транспортних засобів не вказана в файлі.")
        return
    
    # Точний алгоритм
    print("Виконання точного алгоритму...")
    results_exact = run_and_evaluate(
        lambda coords, dem, dep, cap: cvrp_pulp(coords, dem, dep, cap, num_vehicles),
        coordinates, demands, depot, capacity)
    
    if results_exact['routes'] is not None:
        print("Маршрути точного алгоритму:")
        print(results_exact['routes'])
        save_results(results_exact['routes'], results_exact['total_distance'],
                     results_exact['execution_time'], results_exact['num_vehicles'], 'exact_results.txt')
        plot_routes(coordinates, results_exact['routes'], depot, title='Точний алгоритм')
    else:
        print("Точний алгоритм не знайшов рішення.")
    
    # Метаевристичний алгоритм
    print("Виконання метаевристичного алгоритму...")
    results_heuristic = run_and_evaluate(
        lambda coords, dem, dep, cap: cvrp_nearest_neighbor(coords, dem, dep, cap, num_vehicles),
        coordinates, demands, depot, capacity)
    print("Маршрути метаевристичного алгоритму:")
    print(results_heuristic['routes'])
    save_results(results_heuristic['routes'], results_heuristic['total_distance'],
                 results_heuristic['execution_time'], results_heuristic['num_vehicles'], 'heuristic_results.txt')
    plot_routes(coordinates, results_heuristic['routes'], depot, title='Метаевристичний алгоритм')
    
    # Створення таблиці результатів
    data = {
        'Algorithm': ['Exact', 'Heuristic'],
        'Total Distance': [results_exact['total_distance'] if results_exact['routes'] else 'N/A',
                           results_heuristic['total_distance']],
        'Execution Time': [results_exact['execution_time'] if results_exact['routes'] else 'N/A',
                           results_heuristic['execution_time']],
        'Number of Vehicles': [results_exact['num_vehicles'] if results_exact['routes'] else 'N/A',
                               results_heuristic['num_vehicles']]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('comparison_results.csv', index=False)
    print(df)

if __name__ == '__main__':
    main()
