import math
import time

# Функція для парсингу даних із файлу .vrp
def parse_vrp_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    coordinates = {}
    demands = {}
    capacity = 0
    section = None

    for line in data:
        line = line.strip()
        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif line == "NODE_COORD_SECTION":
            section = "COORDS"
        elif line == "DEMAND_SECTION":
            section = "DEMANDS"
        elif line == "DEPOT_SECTION":
            section = "DEPOT"
        elif line == "EOF":
            break
        elif section == "COORDS":
            parts = line.split()
            node = int(parts[0])
            x, y = map(float, parts[1:])
            coordinates[node] = (x, y)
        elif section == "DEMANDS":
            parts = line.split()
            node = int(parts[0])
            demand = int(parts[1])
            demands[node] = demand

    return coordinates, demands, capacity

# Функція для створення матриці відстаней
def calculate_distance_matrix(coordinates):
    nodes = list(coordinates.keys())
    n = len(nodes)
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coordinates[nodes[i]]
                x2, y2 = coordinates[nodes[j]]
                distance_matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return distance_matrix

# Алгоритм "Savings"
def calculate_savings(distance_matrix, depot=0):
    n = len(distance_matrix)
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = distance_matrix[i][depot] + distance_matrix[depot][j] - distance_matrix[i][j]
            savings.append((i, j, s))
    return sorted(savings, key=lambda x: -x[2])  # Сортуємо за спаданням "економії"

def savings_method(distance_matrix, demands, capacity):
    start_time = time.time()
    savings = calculate_savings(distance_matrix)
    routes = [[i] for i in range(1, len(distance_matrix))]
    route_demands = {tuple(route): demands[route[0]] for route in routes}
    
    while savings:
        i, j, s = savings.pop(0)
        route_i = next((route for route in routes if i in route), None)
        route_j = next((route for route in routes if j in route), None)
        
        if route_i != route_j:
            # Загальний попит об'єднаних маршрутів
            total_demand = route_demands[tuple(route_i)] + route_demands[tuple(route_j)]
            if total_demand <= capacity:
                # Об'єднуємо маршрути
                route_i.extend(route_j)
                routes.remove(route_j)
                route_demands[tuple(route_i)] = total_demand

    # Обчислюємо загальну довжину маршрутів
    total_distance = 0
    for route in routes:
        total_distance += distance_matrix[0][route[0]]  # Від депо до першого клієнта
        for k in range(len(route) - 1):
            total_distance += distance_matrix[route[k]][route[k + 1]]  # Від клієнта до клієнта
        total_distance += distance_matrix[route[-1]][0]  # Від останнього клієнта до депо

    execution_time = time.time() - start_time
    print(f"Метод 'Savings': Час виконання: {execution_time:.4f} с, Загальна довжина маршруту: {total_distance}")
    print("Маршрути:", routes)
    return execution_time, total_distance, routes

# Основна програма
file_path = "A-n80-k10.vrp"
coordinates, demands, capacity = parse_vrp_file(file_path)
distance_matrix = calculate_distance_matrix(coordinates)

# Перетворюємо demands у список
demands_list = [demands[node] for node in sorted(coordinates.keys())]

# Викликаємо метод "Savings"
savings_method(distance_matrix, demands_list, capacity)
