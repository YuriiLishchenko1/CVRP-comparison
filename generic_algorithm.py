import math
import random
import time
from deap import base, creator, tools, algorithms

# Функція для зчитування даних із файлу .vrp
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

# Функція для обчислення матриці відстаней
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

# Функція генетичного алгоритму
def genetic_algorithm(distance_matrix, num_clients):
    start_time = time.time()
    
    # Визначення компонентів DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(1, num_clients + 1), num_clients)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        """Обчислює загальну довжину маршруту."""
        total_distance = distance_matrix[0][individual[0]]
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i]][individual[i + 1]]
        total_distance += distance_matrix[individual[-1]][0]
        return total_distance,

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Створюємо початкову популяцію
    population = toolbox.population(n=50)

    # Запуск генетичного алгоритму
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False)

    # Знаходимо найкращий маршрут
    best_route = tools.selBest(population, k=1)[0]
    best_distance = evaluate(best_route)[0]
    execution_time = time.time() - start_time

    print(f"Генетичний алгоритм:")
    print(f"Час виконання: {execution_time:.4f} с")
    print(f"Загальна довжина маршруту: {best_distance}")
    print(f"Найкращий маршрут: {best_route}")
    return execution_time, best_distance, best_route

# Основна програма
def main():
    # Шлях до файлу .vrp
    file_path = "A-n32-k5.vrp"

    # Зчитування даних із файлу
    coordinates, demands, capacity = parse_vrp_file(file_path)

    # Обчислення матриці відстаней
    distance_matrix = calculate_distance_matrix(coordinates)

    # Кількість клієнтів (без урахування депо)
    num_clients = len(coordinates) - 1

    # Виклик генетичного алгоритму
    genetic_algorithm(distance_matrix, num_clients)

if __name__ == "__main__":
    main()
