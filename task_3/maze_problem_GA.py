#Petter Eriksson, 2024-10-8 peer22@student.bth.se
import numpy as np
import random

# Constants
START = (0, 0)
END = (4, 3)
WALL_DATA = [
    # (Top, Right, Bottom, Left) for each square
    [(1000, 1, 1, 1000), (1000, 1, 1000, 1), (1000, 1000, 1, 1), (1000, 1, 1000, 1000), (1000, 1000, 1, 1)],
    [(1, 1000, 1, 1000), (1000, 1, 1000, 1000), (1, 1000, 1, 1), (1000, 1, 1, 1000), (1, 1000, 1, 1)],
    [(1, 1, 1000, 1000), (1000, 1, 1, 1), (1, 1, 1000, 1), (1, 1000, 1, 1), (1, 1000, 1000, 1000)],
    [(1000, 1, 1, 1000), (1, 1000, 1, 1), (1000, 1, 1, 1000), (1, 1, 1000, 1), (1000, 1000, 1, 1)],
    [(1, 1000, 1000, 1000), (1, 1, 1000, 1000), (1, 1000, 1000, 1), (1000, 1, 1000, 1000), (1, 1000, 1000, 1)]
]

# movement directions
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
DIRECTION_NAMES = ['right', 'down', 'left', 'up']

# Function to check if a move is valid
def is_valid_move(position, direction):
    x, y = position
    if x < 0 or x >= len(WALL_DATA) or y < 0 or y >= len(WALL_DATA[0]):
        return False
    
    wall_data = WALL_DATA[x][y]
    if not isinstance(wall_data, tuple) or len(wall_data) != 4:
        print(f"Unexpected wall data at ({x}, {y}): {wall_data}")
        return False

    top, right, bottom, left = wall_data
    
    if direction == (0, 1):  # right
        return right == 1
    elif direction == (1, 0):  # down
        return bottom == 1
    elif direction == (0, -1):  # left
        return left == 1
    elif direction == (-1, 0):  # up
        return top == 1
    return False

def create_random_path(length):
    path = []
    current_position = START
    for _ in range(length):
        possible_moves = [direction for direction in DIRECTIONS if is_valid_move(current_position, direction)]
        if possible_moves:
            direction = random.choice(possible_moves)
            new_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            
            if new_position == END:
                # If the endpoint is reached, add the move and return the path
                path.append(direction)
                return path
            
            path.append(direction)
            current_position = new_position
        else:
            break
    return path

def evaluate_path(path):
    current_position = START
    distance = 0
    for direction in path:
        if is_valid_move(current_position, direction):
            new_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            current_position = new_position
            distance += 1
            if current_position == END:
                return distance  # Return distance if reached the end
        else:
            # Penalize heavily for an invalid move
            distance += 10  # High penalty for invalid move
            return float('inf')  # Stop evaluation immediately on invalid move
    return float('inf') if current_position != END else distance

# Function to select paths based on fitness
def select_paths(population):
    sorted_population = sorted(population, key=lambda p: evaluate_path(p))
    return sorted_population[:len(sorted_population) // 2]

# Function to crossover two paths
def crossover(path1, path2):
    crossover_point = random.randint(1, min(len(path1), len(path2)) - 1)
    return path1[:crossover_point] + path2[crossover_point:]

# Function to mutate a path
def mutate(path):
    if random.random() < 0.1:  # 10% chance to mutate
        index = random.randint(0, len(path) - 1)
        path[index] = random.choice(DIRECTIONS)

# Genetic Algorithm to find the best path
def genetic_algorithm(population_size, path_length, generations):
    population = [create_random_path(path_length) for _ in range(population_size)]
    
    # Print initial population
    print("Initial population:")
    for path in population:
        print([DIRECTION_NAMES[DIRECTIONS.index(dir)] for dir in path])
    
    for generation in range(generations):
        fitnesses = [evaluate_path(path) for path in population]
        min_fitness = min(fitnesses)
        print(f"Generation {generation}: Best fitness = {min_fitness}")
        
        # Check if any path reached the end
        found_end = False
        for path, fitness in zip(population, fitnesses):
            if fitness == len(get_path_positions(path)) and get_path_positions(path)[-1] == END:
                print("Path found reaching the endpoint!")
                found_end = True
                best_path = path  # Store the best path found in this generation
                break
        
        if found_end:
            # Stop the evaluation of the current generation since we found an endpoint
            continue
        
        selected = select_paths(population)
        next_population = selected.copy()
        
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            next_population.append(child)
        
        population = next_population

    # If we found a valid path that reached the endpoint, return it, otherwise find the best overall path
    if 'best_path' in locals():
        return best_path
    else:
        best_path = min(population, key=lambda p: evaluate_path(p))
        return best_path

def get_path_positions(path):
    current_position = START
    positions = [current_position]
    for direction in path:
        if is_valid_move(current_position, direction):
            current_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            positions.append(current_position)
        else:
            break  # Stops if an invalid move is encountered
    return positions

population_size = 100
path_length = 50  # Maximum number moves
generations = 200

best_path = genetic_algorithm(population_size, path_length, generations)

best_path_coordinates = get_path_positions(best_path)

# Evaluate the best path to get the distance
best_path_distance = evaluate_path(best_path)

# Print the best path and its distance
print("Best path (coordinates):", best_path_coordinates)
print("Distance to end:", best_path_distance)

# Optional: Print the best path directions, limited to the distance
best_path_directions = [DIRECTION_NAMES[DIRECTIONS.index(direction)] for direction in best_path[:best_path_distance]]
print("Best path (directions):", best_path_directions)