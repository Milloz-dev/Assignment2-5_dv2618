#Petter Eriksson, 2024-10-8 peer22@student.bth.se
import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
START = (0, 0)
END = (6, 18)
WALL_DATA = [
    # (Top, Right, Bottom, Left) for each square
    #row 0
    [(1000, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1, 1), (1000, 1, 1000, 1), (1000, 1, 1000, 1),
    (1000, 1000, 1, 1), (1000, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1, 1), (1000, 1, 1000, 1), (1000, 1000, 1, 1),
    (1000, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1, 1), (1000, 1, 1, 1),
    (1000, 1, 1, 1), (1000, 1000, 1000, 1)],

    #row 1
    [(1000, 1, 1, 1000), (1000, 1, 1, 1), (1000, 1, 1, 1), (1, 1000, 1, 1), (1000, 1, 1, 1000), (1000, 1000, 1, 1), (1, 1, 1, 1000),
    (1000, 1000, 1, 1), (1000, 1, 1, 1000), (1, 1000, 1, 1), (1000, 1000, 1, 1000), (1, 1, 1, 1000), (1000, 1, 1, 1),
    (1000, 1000, 1, 1), (1000, 1, 1, 1000), (1000, 1, 1, 1), (1, 1, 1, 1), (1, 1000, 1, 1), (1, 1, 1, 1000), (1000, 1000, 1, 1)],

    #row 2
    [(1, 1, 1000, 1000), (1, 1, 1000, 1), (1, 1000, 1, 1), (1, 1, 1000, 1000), (1, 1, 1, 1), (1, 1, 1000, 1), (1, 1000, 1000, 1),
    (1, 1, 1000, 1000), (1, 1000, 1000, 1), (1, 1, 1, 1000), (1, 1, 1000, 1), (1, 1, 1000, 1), (1, 1, 1000, 1), (1, 1000, 1, 1),
    (1, 1, 1000, 1000), (1, 1, 1000, 1), (1, 1, 1000, 1), (1, 1, 1000, 1), (1, 1000, 1000, 1), (1, 1000, 1, 1000)],

    #row 3
    [(1000, 1, 1, 1000), (1000, 1000, 1, 1), (1, 1, 1000, 1000), (1000, 1000, 1, 1), (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1000, 1),
    (1000, 1, 1000, 1), (1000, 1000, 1, 1), (1, 1000, 1, 1000), (1000, 1, 1, 1000), (1000, 1, 1000, 1), (1000, 1000, 1, 1),
    (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1, 1), (1000, 1, 1000, 1), (1, 1000, 1000, 1)],

    #row 4
    [(1, 1000, 1, 1000), (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1, 1000, 1000, 1), (1000, 1, 1, 1000), (1000, 1000, 1, 1), (1000, 1, 1, 1000),
    (1000, 1000, 1, 1), (1, 1000, 1, 1000), (1, 1000, 1, 1000), (1, 1000, 1, 1000), (1000, 1, 1, 1000), (1, 1000, 1, 1), (1000, 1, 1, 1000),
    (1000, 1000, 1, 1), (1000, 1, 1, 1000), (1000, 1, 1000, 1), (1, 1000, 1, 1), (1000, 1, 1, 1000), (1000, 1000, 1, 1)],

    #row 5
    [(1, 1000, 1, 1000), (1000, 1, 1, 1000), (1000, 1, 1000, 1), (1000, 1000, 1, 1), (1, 1000, 1, 1000), (1, 1, 1000, 1000), (1, 1000, 1000, 1),
    (1, 1000, 1, 1000), (1, 1, 1000, 1000), (1, 1000, 1000, 1), (1, 1000, 1, 1000), (1, 1000, 1, 1000), (1, 1, 1000, 1000), (1, 1000, 1000, 1),
    (1, 1000, 1, 1000), (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1, 1, 1, 1), (1, 1000, 1000, 1), (1, 1000, 1, 1000)],

    #row 6
    [(1, 1000, 1000, 1000), (1, 1, 1000, 1000), (1000, 1000, 1000, 1), (1, 1, 1000, 1000), (1, 1000, 1000, 1), (1000, 1, 1000, 1000), (1000, 1, 1000, 1),
    (1, 1, 1000, 1), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1, 1000, 1000, 1), (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1000, 1000, 1),
    (1, 1, 1000, 1000), (1000, 1, 1000, 1), (1000, 1, 1000, 1), (1, 1000, 1000, 1), (1000, 1, 1000, 1000), (1, 1000, 1000, 1)]                
    ]

# movement directions
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
DIRECTION_NAMES = ['right', 'down', 'left', 'up']

def is_valid_move(position, direction, last_direction=None):
    x, y = position
    
    # Check if the position is within bounds
    if x < 0 or x >= len(WALL_DATA) or y < 0 or y >= len(WALL_DATA[0]):
        return False
    
    wall_data = WALL_DATA[x][y]
    
    # Check if the wall data is valid
    if not isinstance(wall_data, tuple) or len(wall_data) != 4:
        print(f"Unexpected wall data at ({x}, {y}): {wall_data}")
        return False

    # Check if the move is a backward move
    if last_direction is not None:
        backward_moves = {
            (0, 1): (0, -1),   # right -> left
            (1, 0): (-1, 0),   # down -> up
            (0, -1): (0, 1),    # left -> right
            (-1, 0): (1, 0)     # up -> down
        }
        if backward_moves[direction] == last_direction:
            return False

    top, right, bottom, left = wall_data
    
    # Check for valid moves based on wall data
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
    last_direction = None  # Track the last direction taken
    for _ in range(length):
        possible_moves = [direction for direction in DIRECTIONS if is_valid_move(current_position, direction, last_direction)]
        if possible_moves:
            direction = random.choice(possible_moves)
            new_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            
            if new_position == END:
                # If the endpoint is reached, add the move and return the path
                path.append(direction)
                return path
            
            path.append(direction)
            last_direction = direction  # Update the last direction taken
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

def select_paths(population):
    # Tournament selection
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, 5)
        winner = min(tournament, key=evaluate_path)
        selected.append(winner)
    return selected

def crossover(path1, path2):
    # Multi-point crossover for more variety
    crossover_points = random.sample(range(1, min(len(path1), len(path2)) - 1), 2)
    crossover_points.sort()
    return path1[:crossover_points[0]] + path2[crossover_points[0]:crossover_points[1]] + path1[crossover_points[1]:]

def mutation(solution, mutation_rate):
    mutated_solution = solution.copy()
    
    # Start mutating from a random index
    start_index = random.randint(0, len(mutated_solution) - 1)
    
    for i in range(start_index, len(mutated_solution)):
        if random.random() < mutation_rate:
            valid_moves = [direction for direction in DIRECTIONS if is_valid_move(mutated_solution[i], direction)]
            if valid_moves:
                mutated_solution[i] = random.choice(valid_moves)
                
    return mutated_solution


def genetic_algorithm(population_size, path_length, generations):
    # Create initial population
    population = [create_random_path(path_length) for _ in range(population_size)]
    best_overall_path = min(population, key=evaluate_path)
    best_overall_distance = evaluate_path(best_overall_path)
    
    stagnation_counter = 0

    for generation in range(generations):
        fitnesses = [evaluate_path(path) for path in population]
        min_fitness = min(fitnesses)

        # Update best overall path if a better one is found
        if min_fitness < best_overall_distance:
            best_overall_distance = min_fitness
            best_overall_path = population[fitnesses.index(min_fitness)]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        print(f"Generation {generation}: Best fitness = {min_fitness}")

        # Preserve the best path ,(elitism)
        sorted_population = sorted(population, key=evaluate_path)
        next_population = sorted_population[:2]  # Keep the top 2 for elitism

        # Select parents for the next generation
        selected = select_paths(population)

        # Create the next generation
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            # Dynamic mutation rate
            mutation_rate = 0.1 if stagnation_counter < 10 else 0.2  
            mutated_child = mutation(child, mutation_rate)
            next_population.append(mutated_child)

        population = next_population
        
        # random paths if stagnation occurs
        if stagnation_counter > 35:
            random_injections = [create_random_path(path_length) for _ in range(population_size // 10)]
            population[-len(random_injections):] = random_injections
            stagnation_counter = 0

    return best_overall_path

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

def draw_maze(maze_data, start=None, end=None, path=None):
    rows = len(maze_data)
    cols = len(maze_data[0])
    fig, ax = plt.subplots(figsize=(cols, rows))

    for row in range(rows):
        for col in range(cols):
            cell = maze_data[row][col]
            x, y = col, rows - row - 1  # This ensures row 0 is at the top

            # Draw each wall based on the value
            # Top wall
            if cell[0] == 1000:
                ax.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=2)
            # Right wall
            if cell[1] == 1000:
                ax.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=2)
            # Bottom wall
            if cell[2] == 1000:
                ax.plot([x, x + 1], [y, y], color="black", linewidth=2)
            # Left wall
            if cell[3] == 1000:
                ax.plot([x, x], [y, y + 1], color="black", linewidth=2)

    # Draw the start point
    if start is not None:
        ax.scatter(start[1] + 0.5, rows - start[0] - 0.5, color='blue', s=100, label='Start')

    # Draw the end point
    if end is not None:
        ax.scatter(end[1] + 0.5, rows - end[0] - 0.5, color='red', s=100, label='End')
    # Draw the path
    if path is not None:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [rows - p[0] - 0.5 for p in path]
        ax.plot(path_x, path_y, color='green', linewidth=3, label='Path')

    # Set limits and aspect
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.legend()
    plt.show()

# Parameters for the genetic algorithm
population_size = 100
path_length = 100
generations = 1000

best_path = genetic_algorithm(population_size, path_length, generations)

best_path_coordinates = get_path_positions(best_path)

# Evaluate the best path to get the distance
best_path_distance = evaluate_path(best_path)

# best path and its distance
print("Best path (coordinates):", best_path_coordinates)
print("Distance to end:", best_path_distance)

# Optional: Print the best path directions, limited to the distance
if best_path_distance != float('inf'):
    best_path_directions = [DIRECTION_NAMES[DIRECTIONS.index(direction)] for direction in best_path[:best_path_distance]]
    print("Best path (directions):", best_path_directions)

else:
    print("No valid path found.")

draw_maze(WALL_DATA, start=START, end=END, path=best_path_coordinates)