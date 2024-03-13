import random
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product

# Define puzzle state representation
class PuzzleState:
    def __init__(self, state):
        self.state = state
        self.size = len(state)
        self.blank_position = self.find_blank_position()

    def find_blank_position(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j] == 0:
                    return (i, j)

    def swap(self, pos1, pos2):
        temp = self.state[pos1[0]][pos1[1]]
        self.state[pos1[0]][pos1[1]] = self.state[pos2[0]][pos2[1]]
        self.state[pos2[0]][pos2[1]] = temp
        self.blank_position = pos2

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(tuple(map(tuple, self.state)))

# Define cost function (number of misplaced tiles)
def cost(state):
    n = len(state)
    goal_state = np.arange(n * n).reshape(n, n)
    return np.sum(state != goal_state)

# Define simulated annealing algorithm
def simulated_annealing(initial_state, temperature=1.0, cooling_rate=0.999, min_temperature=0.01):
    current_state = PuzzleState(initial_state)
    current_cost = cost(current_state.state)
    start_time = time.time()

    state_transitions = [initial_state]  # Record state transitions

    while temperature > min_temperature:
        next_state = PuzzleState(current_state.state)
        move_blank_tile(next_state)
        next_cost = cost(next_state.state)

        delta_cost = next_cost - current_cost
        if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temperature):
            current_state = next_state
            current_cost = next_cost

        temperature *= cooling_rate
        state_transitions.append(current_state.state)  # Record state transitions

    end_time = time.time()
    runtime = end_time - start_time

    return current_state.state, state_transitions, runtime

# Function to move the blank tile
def move_blank_tile(state):
    blank_position = state.blank_position
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random_move = random.choice(moves)

    next_blank_pos = tuple(np.add(blank_position, random_move))
    if 0 <= next_blank_pos[0] < state.size and 0 <= next_blank_pos[1] < state.size:
        state.swap(blank_position, next_blank_pos)

# Function to display puzzle state
def display_puzzle(state):
    for row in state:
        print(" ".join(map(str, row)))

# Function to visualize the pathfinding process
def visualize_path(state_transitions):
    n_steps = len(state_transitions)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the initial state
    ax.imshow(state_transitions[0], cmap='viridis', aspect='equal')
    ax.axis('off')  # Remove axis

    # Plot lines connecting states
    for i in range(n_steps - 1):
        current_state = state_transitions[i]
        next_state = state_transitions[i + 1]

        for r in range(len(current_state)):
            for c in range(len(current_state[0])):
                if current_state[r][c] != next_state[r][c]:
                    ax.plot([c, c], [r, r], color='black', linewidth=2)

    plt.title('Paths to Goal State')
    plt.show()

# Ask user for N
N = int(input("Enter the value of N (default = 8): ") or "8")
if not (8 <= N < 100):
    print("N should be between 8 and 99.")
    exit()

# Ask user for start case input
print(f"Enter the start case input (e.g., for {N}x{N} puzzle):")
initial_state = [list(map(int, input().split())) for _ in range(N)]

# Run simulated annealing
final_state, state_transitions, runtime = simulated_annealing(initial_state)

# Display final solution
print("Final solution:")
display_puzzle(final_state)

# Number of actions needed
print("Number of actions needed:", cost(final_state))

# Runtime to find the final solution
print("Runtime to find the final solution:", runtime, "seconds")

# Visualize the pathfinding process
visualize_path(state_transitions)
