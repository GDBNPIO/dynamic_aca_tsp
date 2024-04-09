import numpy as np
from scipy import spatial
import pandas as pd
from matplotlib import animation
import pandas as pd
import os
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt


output_path = '/path/to/output/chart/'


num_points = 50

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def update_points_and_distance_matrix(aca_obj, new_points_coordinate, changed_indices):
    # Update the coordinates of points
    aca_obj.points_coordinate = new_points_coordinate
    # Recalculate the distance matrix
    aca_obj.distance_matrix = spatial.distance.cdist(new_points_coordinate, new_points_coordinate, metric='euclidean')
    # Recalculate the probability matrix
    aca_obj.prob_matrix_distance = 1 / (aca_obj.distance_matrix + 1e-10 * np.eye(len(new_points_coordinate), len(new_points_coordinate)))
    # Update the number of cities
    aca_obj.n_dim = len(new_points_coordinate)
    
    # Reset only the pheromone of changed points
    for index in changed_indices:
        # Assuming all pheromones have an initial value of 1, adjust as needed
        aca_obj.Tau[index, :] = 1  # Reset the corresponding row of the changed point
        aca_obj.Tau[:, index] = 1  # Reset the corresponding column of the changed point

    # Note: No need to reset the ant's path table, omitted here



def cal_total_distance(routine, distance_matrix):
    return sum([distance_matrix[routine[i % len(routine)], routine[(i + 1) % len(routine)]] for i in range(len(routine))])



class ACA_TSP:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 points_coordinate=None,
                 new_num_points=5,
                 ):
        self.func = func
        self.n_dim = n_dim  # Number of cities
        self.size_pop = size_pop  # Number of ants
        self.max_iter = max_iter  # Number of iterations
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta  # Importance of fitness
        self.rho = rho  # Pheromone evaporation rate
        self.points_coordinate = points_coordinate  # Define the coordinates of points
        self.distance_matrix = distance_matrix
        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # Avoid division by zero error
        self.Tau = np.ones((n_dim, n_dim))  # Pheromone matrix, updated every iteration
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int64)  # Crawling path of each ant in a generation
        self.y = None  # Total distance traveled by each ant in a generation
        self.generation_best_X, self.generation_best_Y = [], []  # Record the best solution of each generation
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  # For historical reasons, to maintain consistency
        self.best_x, self.best_y = None, None

    def run(self, new_num_points, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        print('self.max_iter:', self.max_iter)

        for i in range(self.max_iter):  # For each iteration
            # Modify the point set after the 50th iteration
            if i == 50 or i == 150:
                new_points = np.random.rand(new_num_points, 2)  # Generate coordinates for 5 new points
                updated_points_coordinate = np.vstack((self.points_coordinate[:-new_num_points, :], new_points))  # Replace old points
                changed_indices = list(range(self.n_dim-new_num_points, self.n_dim))  # Calculate the indices of changed points
                update_points_and_distance_matrix(self, updated_points_coordinate, changed_indices)  # Update point set and distance matrix, pass in the indices of changed points


            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance ** self.beta)
            for j in range(self.size_pop):  # For each ant
                self.Table[j, 0] = 0  # start point, can be random, but it doesn't make a difference
                for k in range(self.n_dim - 1):  # Each node the ant reaches
                    taboo_set = set(self.Table[j, :k + 1])  # Points already visited and the current point, cannot be visited again
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # Choose from these points
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # Normalize the probabilities
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # Calculate the distance
            y = np.array([self.func(self.Table[i], self.distance_matrix) for i in range(self.size_pop)])

            # Record the best solution in the history
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # Calculate the pheromones to be updated
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # For each ant
                for k in range(self.n_dim - 1):  # For each node
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # The ant crawls from node n1 to node n2
                    delta_tau[n1, n2] += 1 / y[j]  # Update the pheromones
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # The ant crawls back from the last node to the first node
                delta_tau[n1, n2] += 1 / y[j]  # Update the pheromones

            # Pheromone evaporation + pheromone update
            self.Tau = (1 - self.rho) * self.Tau + delta_tau
        
        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run

def dynamic_chart(aca,max_iter,new_num_points):
    # plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(1, figsize=(10, 8))
    for i in range(max_iter):
        # Simulate a solution of the ant algorithm with a random sequence here
        routine = aca.x_best_history[i]
        
        ax.cla()  # Clear the current axis
        # ax[1].cla()  # Clear the history of the best values plot
        # Plot the path
        best_points_ = np.concatenate([routine, [routine[0]]])
        best_points_coordinate = aca.points_coordinate[best_points_, :]
        ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
        ax.set_title(f'Iteration: {i + 1}')
     
        plt.pause(0.1)  # Pause for a while to update the chart

    # pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(aca.y_best_history, label="Distance")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.title("Distance vs. Iteration")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path + 'ACA_TSP_without_reset_rho_' + f'{new_num_points}' + '.png')  # Save before showing
    print("Chart saved in", output_path)
    # plt.ioff()  # Disable interactive mode
# def dynamic_chart(aca, max_iter, new_num_points):
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Function to update each frame of the animation
#     def update(i):
#         ax.clear()  # Clear previous drawing
#         routine = aca.x_best_history[i]
#         best_points_ = np.concatenate([routine, [routine[0]]])  # Complete the loop
#         best_points_coordinate = aca.points_coordinate[best_points_, :]
#         ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
#         ax.set_title(f'Iteration: {i + 1}')
#         # Set limits if your points go out of the initial figure size
#         ax.set_xlim([np.min(aca.points_coordinate[:, 0]), np.max(aca.points_coordinate[:, 0])])
#         ax.set_ylim([np.min(aca.points_coordinate[:, 1]), np.max(aca.points_coordinate[:, 1])])

#     # Create animation
#     anim = FuncAnimation(fig, update, frames=max_iter, repeat=False)

#     # Save the animation
#     anim.save(output_path + f'ACA_TSP_without_reset_rho_{new_num_points}.gif', writer='pillow', fps=10)
#     print(f"Animation saved as ACA_TSP_without_reset_rho_{new_num_points}.gif in {output_path}")

if __name__ == "__main__":
    max_iter = 300
    for New_Num_Points in [5,10]:
        aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                    size_pop=50, max_iter=max_iter,
                    distance_matrix=distance_matrix,points_coordinate=points_coordinate,
                    new_num_points=New_Num_Points)

        best_x, best_y = aca.run(New_Num_Points)
        dynamic_chart(aca,max_iter,New_Num_Points)
        # Save points_coordinate to a CSV file
        df = pd.DataFrame(aca.points_coordinate, columns=['x', 'y'])
        df.to_csv(output_path+'points_coordinate.csv', index=False)
