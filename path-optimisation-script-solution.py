import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import heapq

# Reading CSV file into a DataFrame
en_cost_df = pd.read_csv('energy_cost.csv')

# Reading CSV file into a DataFrame
al_map_df = pd.read_csv('altitude_map.csv', header=None)

# Split data
X = en_cost_df[['gradient']]
y = en_cost_df['energy_cost']

# Testing different polynomial degrees and selecting the best-fit model based on R-squared score
degrees = range(1, 4)
best_r2 = -np.inf
best_degree = None
best_model = None

for degree in degrees:
    # Apply PolynomialFeatures to transform X to higher-degree features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Create and fit the Polynomial Regression model
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Make predictions using the model
    y_pred = poly_model.predict(X_poly)

    # Calculate the R-squared score
    r2 = r2_score(y, y_pred)

    # Keeping the model with higher R-squared score
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        best_model = poly_model

# Best-fit model predictions
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_poly_pred = PolynomialFeatures(degree=best_degree).fit_transform(X_pred)
y_pred = best_model.predict(X_poly_pred)

# Lets adjust the gradient values for each point in the map.
def calculate_gradient(altitude_map, point):
    x, y = point
    if x == 0 or x == altitude_map.shape[0] - 1 or y == 0 or y == altitude_map.shape[1] - 1:
        return 0.0  # Border points have a gradient of 0
    else:
        dx = altitude_map[x + 1, y] - altitude_map[x - 1, y]
        dy = altitude_map[x, y + 1] - altitude_map[x, y - 1]
        return np.arctan2(dy, dx)
    
# Since we are looking to minimize the energy expenditure, lets use the straight-line distance as the heuristic
def heuristic_cost_estimate(point, goal):
    # Calculate the straight-line distance from 'point' to the 'goal'
    return np.sqrt((goal[0] - point[0]) ** 2 + (goal[1] - point[1]) ** 2)

# Determining its neighboring points
def get_neighbors(point, altitude_map):
    # Get adjacent points from the altitude map
    neighbors = []
    x, y = point

    if x > 0:
        neighbors.append((x - 1, y))
    if x < altitude_map.shape[0] - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y <  altitude_map.shape[1] - 1:
        neighbors.append((x, y + 1))

    return neighbors 

# Implementing the A* search algorithm
def a_star_search(altitude_map, start, goal, model):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_cost_estimate(start, goal)}

    # Keep track of points in the open set using a hash table
    open_set_hash = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, altitude_map):
            neighbor_gradient = calculate_gradient(altitude_map, neighbor)
            neighbor_altitude = altitude_map[neighbor]

            # Generate polynomial features for the neighbor
            polynomial_features = PolynomialFeatures(degree=3, include_bias=False)
            neighbor_features = np.array([neighbor_altitude]).reshape(1, 1) # Reshape to 2D array
            neighbor_features_poly = polynomial_features.fit_transform(neighbor_features)

            # Concatenate gradient and polynomial features
            neighbor_features_poly = np.concatenate([neighbor_features_poly, [[neighbor_gradient]]], axis=1)

            # Calculate the tentative g_score for the neighbor
            tentative_g_score = g_score[current] + model.predict(neighbor_features_poly)[0]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_cost_estimate(neighbor, goal)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return None  # No path found

# Now, lets find the optimal path from any point in the south border to the given goal, which is (200, 559)
start = (558, 0)
goal = (200, 559)
optimal_path = a_star_search(al_map_df.values, start, goal, best_model)

def write_path_solution(optimal_path, output_csv):
    # Convert the optimal_path to a DataFrame with columns "x_coord" and "y_coord"
    path_df = pd.DataFrame(optimal_path, columns=["x_coord", "y_coord"])
    # Save the DataFrame to a .csv file
    path_df.to_csv(output_csv, index=False)

def solution_map(altitude_map, optimal_path, output_png):
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the altitude map
    im = ax.imshow(altitude_map, cmap="terrain", origin="lower", extent=[0, altitude_map.shape[1], 0, altitude_map.shape[0]])
    plt.colorbar(im, ax=ax, label="Altitude (m)")
    
    # Extract x and y coordinates from the optimal path
    x_coords = [point[0] for point in optimal_path]
    y_coords = [point[1] for point in optimal_path]
    
    # Plot the optimal path over the altitude map
    ax.plot(x_coords, y_coords, color="red", linewidth=1, marker="o", label="Optimal Path")
    
    # Plot the start point with a circle marker
    ax.scatter(start[0], start[1], color="green", marker="o", s=300, label="Start")
    
    # Plot the goal point with a diamond marker
    ax.scatter(goal[0], goal[1], color="blue", marker="D", s=300, label="Goal")
    
    
    # Set axis labels and title
    ax.set_xlabel("x_coord")
    ax.set_ylabel("y_coord")
    ax.set_title("Optimal Path Overlaid on the Altitude Map")
    ax.legend()
    
    # Save the visualization as a .png file
    plt.savefig(output_png)
    plt.close()

# Save objects into the desired files
output_csv = "optimal_path_coords.csv"
output_png = "optimal_path_visualization.png"

write_path_solution(optimal_path, output_csv)
solution_map(al_map_df.values, optimal_path, output_png)

def write_advice_to_txt(advice, output_txt):
    with open(output_txt, 'w') as file:
        file.write(advice)

advice_sentence = "To optimize the trail construction for Protea Treks, consider gathering participant-specific data like body mass, walking speed, and any medical conditions. Incorporate real-time weather conditions, terrain information such as surface roughness or obstacles, and employ advanced machine learning techniques for accurate energy expenditure predictions. Utilize user feedback and trail usage data for dynamic adjustments, tailoring the path to user preferences or skill level, and improving the overall trekking experience. This data-driven approach ensures an energy-efficient beginner's trail that offers an enhanced adventure."

output_txt = "optimal_path_advice.txt"

write_advice_to_txt(advice_sentence, output_txt)