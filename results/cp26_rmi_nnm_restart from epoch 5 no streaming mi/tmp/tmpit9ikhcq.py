import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    r_initial = 0.11  # Increased initial radius for better utilization
    vertical_spacing = np.sqrt(3) * r_initial
    rows = 5
    num_circles_per_row = [5, 5, 5, 5, 6]  # Alternate row counts to fit 26 circles
    initial_vars = []
    
    for i in range(rows):
        if i % 2 == 0:
            # Even rows: start at r_initial, spaced by 2r_initial
            num_circles = num_circles_per_row[i]
            x_start = r_initial
            x_step = 2 * r_initial
            x_positions = [x_start + j * x_step for j in range(num_circles)]
        else:
            # Odd rows: shifted by r_initial/2, spaced by 2r_initial
            num_circles = num_circles_per_row[i]
            x_start = r_initial + r_initial / 2
            x_step = 2 * r_initial
            x_positions = [x_start + j * x_step for j in range(num_circles)]
        y_position = r_initial + i * vertical_spacing
        for x in x_positions:
            initial_vars.append(x)
            initial_vars.append(y_position)
            initial_vars.append(r_initial)
    
    # Objective function: maximize sum of radii
    def objective(vars):
        radii = [vars[3*i + 2] for i in range(n_circles)]
        return -sum(radii)
    
    # Constraints
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n_circles):
        def constraint_x_lower(vars, i=i):
            idx = 3*i
            r_idx = 3*i + 2
            return vars[idx] - vars[r_idx]
        constraints.append({'type': 'ineq', 'fun': constraint_x_lower})
        
        def constraint_x_upper(vars, i=i):
            idx = 3*i
            r_idx = 3*i + 2
            return 1 - vars[idx] - vars[r_idx]
        constraints.append({'type': 'ineq', 'fun': constraint_x_upper})
        
        def constraint_y_lower(vars, i=i):
            idx = 3*i + 1
            r_idx = 3*i + 2
            return vars[idx] - vars[r_idx]
        constraints.append({'type': 'ineq', 'fun': constraint_y_lower})
        
        def constraint_y_upper(vars, i=i):
            idx = 3*i + 1
            r_idx = 3*i + 2
            return 1 - vars[idx] - vars[r_idx]
        constraints.append({'type': 'ineq', 'fun': constraint_y_upper})
    
    # Distance constraints between all pairs
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            def constraint_pair(vars, i=i, j=j):
                idx_i_x, idx_i_y, idx_i_r = 3*i, 3*i + 1, 3*i + 2
                idx_j_x, idx_j_y, idx_j_r = 3*j, 3*j + 1, 3*j + 2
                x_i, y_i, r_i = vars[idx_i_x], vars[idx_i_y], vars[idx_i_r]
                x_j, y_j, r_j = vars[idx_j_x], vars[idx_j_y], vars[idx_j_r]
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                return dist_sq - (r_i + r_j)**2
            constraints.append({'type': 'ineq', 'fun': constraint_pair})
    
    # Run optimization with improved parameters
    result = minimize(
        objective, 
        initial_vars, 
        constraints=constraints, 
        method='trust-constr', 
        tol=1e-9, 
        options={'maxiter': 5000, 'verbose': 2}
    )
    
    # Extract results
    centers = []
    radii = []
    for i in range(n_circles):
        x = result.x[3*i]
        y = result.x[3*i + 1]
        r = result.x[3*i + 2]
        centers.append([x, y])
        radii.append(r)
    
    sum_radii = sum(radii)
    return (np.array(centers), np.array(radii), sum_radii)