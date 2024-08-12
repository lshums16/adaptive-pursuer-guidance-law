import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def acceleration_dynamics(t, y):
    # Example of a sinusoidal acceleration command
    acceleration_command = np.sin(t)
    
    # System parameters
    zeta = 0.7      # Damping ratio
    omega_n = 2      # Natural frequency

    # Define the ODEs representing the second-order system
    dydt = [y[1], -2 * zeta * omega_n * y[1] - omega_n**2 * y[0] + acceleration_command]

    return dydt

# Initial conditions
initial_conditions = [0, 0]

# Time span
total_time = 10  # Total simulation time
t_span = (0, total_time)

# Solve the ODE using solve_ivp
sol = solve_ivp(acceleration_dynamics, t_span, initial_conditions, t_eval=np.linspace(0, total_time, 100))

# Plot the results
plt.figure()
plt.plot(sol.t, np.sin(sol.t), label = "Unfiltered Acceleration")
plt.plot(sol.t, sol.y[0], label='Filtered Acceleration')
plt.xlabel('Time')
plt.ylabel('Filtered Acceleration')
plt.title('Second-order Filtered Acceleration Command')
plt.legend()
plt.grid(True)
plt.show()
