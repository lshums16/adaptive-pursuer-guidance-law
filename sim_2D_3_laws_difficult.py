import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from derivatives import Derivatives
import plotter as p
import matplotlib.animation as animation



        
t_start = 0
t_end = 250
timesteps = 300

target_accel_type = "weave"
target_accel = -25
target_accel_amp = -10

collision_buffer_range = 0.5

# d_adaptive = Derivatives(filter_cmd = True, target_accel_type = target_accel_type, target_accel = target_accel, target_accel_amp = target_accel_amp, accel_sat = 5.)
d_png = Derivatives(filter_cmd = True, target_accel_type = 'weave', target_accel = -25, target_accel_amp = -10, accel_sat = np.inf)
d_aftng = Derivatives(filter_cmd = True, target_accel_type = 'weave', target_accel = -25, target_accel_amp = -10, accel_sat = np.inf)
d_ftcg = Derivatives(filter_cmd = True, target_accel_type = 'weave', target_accel = -25, target_accel_amp = -10, accel_sat = np.inf)

seeker0_PNG   = np.array([0., 0., 800., np.deg2rad(5)])
seeker0_FTCG  = np.array([0., 0., 800., np.deg2rad(5)])
seeker0_AFTNG = np.array([0., 0., 800., np.deg2rad(5), 0., 0.])
seeker0_S_AFTNG = np.array([0., 0., 800., np.deg2rad(5), 0.])
target0 = np.array([40000., 0., 700., np.deg2rad(150)])
filter0 = np.array([0., 0.])

# x0 = np.concatenate((seeker0_PNG, seeker0_FTCG, seeker0_S_AFTNG, target0, filter0))
# x0_adaptive = np.concatenate((seeker0_S_AFTNG, target0, filter0))
x0_png = np.concatenate((seeker0_PNG, target0, filter0))
x0_ftcg = np.concatenate((seeker0_FTCG, target0, filter0))
x0_aftng = np.concatenate((seeker0_AFTNG, target0, filter0))


# S_AFTNG_solution = solve_ivp(d_adaptive.S_AFTNG_derivatives, [t_start, t_end], x0_adaptive, method='RK45', t_eval = np.linspace(t_start, t_end, timesteps))

PNG_solution = solve_ivp(d_png.pronav_derivatives, [t_start, t_end], x0_png, method='RK45', t_eval = np.linspace(t_start, t_end, timesteps))
AFTNG_solution = solve_ivp(d_aftng.AFTNG_derivatives, [t_start, t_end], x0_aftng, method='RK45', t_eval = np.linspace(t_start, t_end, timesteps))
FTCG_solution = solve_ivp(d_ftcg.FTCG_derivatives, [t_start, t_end], x0_ftcg, method='RK45', t_eval = np.linspace(t_start, t_end, timesteps))

### PLOTTING ###

png_collision_idx, png_miss_dist = p.get_collision_info(PNG_solution.t, PNG_solution.y[:4, :], PNG_solution.y[4:8], d_png, 2)
# aftng_collision_idx, aftng_miss_dist = p.get_collision_info(AFTNG_solution.t, AFTNG_solution.y[:4, :], AFTNG_solution.y[6:10], d_aftng, 2)
ftcg_collision_idx, ftcg_miss_dist = p.get_collision_info(FTCG_solution.t, FTCG_solution.y[:4, :], FTCG_solution.y[4:8], d_ftcg, 2)

png_collision_idx, png_miss_dist = p.plot_PNG_data(PNG_solution, d_png, collision_buffer_range, target_accel_type, target_accel, target_accel_amp)
# aftng_collision_idx, aftng_miss_dist = p.plot_S_AFTNG_data(S_AFTNG_solution, d_adaptive, collision_buffer_range, target_accel_type, target_accel, target_accel_amp)


seekers = {#"S-AFTNG": {"t_array": S_AFTNG_solution.t[:(collision_idx + 1)],
                    #    "hist": S_AFTNG_solution.y[:4, :(collision_idx + 1)],
                    #    "LOS_lines": False,
                    #    "LOS_period": 4}
           "PNG": {"t_array": PNG_solution.t[:png_collision_idx + 1],
                   "target_hist": PNG_solution.y[4:8, :png_collision_idx + 1],
                   "hist": PNG_solution.y[:4, :png_collision_idx + 1],
                   "LOS_lines": False,
                   "line_color": "red",
                   "LOS_period": 8
           },
           "AFTNG": {
                "t_array": AFTNG_solution.t[:png_collision_idx + 1],
                "hist": AFTNG_solution.y[:4, :png_collision_idx + 1],
                "LOS_lines": False,
                "line_color": "orange",
                "LOS_period": 8
              },
           "FTCG": {
                "t_array": FTCG_solution.t[:200],
                "hist": FTCG_solution.y[:4, :200],
                "LOS_lines": False,
                "line_color": "purple",
                "LOS_period": 8
           }
           
           }

# p.plot_trajectories(seekers, PNG_solution.y[4:8])


# p.animate_trajectories(S_AFTNG_solution.y[:4, :(collision_idx + 1)], S_AFTNG_solution.y[5:9, :(collision_idx + 1)], S_AFTNG_solution.t[:(collision_idx + 1)])
# p.animate_trajectories(PNG_solution.y[:4, :(collision_idx + 1)], PNG_solution.y[4:8, :(collision_idx + 1)], PNG_solution.t[:(collision_idx + 1)])
# input("Press enter to close")

p.animate_trajectories(seekers)





