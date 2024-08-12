import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def inHalfSpace(pos, halfspace_r, halfspace_n):
    if (pos - halfspace_r).T @ halfspace_n >= 0:
        return True
    else:
        return False

def check_agent_collision(Vr, R, t_array, i, seeker_pos, seeker_pos_prev, target_pos, target_pos_prev, collision_buffer_range = 0.5):
    ts = t_array[i] - t_array[i - 1]
    check_buffer = Vr*ts # this is the minimum distance the simulation at this time step is able to detect
    
    if R <= check_buffer: # if the distance between the two agents is less than the buffer (plus 50% cushion)
        new_ts = int(Vr/collision_buffer_range) + 1
        agent1_pos_list = np.linspace(seeker_pos_prev, seeker_pos, new_ts)
        agent2_pos_list = np.linspace(target_pos_prev, target_pos, new_ts)
        
        # import matplotlib.pyplot as plt 
        
        # # plot 3D trajectory (linear approximation) of both agents between last timestep and this timestep
        # plt.figure()
        # # xRange = np.max([np.max(position[i][:, 1]) for i in range(num_missiles)]) - np.min([np.min(position[i][:, 1]) for i in range(num_missiles)])
        # # yRange = np.max([np.max(position[i][:, 0]) for i in range(num_missiles)]) - np.min([np.min(position[i][:, 0]) for i in range(num_missiles)])
        # # zRange = np.max([np.max(position[i][:, 2]) for i in range(num_missiles)]) - np.min([np.min(position[i][:, 2]) for i in range(num_missiles)])
        # # ax.set_box_aspect([xRange, yRange, zRange])
        # plt.plot(agent1_pos_list[:, 0], agent1_pos_list[:, 1], label = "Agent 1")
        # plt.plot(agent2_pos_list[:, 0], agent2_pos_list[:, 1], label = "Agent 2", color = 'r')
        # # plt.scatter(other_agent.state.pos[1, 0], other_agent.state.pos[0, 0], -other_agent.state.pos[2, 0], color = 'r')
        # # plt.xlabel("East (m)")
        # # plt.set_ylabel("North (m)")
        # # plt.set_zlabel("Altitude (m)")
        # plt.show(block = False)
        
        distance = [None]*new_ts # only needed for plotting
        for i in range(new_ts):
            distance[i] = np.linalg.norm(agent1_pos_list[i] - agent2_pos_list[i]) # only needed for plotting
        
        # plot distance between agents between last timestep and this timestep (using the linear approximation of each trajectory)
        # plt.figure()
        # plt.plot(distance)
        # plt.show(block = False)
        if np.min(distance) < collision_buffer_range:
            return True, np.min(distance)
        else:
            return False, np.min(distance)
    else:
        return False, np.linalg.norm(target_pos - seeker_pos)

def plot_trajectories(seekers, target_hist):
    plt.figure()
    for key in seekers:
        seeker_hist = seekers[key]["hist"]
        plt.plot(seeker_hist[0], seeker_hist[1], label = key)
        if seekers[key]["LOS_lines"] == True:
            LOS_period = seekers[key]["LOS_period"]
            LOS_line_itr = 0
            numsteps = seeker_hist.shape[1]
            for i in range(numsteps):
                t = seekers[key]["t_array"][i]
                if np.isclose(t, LOS_line_itr*LOS_period) or t > LOS_line_itr*LOS_period or i == numsteps - 1:
                    plt.scatter(seeker_hist[0, i], seeker_hist[1, i], color = 'r')
                    plt.scatter(target_hist[0, i], target_hist[1, i], color = 'r')
                    plt.plot([seeker_hist[0, i], target_hist[0, i]], [seeker_hist[1, i], target_hist[1, i]], color = 'r')
                    LOS_line_itr += 1
    
    plt.plot(target_hist[0], target_hist[1], label = "Target")
    plt.legend()
    plt.gca().set_aspect('equal')
    
    plt.show(block = False)
            
# Plotting
def plot_PNG_data(solution, d, collision_buffer_range, target_accel_type, target_accel, target_accel_amp = 0.):
    ######################## PLOT SATURATION #######################################################################
    PNG_hist = solution.y[:4, :]
    target_hist = solution.y[4:8, :]
    filter_hist = solution.y[8:, :]

    # extract data for plots
    qdot = []
    Vq_hist = []
    accel = []
    target_accel_n_hist = []
    R_hist = []
    min_dist = (len(solution.t) - 1, np.inf)
    for i in range(len(solution.t)):
        q, Vr, Vq, R, seeker_crs_angle  = d.guidance.get_coeffs(PNG_hist[:, i], target_hist[:, i])
        R_hist.append(R)
        qdot.append(Vq/R)
        Vq_hist.append(Vq)
        accel.append(d.saturate(d.guidance.ProNav(PNG_hist[:4, i], target_hist[:, i])))
        
        # keep track of the miss distance
        hasCollided, min_dist_temp = check_agent_collision(Vr, R, solution.t, i, PNG_hist[:2, i], PNG_hist[:2, i-1], target_hist[:2, i], target_hist[:2, i - 1], collision_buffer_range)
        if hasCollided: 
            min_dist = (i, min_dist_temp)    
        elif min_dist_temp < min_dist[1]:
            min_dist = (i, min_dist_temp)
          
        # calc acceleration data      
        LOS = np.array([target_hist[:2, i] - PNG_hist[:2, i]]).T
        target_V = target_hist[2, i]
        target_crs_angle = target_hist[3, i]
        target_V_vec = np.array([[np.cos(target_crs_angle), np.sin(target_crs_angle)]]).T
        R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])
        
        if target_accel_type == "constant":
            target_accel = target_accel
        elif target_accel_type == "weave":
            target_accel = target_accel + target_accel_amp*np.sin(solution.t[i])
        else:
            raise ValueError("target_accel_type must be \"constant\" or \"weave\"")
        target_accel_vec = target_accel*R@target_V_vec
        accel_n  = target_accel_vec - (LOS.T@target_accel_vec)/(LOS.T@LOS)*LOS
        if inHalfSpace(accel_n, LOS, target_V_vec):
            A_Tq = np.linalg.norm(accel_n)
        else:
            A_Tq = -np.linalg.norm(accel_n)
        target_accel_n_hist.append(A_Tq)

        if hasCollided:
            break
        
    collision_idx = min_dist[0]
    fig, axs = plt.subplots(2, 1, sharex = True)
    ### LOS RATES ###
    axs[1].plot(solution.t, qdot, label = "PNG")
    axs[1].set_title("LOS Rate Profiles")
    axs[1].set_ylabel("LOS Rate (rad/s)")
    axs[1].set_xlabel("Time (s)") 
    axs[1].set_ylim([-0.02, 0.02])

    ### SEEKER ACCELERATION ###
    axs[0].plot(accel, label = "PNG command")
    if not filter_hist is None and len(filter_hist) != 0:
        axs[0].plot(d.saturate(filter_hist[1]), label = "PNG acceleration")
    axs[0].set_ylim([-60, 60])
    axs[0].set_title("Missile Acceleration Profiles")
    axs[0].set_ylabel("Missile Acceleration (m/s^2)")
    axs[0].set_xlabel("Time (s)")
    axs[0].legend()

    # ### TARGET ACCELERATION ###
    # axs[1, 0].plot(solution.t, PNG_hist[4], label = "Adaptive Term")
    # axs[1, 0].plot(solution.t, target_accel_n_hist, label = "Target Acceleration")
    # axs[1, 0].set_title("Adaptive Term")
    # axs[1, 0].set_ylim([-50, 50])
    # axs[1, 0].set_ylabel("Adaptive Term (m/s^2)")
    # axs[1, 0].set_xlabel("Time (s)")
    # axs[1, 0].legend()

    # ### LYAPUNOV CANDIDATE ###
    # axs[1, 1].plot(solution.t, lyapunov_hist)
    # axs[1, 1].set_ylim([0, np.max(lyapunov_hist)*1.2])
    # axs[1, 1].set_title("Lyapunov Candidate")
    # axs[1, 1].set_ylabel("Lyapunov Candidate")
    # axs[1, 1].set_xlabel("Time (s)")

    ### MISS DISTANCE ###
    plt.figure()
    plt.plot(solution.t, R_hist)
    plt.title("Miss Distance")
    plt.ylabel("Miss Distance (m)")
    plt.xlabel("Time (s)")

    print("Miss distance (PNG):", min_dist[1])

    plt.show(block = False)
    
    return min_dist

def plot_S_AFTNG_data(solution, d, collision_buffer_range, target_accel_type, target_accel, target_accel_amp = 0.):
    ######################## PLOT SATURATION #######################################################################
    S_AFTNG_hist = solution.y[:5, :]
    target_hist = solution.y[5:9, :]
    filter_hist = solution.y[9:, :]

    # extract data for plots
    S_AFTNG_qdot = []
    S_AFTNG_Vq = []
    S_AFTNG_accel = []
    target_accel_n_hist = []
    lyapunov_hist = []
    R_hist = []
    min_dist = (len(solution.t) - 1, np.inf)
    for i in range(len(solution.t)):
        q, Vr, Vq, R, seeker_crs_angle  = d.guidance.get_coeffs(S_AFTNG_hist[:, i], target_hist[:, i])
        R_hist.append(R)
        S_AFTNG_qdot.append(Vq/R)
        S_AFTNG_Vq.append(Vq)
        S_AFTNG_accel.append(d.saturate(d.guidance.S_AFTNG(S_AFTNG_hist[:4, i], S_AFTNG_hist[4, i], target_hist[:, i])[0]))
        
        # keep track of the miss distance
        hasCollided, min_dist_temp = check_agent_collision(Vr, R, solution.t, i, S_AFTNG_hist[:2, i], S_AFTNG_hist[:2, i-1], target_hist[:2, i], target_hist[:2, i - 1], collision_buffer_range)
        if hasCollided: 
            min_dist = (i, min_dist_temp)    
        elif min_dist_temp < min_dist[1]:
            min_dist = (len(solution.t) - 1, min_dist_temp)
          
        # calc acceleration data      
        LOS = np.array([target_hist[:2, i] - S_AFTNG_hist[:2, i]]).T
        target_V = target_hist[2, i]
        target_crs_angle = target_hist[3, i]
        target_V_vec = np.array([[np.cos(target_crs_angle), np.sin(target_crs_angle)]]).T
        R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])
        
        if target_accel_type == "constant":
            target_accel = target_accel
        elif target_accel_type == "weave":
            target_accel = target_accel + target_accel_amp*np.sin(solution.t[i])
        else:
            raise ValueError("target_accel_type must be \"constant\" or \"weave\"")
        target_accel_vec = target_accel*R@target_V_vec
        accel_n  = target_accel_vec - (LOS.T@target_accel_vec)/(LOS.T@LOS)*LOS
        if inHalfSpace(accel_n, LOS, target_V_vec):
            A_Tq = np.linalg.norm(accel_n)
        else:
            A_Tq = -np.linalg.norm(accel_n)
        target_accel_n_hist.append(A_Tq)

        # extract Lyapunov values
        k1 = 1
        lyapunov_hist.append(k1*abs(Vq) + (A_Tq - S_AFTNG_hist[4, i])**2/2)
        
        if hasCollided:
            break
        
    collision_idx = min_dist[0]
    fig, axs = plt.subplots(2, 2, sharex = True)
    ### LOS RATES ###
    axs[0, 1].plot(solution.t[:collision_idx + 1], S_AFTNG_qdot, label = "S-AFTNG")
    axs[0, 1].set_title("LOS Rate Profiles")
    axs[0, 1].set_ylabel("LOS Rate (rad/s)")
    axs[0, 1].set_xlabel("Time (s)") 
    axs[0, 1].set_ylim([-0.02, 0.02])

    ### SEEKER ACCELERATION ###
    axs[0, 0].plot(solution.t[:collision_idx + 1], S_AFTNG_accel, label = "S-AFTNG command")
    axs[0, 0].plot(solution.t[:collision_idx + 1], d.saturate(filter_hist[1, :(collision_idx + 1)]), label = "S-AFTNG acceleration")
    axs[0, 0].set_ylim([-60, 60])
    axs[0, 0].set_title("Missile Acceleration Profiles")
    axs[0, 0].set_ylabel("Missile Acceleration (m/s^2)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].legend()

    ### TARGET ACCELERATION AND ADAPTIVE TERM ###
    axs[1, 0].plot(solution.t[:collision_idx + 1], S_AFTNG_hist[4, :(collision_idx + 1)], label = "Adaptive Term")
    axs[1, 0].plot(solution.t[:collision_idx + 1], target_accel_n_hist, label = "Target Acceleration")
    axs[1, 0].set_title("Adaptive Term")
    axs[1, 0].set_ylim([-50, 50])
    axs[1, 0].set_ylabel("Adaptive Term (m/s^2)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].legend()

    ### LYAPUNOV CANDIDATE ###
    axs[1, 1].plot(solution.t[:collision_idx + 1], lyapunov_hist)
    axs[1, 1].set_ylim([0, np.max(lyapunov_hist)*1.2])
    axs[1, 1].set_title("Lyapunov Candidate")
    axs[1, 1].set_ylabel("Lyapunov Candidate")
    axs[1, 1].set_xlabel("Time (s)")

    ### MISS DISTANCE ###
    plt.figure()
    plt.plot(solution.t[:(collision_idx + 1)], R_hist)
    plt.title("Miss Distance")
    plt.ylabel("Miss Distance (m)")
    plt.xlabel("Time (s)")

    print("Miss distance (S-AFTNG):", min_dist[1])

    plt.show(block = False)
    
    return min_dist

# def animate_trajectories(seeker, target, time):
#     ### Animation ###
#     fig, ax = plt.subplots()
#     buffer= 10000
#     xmax = np.max([np.max(seeker[0]), np.max(target[0])]) + buffer
#     xmin = np.min([np.min(seeker[0]), np.min(target[0])]) - buffer
#     ymax = np.max([np.max(seeker[1]), np.max(target[1])]) + buffer
#     ymin = np.min([np.min(seeker[1]), np.min(target[1])]) - buffer
#     def animate(i):
#         ax.clear()
#         ax.text(xmax - 12000, ymax + 1000, f"t = {time[i]:.2f}s")
        
#         # PNG 
#         plt.plot(seeker[0, :i], seeker[1, :i])
#         plt.scatter(seeker[0, i], seeker[1, i], color = 'blue')
        
#         # target
#         plt.plot(target[0, :i], target[1, :i])
#         plt.scatter(target[0, i], target[1, i], color = 'orange')
        
#         # LOS
#         plt.plot([seeker[0, i], target[0, i]], [seeker[1, i], target[1, i]], color = 'red')
        
        
#         plt.xlim((xmin, xmax))
#         plt.ylim((ymin, ymax))
#         ax.set_aspect('equal')

#     dt = time[1] - time[0]
#     anim = animation.FuncAnimation(fig, animate, frames = seeker.shape[1], interval = dt*100)

#     plt.show()
    
def animate_trajectories(seekers):
    fig, ax = plt.subplots()
    
    def animate(i):
        ax.clear()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Missile Trajectories")
        
        for label, seeker in seekers.items():
            seeker_hist = seeker["hist"]
            if i < seeker_hist.shape[1]:
                seeker_traj_x = seeker_hist[0, :(i + 1)]
                seeker_traj_y = seeker_hist[1, :(i + 1)]
                ax.scatter(seeker_traj_x[i], seeker_traj_y[i], color = seeker["line_color"])
            else:
                seeker_traj_x = seeker_hist[0]
                seeker_traj_y = seeker_hist[1]
                ax.scatter(seeker_traj_x[len(seeker_traj_x) - 1], seeker_traj_y[len(seeker_traj_y) - 1], color = seeker["line_color"])
            ax.plot(seeker_traj_x, seeker_traj_y, label = label, color = seeker["line_color"])
        
        target_traj_x = seekers['PNG']['target_hist'][0, :(i + 1)]
        target_traj_y = seekers['PNG']['target_hist'][1, :(i + 1)]
        
        ax.set_xlim(-5000, 80000)
        ax.set_ylim(-5000, 50000)
        ax.set_aspect('equal')
        ax.plot(target_traj_x, target_traj_y, color = "green", label = "Target")
        ax.scatter(target_traj_x[i], target_traj_y[i], color = "green")
        plt.text(70000, 51000, f"t = {seekers['PNG']['t_array'][i]:.2f}s", ha='center')
        ax.legend()
    
    ani = animation.FuncAnimation(fig, animate, frames = len(seekers['PNG']['t_array']), interval = 100)
    ani.save('3_laws.gif')
    # plt.show()
    
def get_collision_info(t, seeker_hist, target_hist, d, collision_buffer_range):
    min_dist = [0, np.inf]
    for i in range(2, len(t)):
        
        q, Vr, Vq, R, seeker_crs_angle  = d.guidance.get_coeffs(seeker_hist[:, i], target_hist[:, i])
    
        # keep track of the miss distance
        hasCollided, min_dist_temp = check_agent_collision(Vr, R, t, i, seeker_hist[:2, i], seeker_hist[:2, i-1], target_hist[:2, i], target_hist[:2, i - 1], collision_buffer_range)
        # if hasCollided: 
        #     min_dist = (i, min_dist_temp)    
        if min_dist_temp < min_dist[1]:
            min_dist = (i, min_dist_temp)
            
    return min_dist