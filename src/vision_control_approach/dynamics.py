import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

class Vehicle:
    def __init__(self, states_docu, control_docu=None, wheelbase=2.5):
        self.L = wheelbase
        self.states_docu = states_docu
        self.control_docu = control_docu
    
    def dynamics(self, state, control):
        x, y, phi, v = state
        a, delta = control
        
        x_dot = v * np.cos(phi)
        y_dot = v * np.sin(phi)
        phi_dot = (v / self.L) * np.tan(delta)
        v_dot = a
        
        return np.array([x_dot, y_dot, phi_dot, v_dot])
    
    def get_state_names(self):
        return self.states_docu
    
    def get_control_names(self):
        return self.control_docu
    
    def set_hunted_control(self):
        # Constant steering for circular motion
        def control(t):
            # return [0.0, 0.2]  # no acceleration, constant steering
            angl = t*0.05
            acce = 0.0 #np.sin(t)
            return [acce,angl]
        self.control_docu = ["acceleration", "steer_angle"]
        self.control = control
    
class Simulator:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def integrate(self, initial_state, t_end, dt=0.1):
        def dynamics_wrapper(t, state):
            return self.vehicle.dynamics(state, vehicle.control(t))
        
        sol = solve_ivp(dynamics_wrapper, [0, t_end], initial_state, 
                        method='RK45', dense_output=True)
        
        t_eval = np.arange(0, t_end + dt, dt)  # Equidistant time steps
        states = sol.sol(t_eval).T
        
        return t_eval, states
    
    def get_control(self, control_func, times):
        # Get a sample control output to determine dimensions
        sample_control = control_func(times[0])
        if np.isscalar(sample_control):
            # Single control input
            c = np.zeros(len(times))
            for i, t in enumerate(times):
                c[i] = control_func(t)
        else:
            # Multiple control inputs
            control_dim = len(sample_control)
            c = np.zeros((len(times), control_dim))
            for i, t in enumerate(times):
                c[i] = control_func(t)
        
        return c
    
    def simulate(self, initial_state, t_end, dt):
        
        t, s = self.integrate(initial_state, t_end, dt)

        c = self.get_control(vehicle.control, t)

        # Create dictionaries by zipping names with columns of the arrays
        states = {key + " (state)": s[:, i] for i, key in enumerate(self.vehicle.get_state_names())}
        
        # Handle both 1D and 2D control arrays
        if c.ndim == 1:
            controls = {self.vehicle.get_control_names()[0] + " (control)": c}
        else:
            controls = {key + " (control)": c[:, i] for i, key in enumerate(self.vehicle.get_control_names())}
        
        times = {"times": t}
        data = {**times, **states, **controls}
        
        return data
        

class Visualizer():
    def __init__(self):
        True

    def staticPosition2D(self, data_dict, data_dict_hunt=None):
        """
        Plot 2D trajectory from simulation data dictionary
        
        Args:
            data_dict: Dictionary with simulation data (hunted car)
            data_dict_hunt: Dictionary with hunter car data (optional)
        """
        # Extract position data from the hunted car dictionary
        times = data_dict["times"]
        x_positions = data_dict["x_position (state)"]
        y_positions = data_dict["y_position (state)"]
        orientations = data_dict["orientation_angle (state)"]
        
        # Plot hunted car trajectory
        plt.figure(figsize=(12, 10))
        plt.plot(x_positions, y_positions, 'b-', linewidth=2, label='Hunted Car Path')
        plt.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Hunted Start')
        plt.plot(x_positions[-1], y_positions[-1], 'bo', markersize=10, label='Hunted End')
        
        # Show hunted car orientation with arrows every 10 steps
        for i in range(0, len(x_positions), 10):
            x, y, phi = x_positions[i], y_positions[i], orientations[i]
            dx, dy = 2 * np.cos(phi), 2 * np.sin(phi)
            plt.arrow(x, y, dx/2, dy/2, head_width=0.3, head_length=0.3, fc='blue', ec='blue', linewidth=0.5, alpha=0.7)
        
        # Plot hunter car trajectory if provided
        if data_dict_hunt is not None:
            x_positions_hunt = data_dict_hunt["x_position (state)"]
            y_positions_hunt = data_dict_hunt["y_position (state)"]
            orientations_hunt = data_dict_hunt["orientation_angle (state)"]
            
            plt.plot(x_positions_hunt, y_positions_hunt, 'r-', linewidth=2, label='Hunter Car Path')
            plt.plot(x_positions_hunt[0], y_positions_hunt[0], 'mo', markersize=10, label='Hunter Start')
            plt.plot(x_positions_hunt[-1], y_positions_hunt[-1], 'ro', markersize=10, label='Hunter End')
            
            # Show hunter car orientation with arrows every 10 steps
            for i in range(0, len(x_positions_hunt), 10):
                x, y, phi = x_positions_hunt[i], y_positions_hunt[i], orientations_hunt[i]
                dx, dy = 2 * np.cos(phi), 2 * np.sin(phi)
                plt.arrow(x, y, dx/2, dy/2, head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=0.5, alpha=0.7)
        
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Vehicle Trajectories' if data_dict_hunt is not None else 'Vehicle Position')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        print(f"Completed {len(times)} time steps")
        print(f"Hunted car final position: ({x_positions[-1]:.1f}, {y_positions[-1]:.1f})")
        if data_dict_hunt is not None:
            print(f"Hunter car final position: ({x_positions_hunt[-1]:.1f}, {y_positions_hunt[-1]:.1f})")

    def dynamicPosition2D(self, data_dict, dt=0.1):
        """
        Animate vehicle trajectory dynamically from simulation data dictionary
        
        Args:
            data_dict: Dictionary with simulation data (same format as stateHistory)
            dt: Animation delay between frames
        """
        # Extract data from the dictionary
        times = data_dict["times"]
        x_positions = data_dict["x_position (state)"]
        y_positions = data_dict["y_position (state)"]
        orientations = data_dict["orientation_angle (state)"]
        velocities = data_dict["velocity (state)"]
        
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set plot limits based on trajectory
        x_min, x_max = np.min(x_positions) - 5, np.max(x_positions) + 5
        y_min, y_max = np.min(y_positions) - 5, np.max(y_positions) + 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Dynamic Vehicle Motion')
        ax.grid(True)
        ax.axis('equal')
        
        # Initialize plot elements
        trail, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Trail')
        vehicle, = ax.plot([], [], 'ro', markersize=10, label='Vehicle')
        arrow = None
        ax.legend()
        
        # Animate through each time step
        for i, t_current in enumerate(times):
            x, y = x_positions[i], y_positions[i]
            phi, v = orientations[i], velocities[i]
            
            # Update trail (path so far)
            trail.set_data(x_positions[:i+1], y_positions[:i+1])
            
            # Update vehicle position
            vehicle.set_data([x], [y])
            
            # Remove previous arrow and add new one
            if arrow is not None:
                arrow.remove()
            dx, dy = v * np.cos(phi), v * np.sin(phi)
            arrow = ax.arrow(x, y, dx/5, dy/5, head_width=0.5, head_length=0.5, 
                            fc='red', ec='red', alpha=0.8)
            
            # Update title with current info
            ax.set_title(f'Vehicle Motion - t={t_current:.1f}s, v={v:.1f}m/s')
            
            plt.draw()
            plt.pause(dt)  # This forces the plot to update
        
        plt.ioff()  # Turn off interactive mode
        plt.show()


    def stateHistory(self, data_dict, subplot_layout=None):
        """
        Plot multiple time series in subplots
        
        Args:
            times: Time array
            data_dict: Dictionary with keys as plot labels and values as data arrays
            subplot_layout: Tuple (rows, cols) or None for automatic layout
        """
        n_plots = len(data_dict)

        times = data_dict["times"]
        
        # Determine subplot layout
        if subplot_layout is None:
            if n_plots <= 2:
                rows, cols = 1, n_plots
            elif n_plots <= 4:
                rows, cols = 2, 2
            elif n_plots <= 6:
                rows, cols = 2, 3
            elif n_plots <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = int(np.ceil(np.sqrt(n_plots))), int(np.ceil(n_plots / np.ceil(np.sqrt(n_plots))))
        else:
            rows, cols = subplot_layout
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'orange', 'purple']
        
        for i, (label, data) in enumerate(data_dict.items()):
            if i < len(axes):
                color = colors[i % len(colors)]
                axes[i].plot(times, data, color=color, linewidth=2)
                axes[i].set_xlabel('Time [s]')
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{label} vs Time')
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    # Simulate hunted car:
    
    # Initial state: [x, y, phi, v]
    initial_state = [0.0, 0.0, 0.0, 3.0]
    states_docu = ["x_position", "y_position","orientation_angle","velocity"]

    vehicle = Vehicle(states_docu)
    vehicle.set_hunted_control()
    
    vis = Visualizer()
    sim = Simulator(vehicle, )
    sim_data = sim.simulate(initial_state, 20.0, 0.1)
    #vis.dynamicPosition2D(sim_data)
    
    # Create hunting car

    initial_state_hunter = [10.0, 5.0, np.pi/4, 4.0]  # Different position, angle, and speed
    states_docu_hunter = ["x_position", "y_position", "orientation_angle", "velocity"]
    
    # Create hunter vehicle
    hunter_vehicle = Vehicle(states_docu_hunter)
    hunter_vehicle.set_hunted_control()
    
    # Set different control for hunter car
    # def hunter_control(t):
    #     # More aggressive maneuvering for hunting behavior
    #     angl = np.sin(t * 0.3) * 0.4  # Oscillating steering pattern
    #     acce = 0.2 + 0.1 * np.cos(t * 0.2)  # Variable acceleration
    #     return [acce, angl]
    
    # hunter_vehicle.control_docu = ["acceleration", "steer_angle"]
    # hunter_vehicle.control = hunter_control
    
    sim_hunter = Simulator(hunter_vehicle)
    sim_data_hunter = sim_hunter.simulate(initial_state_hunter, 20.0, 0.1)
    
    # Combine both simulation data into one dictionary
    # Add "hunter: " prefix to hunter car keys to distinguish them
    combined_data = sim_data.copy()  # Start with hunted car data
    
    for key, value in sim_data_hunter.items():
        if key == "times":
            # Keep only one times array (they should be the same)
            continue
        else:
            # Add hunter prefix to distinguish from hunted car
            combined_data["hunter: " + key] = value
    
    # Plot both cars' data in one visualization
    # vis.stateHistory(combined_data)

    # TODO: plot the two trajectories static in a 2D plane
    vis.staticPosition2D(sim_data, sim_data_hunter)




    
    