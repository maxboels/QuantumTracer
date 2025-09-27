import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

class BaseVehicle:
    """Base class for vehicle dynamics"""
    def __init__(self, states_docu, control_docu=None, wheelbase=2.5):
        self.L = wheelbase
        self.states_docu = states_docu
        self.control_docu = control_docu
    
    def dynamics(self, state, control):
        """Common vehicle dynamics model"""
        x, y, phi, v = state
        a, delta = control
        
        x_dot = v * np.cos(phi)
        y_dot = v * np.sin(phi)
        phi_dot = (v / self.L) * np.tan(delta)
        v_dot = a
        
        return [x_dot, y_dot, phi_dot, v_dot]
    
    def get_state_names(self):
        return self.states_docu
    
    def get_control_names(self):
        return self.control_docu

class HuntedVehicle(BaseVehicle):
    """Vehicle that is being hunted - simple time-based control"""
    def __init__(self, states_docu, wheelbase=2.5):
        super().__init__(states_docu, ["acceleration", "steer_angle"], wheelbase)
        self._setup_control()
    
    def _setup_control(self):
        """Setup simple time-based control for circular motion"""
        def control(t, state):
            angl = t * 0.05  # Gradually increasing steering angle
            acce = 0.0       # No acceleration
            return [acce, angl]
        
        self.control = control

class HunterVehicle(BaseVehicle):
    """Vehicle that hunts the target - state-feedback control"""
    def __init__(self, states_docu, wheelbase=2.5):
        super().__init__(states_docu, ["acceleration", "steer_angle"], wheelbase)
        self._setup_control()
    
    def _setup_control(self):
        """Setup hunting control that tracks the target"""
        def control(t, state, target_state):
            # Measure angle to target
            angle_to_target = self.measure(state, target_state)
            
            # Estimate steering command
            angle = self.estimate(angle_to_target)
            angle_steer = 0.3 * angle
            acceleration = 0.0  # Constant speed hunting
            
            return [acceleration, angle_steer]
        
        self.control = control
    
    def measure(self, state_own, state_target):
        """Measure relative angle to target vehicle"""
        x1, y1, phi1, v1 = state_target  # Target position
        x2, y2, phi2, v2 = state_own     # Own position and heading
        
        # Compute relative position vector
        distance_x = x1 - x2
        distance_y = y1 - y2
        
        # Compute absolute angle from hunter to target (in global frame)
        target_angle_global = np.arctan2(distance_y, distance_x)
        
        # Compute relative angle: difference between target direction and current heading
        relative_angle = target_angle_global - phi2
        
        # Wrap angle to [-pi, pi] for shortest turn
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
            
        return relative_angle
    
    def estimate(self, measurement):
        """Convert measurement to steering command"""
        angle = measurement
        return angle  # Proportional control gain
    

class Simulator:
    def __init__(self, hunted, hunter):
        self.hunter = hunter
        self.hunted = hunted
        
    def integrate(self, initial_state, t_end, dt=0.01):  # Much smaller time step
        """Integrate vehicle dynamics over time with high-frequency control updates"""
        def dynamics_wrapper(t, state):
            # Split the 8-dimensional state into hunted (0:4) and hunter (4:8)
            state_hunted = state[0:4]
            state_hunter = state[4:8]

            # Get control inputs for both vehicles - this happens at every integration step
            control_input_hunted = self.hunted.control(t, state_hunted)
            control_input_hunter = self.hunter.control(t, state_hunter, state_hunted)

            # Get dynamics derivatives for both vehicles
            d_hunted = self.hunted.dynamics(state_hunted, control_input_hunted)
            d_hunter = self.hunter.dynamics(state_hunter, control_input_hunter)
            
            # Combine derivatives into 8-dimensional vector
            return d_hunted + d_hunter
        
        # Use smaller max_step to ensure frequent control updates
        sol = solve_ivp(dynamics_wrapper, [0, t_end], initial_state, 
                        method='RK45', dense_output=True, max_step=0.01)
        
        # Generate output at desired frequency (can be different from integration step)
        t_eval = np.arange(0, t_end + dt, dt)
        states = sol.sol(t_eval).T
        self.sol = sol
        
        return t_eval, states
    
    def get_control(self, states, times):
        """Get control inputs over time for both vehicles"""
        # states is now 8-dimensional: [hunted_x, hunted_y, hunted_phi, hunted_v, hunter_x, hunter_y, hunter_phi, hunter_v]
        n_times = len(times)
        
        # Initialize control arrays for both vehicles (2 controls each: acceleration, steering)
        controls_hunted = np.zeros((n_times, 2))
        controls_hunter = np.zeros((n_times, 2))
        
        for i, t in enumerate(times):
            state_hunted = states[i, 0:4]
            state_hunter = states[i, 4:8]
            
            # Get control inputs for both vehicles
            controls_hunted[i] = self.hunted.control(t, state_hunted)
            controls_hunter[i] = self.hunter.control(t, state_hunter, state_hunted)
        
        return controls_hunted, controls_hunter
    
    def simulate(self, init_hunted, init_hunter, t_end, dt):
        """Simulate both vehicles together"""
        # Combine initial states into 8-dimensional vector
        initial_state = init_hunted + init_hunter
        
        t, s = self.integrate(initial_state, t_end, dt)
        c_hunted, c_hunter = self.get_control(s, t)

        # Split states back into hunted and hunter components
        states_hunted = s[:, 0:4]  # First 4 columns for hunted vehicle
        states_hunter = s[:, 4:8]  # Last 4 columns for hunter vehicle
        
        # Create separate data dictionaries for hunted and hunter vehicles
        hunted_states = {key: states_hunted[:, i] for i, key in enumerate(self.hunted.get_state_names())}
        hunter_states = {key: states_hunter[:, i] for i, key in enumerate(self.hunter.get_state_names())}
        
        #hunted_controls = {key + " (control)": c_hunted[:, i] for i, key in enumerate(self.hunted.get_control_names())}
        hunter_controls = {key + " (control)": c_hunter[:, i] for i, key in enumerate(self.hunter.get_control_names())}
        
        times = {"times": t}
        
        # Create separate data dictionaries
        hunted_data = {**times, **hunted_states}
        hunter_data = {**times, **hunter_states, **hunter_controls}
        
        return hunted_data, hunter_data
        

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
        x_positions = data_dict["x_position"]
        y_positions = data_dict["y_position"]
        orientations = data_dict["orientation_angle"]
        
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
            x_positions_hunt = data_dict_hunt["x_position"]
            y_positions_hunt = data_dict_hunt["y_position"]
            orientations_hunt = data_dict_hunt["orientation_angle"]
            
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
        # print(f"Hunted car final position: ({x_positions[-1]:.1f}, {y_positions[-1]:.1f})")
        # if data_dict_hunt is not None:
        #     print(f"Hunter car final position: ({x_positions_hunt[-1]:.1f}, {y_positions_hunt[-1]:.1f})")

    def dynamicPosition2D(self, hunted_data, hunter_data=None, dt=0.05):
        """
        Animate vehicle trajectory dynamically from simulation data dictionary
        
        Args:
            hunted_data: Dictionary with hunted vehicle simulation data
            hunter_data: Dictionary with hunter vehicle simulation data (optional)
            dt: Animation delay between frames
        """
        # Extract data from the hunted vehicle dictionary
        times = hunted_data["times"]
        x_positions_hunted = hunted_data["x_position"]
        y_positions_hunted = hunted_data["y_position"]
        orientations_hunted = hunted_data["orientation_angle"]
        velocities_hunted = hunted_data["velocity"]
        
        # Extract hunter data if provided
        if hunter_data is not None:
            x_positions_hunter = hunter_data["x_position"]
            y_positions_hunter = hunter_data["y_position"]
            orientations_hunter = hunter_data["orientation_angle"]
            velocities_hunter = hunter_data["velocity"]
        
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set plot limits based on both trajectories
        all_x = x_positions_hunted
        all_y = y_positions_hunted
        if hunter_data is not None:
            all_x = np.concatenate([x_positions_hunted, x_positions_hunter])
            all_y = np.concatenate([y_positions_hunted, y_positions_hunter])
            
        x_min, x_max = np.min(all_x) - 5, np.max(all_x) + 5
        y_min, y_max = np.min(all_y) - 5, np.max(all_y) + 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Dynamic Vehicle Motion')
        ax.grid(True)
        ax.axis('equal')
        
        # Initialize plot elements for hunted vehicle
        trail_hunted, = ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Hunted Trail')
        vehicle_hunted, = ax.plot([], [], 'bo', markersize=12, label='Hunted Vehicle')
        arrow_hunted = None
        
        # Initialize plot elements for hunter vehicle if present
        if hunter_data is not None:
            trail_hunter, = ax.plot([], [], 'r-', alpha=0.6, linewidth=2, label='Hunter Trail')
            vehicle_hunter, = ax.plot([], [], 'ro', markersize=12, label='Hunter Vehicle')
            arrow_hunter = None
        
        ax.legend()
        
        # Animate through each time step
        for i, t_current in enumerate(times):
            # Update hunted vehicle
            x_h, y_h = x_positions_hunted[i], y_positions_hunted[i]
            phi_h, v_h = orientations_hunted[i], velocities_hunted[i]
            
            # Update hunted trail (path so far)
            trail_hunted.set_data(x_positions_hunted[:i+1], y_positions_hunted[:i+1])
            
            # Update hunted vehicle position
            vehicle_hunted.set_data([x_h], [y_h])
            
            # Remove previous arrow and add new one for hunted vehicle
            if arrow_hunted is not None:
                arrow_hunted.remove()
            dx_h, dy_h = 3 * np.cos(phi_h), 3 * np.sin(phi_h)
            arrow_hunted = ax.arrow(x_h, y_h, dx_h/2, dy_h/2, head_width=0.8, head_length=0.8, 
                                   fc='blue', ec='blue', alpha=0.8)
            
            # Update hunter vehicle if present
            if hunter_data is not None:
                x_hunt, y_hunt = x_positions_hunter[i], y_positions_hunter[i]
                phi_hunt, v_hunt = orientations_hunter[i], velocities_hunter[i]
                
                # Update hunter trail
                trail_hunter.set_data(x_positions_hunter[:i+1], y_positions_hunter[:i+1])
                
                # Update hunter vehicle position
                vehicle_hunter.set_data([x_hunt], [y_hunt])
                
                # Remove previous arrow and add new one for hunter vehicle
                if arrow_hunter is not None:
                    arrow_hunter.remove()
                dx_hunt, dy_hunt = 3 * np.cos(phi_hunt), 3 * np.sin(phi_hunt)
                arrow_hunter = ax.arrow(x_hunt, y_hunt, dx_hunt/2, dy_hunt/2, head_width=0.8, head_length=0.8, 
                                       fc='red', ec='red', alpha=0.8)
                
                # Calculate distance between vehicles
                distance = np.sqrt((x_hunt - x_h)**2 + (y_hunt - y_h)**2)
                
                # Update title with current info for both vehicles
                ax.set_title(f'Hunter-Prey Simulation - t={t_current:.1f}s\n'
                           f'Hunted: v={v_h:.1f}m/s | Hunter: v={v_hunt:.1f}m/s | Distance: {distance:.1f}m')
            else:
                # Update title with current info for single vehicle
                ax.set_title(f'Vehicle Motion - t={t_current:.1f}s, v={v_h:.1f}m/s')
            
            plt.draw()
            plt.pause(dt)  # Animation delay
        
        plt.ioff()  # Turn off interactive mode
        plt.show()


    def stateHistory(self, sim_data_hunted, sim_data_hunter, subplot_layout=None):
        """
        Plot multiple time series in subplots
        
        Args:
            sim_data_hunted: Dictionary with hunted vehicle data
            sim_data_hunter: Dictionary with hunter vehicle data  
            subplot_layout: Tuple (rows, cols) or None for automatic layout
        """
        times = sim_data_hunted["times"]
        
        # Collect all data to plot (excluding "times" key)
        plot_data = []
        
        # Add hunted vehicle data
        for key, data in sim_data_hunted.items():
            if key != "times":
                plot_data.append((f"{key} (hunted)", data, 'blue'))
        
        # Add hunter vehicle data
        for key, data in sim_data_hunter.items():
            if key != "times":
                plot_data.append((f"{key} (hunter)", data, 'red'))
        
        n_plots = len(plot_data)
        
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
        
        # Plot each data series
        for i, (label, data, color) in enumerate(plot_data):
            if i < len(axes):
                axes[i].plot(times, data, color=color, linewidth=2, label=label)
                axes[i].set_xlabel('Time [s]')
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{label} vs Time')
                axes[i].grid(True)
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Create both vehicles
    initial_state_hunted = [0.0, 0.0, 0.0, 3.0]  # [x, y, phi, v]
    initial_state_hunter = [10.0, 5.0, np.pi/4, 4.0]  # Different position, angle, and speed
    
    states_docu = ["x_position", "y_position", "orientation_angle", "velocity"]
    
    hunted_vehicle = HuntedVehicle(states_docu)
    hunter_vehicle = HunterVehicle(states_docu)
    
    # Create combined simulator
    sim = Simulator(hunted_vehicle, hunter_vehicle)
    
    print("Starting combined simulation...")
    sim_data_hunted, sim_data_hunter = sim.simulate(initial_state_hunted, initial_state_hunter, 20.0, 0.1)
    print("Simulation completed!")
    
    # Visualize results
    vis = Visualizer()
    
    # Static trajectory plot
    # vis.staticPosition2D(sim_data_hunted, sim_data_hunter)
    
    # Dynamic animation of the chase
    print("Starting dynamic animation...")
    vis.dynamicPosition2D(sim_data_hunted, sim_data_hunter, dt=0.02)  # Fast animation
    
    # State history plots
    # vis.stateHistory(sim_data_hunted, sim_data_hunter)




    
    