import airsim
import numpy as np
import time

#connecting to airsim
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable control
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Move to a safe altitude
client.moveToZAsync(-10, 2).join()

waypoints = [
    (0, 0, -10),
    (10, 0, -10),
    (10, 10, -10),
    (20, 10, -10)
]

# Energy tracking
total_energy = 0
prev_pos = None

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Fly through waypoints
for wp in waypoints:
    print(f"Moving to {wp}")
    
    client.moveToPositionAsync(wp[0], wp[1], wp[2], 3).join()
    
    # Get current position
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    current_pos = (pos.x_val, pos.y_val, pos.z_val)
    
    # Energy model (simple)
    if prev_pos is not None:
        distance = compute_distance(prev_pos, current_pos)
        energy = distance * 1.0   # α = 1.0 (tunable)
        total_energy += energy
    
    prev_pos = current_pos

print(f"Total Energy Used: {total_energy}")

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

