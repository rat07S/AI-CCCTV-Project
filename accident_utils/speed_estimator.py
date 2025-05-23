import math

# Calibration constants
METERS_PER_PIXEL = 0.1  # Adjusted for more realistic speeds
MAX_SPEED_KMH = 200.0   # Maximum possible speed
MIN_SPEED_KMH = 5.0     # Minimum speed to consider

def calculate_speed(p1, p2, dt, mpp=METERS_PER_PIXEL):
    """Calculate speed between two points with realistic constraints"""
    try:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        pixels = math.hypot(dx, dy)
        
        # Ignore tiny movements (noise)
        if pixels < 2:
            return 0
            
        meters = pixels * mpp
        mps = meters / dt
        kmh = mps * 3.6  # Convert m/s to km/h
        
        # Apply speed limits
        kmh = max(MIN_SPEED_KMH, min(kmh, MAX_SPEED_KMH))
        
        return kmh
    except Exception as e:
        print(f"Speed calculation error: {e}")
        return 0

def estimate_vehicle_speeds(vehicle_data):
    """Estimate vehicle speeds with smoothing and validation"""
    speeding = {}
    speed_threshold = 80  # km/h - speeding threshold
    
    for vid, history in vehicle_data.items():
        if len(history) < 2:
            continue

        speeds = []
        # Calculate speeds with moving average
        window_size = min(5, len(history) - 1)
        
        for i in range(window_size, len(history)):
            # Use average of last few positions for smoothing
            speeds_window = []
            for j in range(max(0, i - window_size), i):
                p1, t1 = history[j]['center'], history[j]['timestamp']
                p2, t2 = history[j + 1]['center'], history[j + 1]['timestamp']
                dt = t2 - t1
                if dt > 0:
                    speed = calculate_speed(p1, p2, dt)
                    if speed > 0:  # Only consider valid speeds
                        speeds_window.append(speed)
            
            if speeds_window:
                # Use median to filter outliers
                speeds.append(sorted(speeds_window)[len(speeds_window)//2])

        if speeds:
            # Calculate the 90th percentile speed to avoid outliers
            speeds.sort()
            index = int(0.9 * len(speeds))
            representative_speed = speeds[index]
            
            if representative_speed > speed_threshold:
                speeding[vid] = round(representative_speed, 1)

    return {
        "total_vehicles": len(vehicle_data),
        "speeding_vehicles": speeding,
        "count_speeding": len(speeding),
    }
