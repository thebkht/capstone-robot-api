import serial
import time
import json

class Rover:
    """
    Simple high-level controller for an ESP32-based UGV rover.

    Supports movement commands like forward, backward, left, right
    with configurable distance (m) and speed (slow/medium/fast).
    """

    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # wait for the serial connection to initialize
        self.speeds = {'slow': 0.2, 'medium': 0.4, 'fast': 0.7}
        self.last_status = {}  # Store last received status data
        print(f"[Rover] Connected on {port} at {baudrate} baud.")

    def _send(self, L, R):
        """Send a single JSON movement command to the rover."""
        cmd = f'{{"T":1,"L":{L:.2f},"R":{R:.2f}}}\r\n'
        self.ser.write(cmd.encode())

    def _stop(self):
        """Send stop command."""
        self.ser.write(b'{"T":1,"L":0,"R":0}\r\n')

    def move(self, direction, distance_m=0.5, speed_label='medium'):
        """
        Move the rover in a given direction for a set distance and speed.

        Args:
            direction (str): 'forward', 'backward', 'left', or 'right'
            distance_m (float): distance to move in meters (approx)
            speed_label (str): 'slow', 'medium', or 'fast'
        """
        if speed_label not in self.speeds:
            raise ValueError("Speed must be 'slow', 'medium', or 'fast'")

        speed = self.speeds[speed_label]
        duration = distance_m / speed if speed > 0 else 0

        # Map directions to left/right wheel speeds
        if direction == 'forward':
            L, R = speed, speed
        elif direction == 'backward':
            L, R = -speed, -speed
        elif direction == 'left':
            L, R = -speed, speed
        elif direction == 'right':
            L, R = speed, -speed
        else:
            print(f"[Rover] Invalid direction '{direction}'")
            return

        print(f"[Rover] Moving {direction} for {distance_m:.2f} m at {speed_label} speed...")

        # Send command repeatedly (10 Hz) to maintain motion
        start = time.time()
        while time.time() - start < duration:
            self._send(L, R)
            time.sleep(0.1)

        # Stop after the duration
        self._stop()
        print("[Rover] Movement complete. Stopped.")

    def stop(self):
        """Manually stop the rover."""
        self._stop()
        print("[Rover] Emergency stop.")
    
    def set_camera_servo(self, pan=90, tilt=15):
        """
        Control camera pan/tilt servo (gimbal).
        
        Args:
            pan: Pan angle (0-180): 0=left, 90=center, 180=right
            tilt: Tilt angle (0-180): 0=down, ~15=forward, 90=up
        """
        pan = max(0, min(180, int(pan)))
        tilt = max(0, min(180, int(tilt)))
        # Use T:133 (CMD_GIMBAL_CTRL_SIMPLE) with X/Y parameters
        cmd = f'{{"T":133,"X":{pan},"Y":{tilt},"SPD":0,"ACC":0}}\r\n'
        self.ser.write(cmd.encode())
        time.sleep(0.05)  # Small delay for servo movement

    def display_text(self, line_num, text):
        """
        Display custom text on the OLED screen.
        
        The 0.91" OLED has 4 lines (0-3), each can hold ~21 characters.
        
        Args:
            line_num (int): Line number 0-3 (top to bottom)
            text (str): Text to display on the line
        """
        if not 0 <= line_num <= 3:
            print(f"[Rover] Invalid line number {line_num}. Must be 0-3.")
            return
        
        # Truncate text if too long (max ~21 chars per line for 128px width)
        text = str(text)[:21]
        cmd = f'{{"T":3,"lineNum":{line_num},"Text":"{text}"}}\r\n'
        self.ser.write(cmd.encode())
        time.sleep(0.01)
    
    def display_multiline(self, lines):
        """
        Display multiple lines of text on the OLED screen.
        
        Args:
            lines (list): List of up to 4 strings (one per line)
        
        Example:
            rover.display_multiline(["Hello", "Rover", "Status: OK", "Battery: 12.5V"])
        """
        for i, text in enumerate(lines[:4]):
            self.display_text(i, text)
    
    def display_reset(self):
        """
        Reset the OLED display to default mode.
        Shows system info like WiFi status and voltage.
        """
        cmd = '{"T":-3}\r\n'
        self.ser.write(cmd.encode())
        print("[Rover] Display reset to default mode.")
    
    def read_feedback(self):
        """
        Read feedback data from the rover (non-blocking).
        
        Returns parsed JSON dict if available, otherwise None.
        Feedback includes: battery voltage, temperature, IMU data, wheel speeds, etc.
        
        Example feedback:
        {
            "T": 650,       # Feedback type
            "L": 0.0,       # Left wheel speed
            "R": 0.0,       # Right wheel speed
            "v": 12.45,     # Battery voltage
            "temp": 28.5,   # Temperature (°C)
            "r": 0.1,       # IMU Roll
            "p": -0.2,      # IMU Pitch
            "y": 45.3       # IMU Yaw
        }
        """
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    data = json.loads(line)
                    # Update last status cache
                    if data.get('T') == 650:  # Base info feedback
                        self.last_status = data
                    return data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Ignore malformed data
                pass
        return None
    
    def get_status(self):
        """
        Get the latest rover status by requesting fresh data.
        
        Returns dict with keys like:
        - voltage: Battery voltage in volts
        - temperature: Temperature in Celsius
        - left_speed, right_speed: Wheel speeds
        - roll, pitch, yaw: IMU orientation
        """
        # Request status update from rover
        cmd = '{"T":130}\r\n'
        self.ser.write(cmd.encode())
        
        # Wait a bit for response
        time.sleep(0.1)
        
        # Try to read fresh feedback (with multiple attempts)
        max_attempts = 5
        for _ in range(max_attempts):
            feedback = self.read_feedback()
            if feedback and feedback.get('T') == 1001:
                # Got fresh base info feedback
                self.last_status = feedback
                break
            time.sleep(0.02)  # Small delay between read attempts
        
        # Use last known status (either fresh or cached)
        status_data = self.last_status
        
        # Parse into friendly format
        status = {
            'voltage': status_data.get('v', 0.0),
            'temperature': status_data.get('temp', 0.0),
            'left_speed': status_data.get('L', 0.0),
            'right_speed': status_data.get('R', 0.0),
            'roll': status_data.get('r', 0.0),
            'pitch': status_data.get('p', 0.0),
            'yaw': status_data.get('y', 0.0),
        }
        return status
    
    
    def cleanup(self):
        """Release serial port safely."""
        self._stop()
        self.ser.close()
        print("[Rover] Serial connection closed.")
