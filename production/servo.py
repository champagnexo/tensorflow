import lgpio
import time

# Configuration
SERVO_PIN = 5
FREQUENCY = 50
MIN_PULSE = 500    # 0.5ms in microseconds
MAX_PULSE = 2500   # 2.5ms in microseconds
NEUTRAL = 1500     # 1.5ms neutral position

def setup_servo():
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(h, SERVO_PIN)
    return h

def set_angle(h, angle):
    # Convert angle to pulse width (500-2500μs)
    pulse = MIN_PULSE + (MAX_PULSE - MIN_PULSE) * (angle + 90) / 180
    lgpio.tx_servo(h, SERVO_PIN, int(pulse))

def test_servo():
    h = setup_servo()
    try:
        print("Testing servo with lgpio...")
        for angle in [-90, -45, 0, 45, 90]:
            print(f"Moving to {angle}°")
            set_angle(h, angle)
            time.sleep(1)
    finally:
        lgpio.gpiochip_close(h)

if __name__ == "__main__":
    test_servo()