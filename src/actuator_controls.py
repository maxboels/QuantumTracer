import RPi.GPIO as GPIO
import os

SPEED_DAMPING_FACTOR = os.getenv("SPEED_DAMPING_FACTOR", 0.4)

assert SPEED_DAMPING_FACTOR > 0 and SPEED_DAMPING_FACTOR <= 1, "SPEED_DAMPING_FACTOR must be between 0 and 1"

# GPIO.cleanup()

fwd_bck_pin = 13
left_right_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(left_right_pin, GPIO.OUT)
GPIO.setup(fwd_bck_pin, GPIO.OUT)
fwd_bck_pwm = GPIO.PWM(fwd_bck_pin, 900)  # 900 Hz
left_right_pwm = GPIO.PWM(left_right_pin, 50)  # 50 Hz

left_right_pwm.start(7)  # Neutral position

fwd_bck_pwm.start(0)  # Stopped


class ActuatorControls:
    def __init__(self):
        pass

    def set_fwd_speed(self, speed):
        if speed < 0 or speed > 1:
            print(f"Speed must be between 0 and 1. Ignoring command {speed}")
            return
        
        # LOW SPEED OVERRIDE
        fwd_bck_pwm.ChangeDutyCycle(0 if speed < 0.1 else 10)
        
        # fwd_bck_pwm.ChangeDutyCycle(int(speed * 70 * SPEED_DAMPING_FACTOR))

    def set_steering_angle(self, angle):
        if angle < -1 or angle > 1:
            print(f"Angle must be between -1 and 1. Ignoring command {angle}")
            return
        
        duty_cycle = 7.7 + angle * 2  # Map -1 to 1 -> 5% to 9%
        print(f"Setting steering angle to {angle} (duty cycle {duty_cycle}%)")
        left_right_pwm.ChangeDutyCycle(duty_cycle)

    def stop(self):
        fwd_bck_pwm.stop()
        left_right_pwm.stop()
        GPIO.cleanup()
