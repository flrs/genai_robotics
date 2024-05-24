import math
import subprocess
import time
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

def run_bash_command(device_id, command, params=None):
    command_str = "roborock -d command --device_id " + device_id + " --cmd " + command
    if params:
        command_str += " --params " + params
    logger.info(f"Sending command to Roborock: {command_str.replace(device_id, '***')}")
    process = subprocess.Popen(
        command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        logger.error(f"Error executing command: {stderr.decode()}")
    return stderr.decode()


class Roborock:
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.manual_seqnum = 0

    def get_status(self):
        # Submit the following as a process and get the result
        # roborock -d command --device_id 7RC4xWPqXwOMKA0tN6JIhB --cmd get_status
        return run_bash_command(self.device_id, "get_status")

    def activate_remote_control(self):
        run_bash_command(self.device_id, "app_rc_start")
        time.sleep(10)
        self.turn_off_fan()
        time.sleep(5)

    def deactivate_remote_control(self):
        return run_bash_command(self.device_id, "app_rc_stop")

    def turn_off_fan(self):
        params = {"fan_speed": "gentle"}
        formatted_params = str([params]).replace("'", '"').replace(" ", "")
        return run_bash_command(
            self.device_id, "set_custom_mode", f"'{formatted_params}'"
        )

    def manual_control(self, rotation: int, velocity: float, duration: int = 1500):
        """Give a command over manual control interface."""
        if rotation <= -180 or rotation >= 180:
            raise ValueError(
                "Given rotation is invalid, should " "be ]-3.1,3.1[, was %s" % rotation
            )
        if velocity <= -0.3 or velocity >= 0.3:
            raise ValueError(
                "Given velocity is invalid, should "
                "be ]-0.3, 0.3[, was: %s" % velocity
            )

        self.manual_seqnum += 1
        params = {
            "omega": round(math.radians(rotation), 1),
            "velocity": velocity,
            "duration": duration,
            "seqnum": self.manual_seqnum,
        }
        formatted_params = str([params]).replace("'", '"').replace(" ", "")
        return run_bash_command(self.device_id, "app_rc_move", f"'{formatted_params}'")


if __name__ == "__main__":
    robot = Roborock(device_id="***")
    robot.activate_remote_control()
    time.sleep(5)
    robot.turn_off_fan()
    time.sleep(15)
    robot.manual_control(rotation=179, velocity=0.1, duration=5000)
    robot.manual_control(rotation=90, velocity=0.2, duration=5000)
    robot.manual_control(rotation=0, velocity=0.2, duration=5000)
    time.sleep(15)
    robot.deactivate_remote_control()
