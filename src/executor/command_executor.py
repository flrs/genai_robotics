import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from logger import get_logger
from navigation.navigator import Navigator
from shared import SharedResources
from yoloworld.model import visualize

mpl.rcParams["figure.dpi"] = 300

DEBUG = True
LOG_DIR = Path(__file__).parent.parent.joinpath("./logs/")
LOG_DIR.mkdir(exist_ok=True, parents=True)

logger = get_logger(__name__)


class Executor:

    def __init__(self):
        self.resources = SharedResources()
        self.cam_reader = self.resources.cam_reader
        self.model = self.resources.model
        self.roborock = self.resources.roborock
        self.voice_client = self.resources.voice_client

        self.navigator = Navigator()

    def execute_plan(self, plan: list[dict]):
        logger.info(f"Executing plan: {str(plan)}")
        self.voice_client.read_out_loud("Alright, let's execute this plan. Here we go!")
        success = False
        for command in plan:
            success = self.execute_command(command)
            if not success:
                self.voice_client.read_out_loud(
                    "The robot could not complete the task. Aborting the plan and "
                    "sorry about the hiccup!"
                )
                break
        if success:
            self.voice_client.read_out_loud(
                "The plan has been successfully executed. Have an awesome day!"
            )
        return success

    def execute_command(self, command):
        if command["action"] == "MOVE":
            success = self.move(command["location"])
        elif command["action"] == "WAIT_UNTIL":
            success = self.wait_until(command["task"])
        elif command["action"] == "END":
            self.voice_client.read_out_loud(
                "The workflow has completed. Thank you and have a great day!"
            )
            success = True
        return success

    def wait_until(self, task: str, max_wait_s=20):
        self.voice_client.read_out_loud(
            "The robot has stopped because there is a task for you. Please complete the following task:"
        )
        time.sleep(1)
        self.voice_client.read_out_loud(task)
        self.voice_client.read_out_loud(
            f'Please say "I have completed the task." when you have completed the task. '
            f"You have {max_wait_s} seconds to complete the task."
        )
        success = False
        start_time = time.time()
        while time.time() - start_time < max_wait_s:
            user_input = self.voice_client.get_user_input()
            if user_input.lower() == "i have completed the task":
                self.voice_client.read_out_loud(
                    "Thank you for completing the task. The robot will continue now."
                )
                success = True
                break
        return success

    def move(self, target: Tuple[float, float], max_steps: int = 20):
        target_position = {"centroid": target}

        replay = []
        distance_travelled_since_last_frame = np.inf
        goal_achieved = False

        self.roborock.activate_remote_control()

        for step in range(max_steps):
            frame = self.cam_reader.capture()
            recognitions = self.model.predict(frame)
            if step == 0:
                self.roborock.manual_control(0, 0.1, 1000)
                distance = None
                angular_difference = None
            else:
                distance = self.navigator.get_distance_target(
                    recognitions, target_position
                )
                angular_difference = self.navigator.get_distance_angular_difference(
                    recognitions, target_position
                )
                logger.info(
                    "distance:", distance, "angular difference:", angular_difference
                )
                if distance < 50:
                    logger.info("Reached target position.")
                    goal_achieved = True
                    break
                if angular_difference > 5:
                    self.roborock.manual_control(int(angular_difference), 0.1, 1000)
                    time.sleep(3)
                self.roborock.manual_control(0, 0.25, 1000)
            if DEBUG:
                plt.imshow(
                    visualize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), recognitions)
                )
                self.navigator.plot_target_approach(recognitions, target_position)
                plt.axis("off")
                plt.subplots_adjust(
                    left=0, right=1, top=1, bottom=0
                )  # Adjust subplot parameters to remove white border
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    fancybox=True,
                    shadow=True,
                    ncol=5,
                    fontsize="small",
                )  # Move legend to bottom
                distance_str = f"{distance:.2f}" if distance is not None else "N/A"
                angular_difference_str = (
                    f"{angular_difference:.2f}"
                    if angular_difference is not None
                    else "N/A"
                )
                plt.title(
                    f"Step {step + 1} - Distance: {distance_str}, Angular difference: {angular_difference_str} deg"
                )
                plt.savefig(
                    LOG_DIR.joinpath(
                        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )  # Save figure without white border
                plt.show()
            last_position = (
                self.navigator.last_robo_position["centroid"]
                if self.navigator.last_robo_position
                else None
            )
            replay.append(
                {
                    "distance": distance,
                    "angular_difference": angular_difference,
                    "last_position": last_position,
                }
            )
            self.navigator.next_frame(recognitions)
            logger.info(
                f'new robot position: {str(self.navigator.last_robo_position["centroid"])}'
            )
            try:
                distance_travelled_since_last_frame = np.linalg.norm(
                    np.array(last_position)
                    - np.array(self.navigator.last_robo_position["centroid"])
                )
            except TypeError:
                distance_travelled_since_last_frame = np.inf
            if distance_travelled_since_last_frame < 5:
                logger.warn("Robot appears stuck. Turning robot around.")
                self.roborock.manual_control(140, 0.1, 1000)
                time.sleep(3)
            time.sleep(3)

        time.sleep(8)
        self.roborock.deactivate_remote_control()

        return goal_achieved


if __name__ == "__main__":
    executor = Executor()
    plan = [
        {"action": "MOVE", "location": (730.0, 290.0)},
    ]
    executor.execute_plan(plan)
