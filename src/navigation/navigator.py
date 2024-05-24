from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from logger import get_logger
from navigation.data.raw.sample import FRAMES

logger = get_logger(__name__)

def get_robo_position(frame, robo_obj_name="robot vacuum"):
    # Get a list of all vaccum robot objects
    robo_objs = []
    for obj in frame:
        if obj["label"] == robo_obj_name:
            robo_objs.append(obj)
    # Calculate the average brightness of each object
    for obj in robo_objs:
        obj["brightness"] = np.mean(obj["average_color"])
    # Get the brightest object
    if len(robo_objs) == 0:
        return None
    robo_obj = max(robo_objs, key=lambda x: x["brightness"])
    return robo_obj


class Navigator:
    def __init__(self):
        self.last_robo_position = None
        self.pose_history = []
        logger.info("Navigator initialized")


    def next_frame(self, current_frame):
        self.pose_history.append(
            [self.last_robo_position, self.get_current_direction(current_frame)]
        )
        self.last_robo_position = copy(get_robo_position(current_frame))
        logger.info(f"Last robot position:\n{self.last_robo_position}")

    def get_distance_target(self, current_frame, target_position):
        robo_position = get_robo_position(current_frame)
        if robo_position is None:
            return None
        return np.linalg.norm(
            np.array(robo_position["centroid"]) - np.array(target_position["centroid"])
        )

    def get_current_direction(self, current_frame):
        robo_position = get_robo_position(current_frame)
        if robo_position is None or self.last_robo_position is None:
            return None
        direction = np.array(robo_position["centroid"]) - np.array(
            self.last_robo_position["centroid"]
        )
        direction_deg = np.degrees(np.arctan2(direction[1], direction[0]))
        return direction_deg

    def get_distance_angular_difference(self, current_frame, target_position):
        robo_position = get_robo_position(current_frame)
        if robo_position is None or self.last_robo_position is None:
            return None
        # Calculate angular difference
        target_vector = np.array(target_position["centroid"]) - np.array(
            robo_position["centroid"]
        )
        robo_vector = np.array(robo_position["centroid"]) - np.array(
            self.last_robo_position["centroid"]
        )
        difference = np.arccos(
            np.dot(target_vector, robo_vector)
            / (np.linalg.norm(target_vector) * np.linalg.norm(robo_vector))
        )
        cross_product = np.cross(target_vector, robo_vector)
        if cross_product < 0:
            difference = -difference
        difference_deg = np.degrees(difference)
        if np.isnan(difference_deg):
            return 0
        return difference_deg

    def plot_target_approach(self, current_frame, target_position):
        """Plot the current position of the robot, the direction vector, the current direction of the robot, and the target position, using matplotlib"""
        if self.last_robo_position is None:
            return
        robo_position = get_robo_position(current_frame)
        if robo_position is None:
            return
        target_vector = np.array(target_position["centroid"]) - np.array(
            robo_position["centroid"]
        )
        plt.quiver(
            *robo_position["centroid"],
            *target_vector,
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        # Current direction of the robot
        direction = self.get_current_direction(current_frame)
        if direction is not None:
            vector_length = 100
            vector = (
                np.array([np.cos(np.radians(direction)), np.sin(np.radians(direction))])
                * vector_length
            )
            plt.quiver(
                *robo_position["centroid"],
                *vector,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="green",
                label="Current direction",
            )
        plt.scatter(
            *self.last_robo_position["centroid"],
            color="lightblue",
            label="Last robot position",
        )
        plt.scatter(
            *robo_position["centroid"], color="blue", label="Current robot position"
        )
        plt.scatter(*target_position["centroid"], color="red", label="Target position")
        plt.legend()


if __name__ == "__main__":
    data = FRAMES
    target_position = {"centroid": (111.5, 187.0)}

    navigator = Navigator()

    for frame in data:
        distance = navigator.get_distance_target(frame, target_position)
        direction = navigator.get_current_direction(frame)
        angular_difference = navigator.get_distance_angular_difference(
            frame, target_position
        )
        if distance is not None and angular_difference is not None:
            print(
                f"Direction: {direction:.2f}, Distance to target: {distance:.2f}, Angular difference in deg: {angular_difference:.2f}"
            )
        navigator.plot_target_approach(frame, target_position)
        plt.equal()
        plt.show()
        navigator.next_frame(frame)
