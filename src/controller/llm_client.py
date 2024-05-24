import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from controller.data.raw.recognition_sample import RECOGNITIONS
from logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, openai_key: str):
        self.client = OpenAI(api_key=openai_key)
        logger.info("LLMClient initialized")


    def create_plan_prompt(
        self, user_input, recognitions, robot_vacuum_position: Tuple[float, float]
    ):
        recognitions_table = self.generate_recognitions_input(
            recognitions, robot_vacuum_position
        )
        distance_table = pd.DataFrame(
            np.nan, index=recognitions_table["id"], columns=recognitions_table["id"]
        )
        # add distance between recognitions
        for nx1, recognition1 in recognitions_table.iterrows():
            for nx2, recognition2 in recognitions_table.iterrows():
                distance = np.linalg.norm(
                    np.array(recognition1["position"])
                    - np.array(recognition2["position"])
                )
                distance_table.loc[nx1, nx2] = distance
        distance_table = distance_table.astype(int)

        # Create DataFrame
        recognitions_table_markdown = recognitions_table.to_markdown(index=False)
        distance_table_markdown = distance_table.to_markdown()

        prompt_template = (
            Path(__file__)
            .parent.joinpath("./data/raw/prompt_template_plan.txt")
            .read_text()
        )
        prompt_template = prompt_template.replace("<USER_INPUT>", user_input)
        prompt_template = prompt_template.replace(
            "<CURRENT_LOCATION>", str(robot_vacuum_position)
        )
        prompt_template = prompt_template.replace(
            "<ITEM_LOCATIONS>", recognitions_table_markdown
        )
        prompt_template = prompt_template.replace(
            "<DISTANCES>", distance_table_markdown
        )
        return prompt_template

    @staticmethod
    def generate_recognitions_input(
        recognitions, robot_vacuum_position: Optional[Tuple[float, float]] = None
    ):
        recognitions_table = []
        for nx, recognition in enumerate(recognitions):
            recognitions_table.append(
                {
                    "id": nx,
                    "label": recognition["label"],
                    "position": recognition["centroid"],
                    "confidence": round(recognition["confidence"], 2),
                    "color_rgb": [
                        np.clip(int(x * 255), 0, 255)
                        for x in recognition["average_color"]
                    ],
                }
            )
        # Find the row with the robot vacuum, place it on top of the list, and remove all other rows with the label 'robot vacuum'
        if robot_vacuum_position is not None:
            robot_vacuum_row = None
            for nx, recognition in enumerate(recognitions_table):
                if (
                    recognition["label"] == "robot vacuum"
                    and recognition["position"] == robot_vacuum_position
                ):
                    robot_vacuum_row = recognition
                    break
            if robot_vacuum_row is not None:
                recognitions_table = [
                    recognition
                    for recognition in recognitions_table
                    if recognition["label"] != "robot vacuum"
                ]
                recognitions_table.insert(0, robot_vacuum_row)
            else:
                raise ValueError("Robot vacuum not found in recognitions.")
        # reset index column from 0 to n
        for nx, recognition in enumerate(recognitions_table):
            recognition["id"] = nx
        recognitions_table = pd.DataFrame(recognitions_table)
        return recognitions_table

    def query(self, prompt):
        logger.info(f"Querying LLM with prompt: {prompt}")
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(completion.choices[0].message.content)

    def get_plan(
        self, user_input, recognitions, robot_vacuum_position: Tuple[float, float]
    ):
        prompt = self.create_plan_prompt(
            user_input, recognitions, robot_vacuum_position
        )
        res = self.query(prompt)
        plan = res["plan"]
        reasoning = res["reasoning"]
        return plan, reasoning

    def explain_plan(self, plan, reasoning):
        prompt_template = (
            Path(__file__)
            .parent.joinpath("./data/raw/prompt_template_explanation.txt")
            .read_text()
        )
        prompt_template = prompt_template.replace("<PLAN>", str(plan))
        prompt_template = prompt_template.replace("<REASONING>", str(reasoning))
        res = self.query(prompt_template)
        return res["summary"]

    def describe_environment(self, recognitions):
        recognitions_input = self.generate_recognitions_input(recognitions)
        # Drop columns id and confidence
        recognitions_input = recognitions_input.drop(columns=["id", "confidence"])
        prompt_template = (
            Path(__file__)
            .parent.joinpath("./data/raw/prompt_template_description.txt")
            .read_text()
        )
        prompt_template = prompt_template.replace(
            "<TABLE_OF_OBJECTS>", recognitions_input.to_markdown(index=False)
        )
        return self.query(prompt_template)


if __name__ == "__main__":
    llm = LLMClient()

    client = OpenAI(api_key=openai_key)
    recognitions = RECOGNITIONS
    print("Environment description")
    description = llm.describe_environment(recognitions)
    print(description)

    user_input = "go to the closest vase, ask the user to count to ten, and then go to the middle of the room"
    current_position = (799.5, 33.5)
    plan = llm.get_plan(user_input, recognitions, current_position)
    explanation = llm.explain_plan(plan)
    print("OUR PLAN")
    print(plan["plan"])
    print("EXPLANATION")
    print(explanation)

    print("done")
