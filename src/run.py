Utilimport matplotlib as mpl
import matplotlib.pyplot as plt

from executor.command_executor import Executor
from logger import get_logger
from navigation.navigator import get_robo_position
from shared import SharedResources

mpl.rcParams['figure.dpi'] = 300

logger = get_logger(__name__)
DEBUG = True


class Runner:
    def __init__(self):
        self.resources = SharedResources()
        self.voice_client = self.resources.voice_client
        self.llm_client = self.resources.llm_client
        self.cam_reader = self.resources.cam_reader
        self.model = self.resources.model
        self.executor = Executor()
        logger.info("Runner initialized")

    def observe_environment(self):
        frame = self.cam_reader.capture()
        recognitions = self.model.predict(frame)
        if DEBUG:
            img = self.model.visualize(frame, recognitions)
            # color transformation
            img = img[:, :, [2, 1, 0]]
            plt.imshow(img)
            plt.show()
        return recognitions

    def make_plan(self):
        logger.info("Making a plan")
        observations = self.observe_environment()
        environment_description = self.llm_client.describe_environment(observations)
        self.voice_client.read_out_loud(
            "Hello, and a warm welcome to the future! I am a collaborative robot assistant "
            "powered by a super modern generative AI and deep learning stack."
        )
        self.voice_client.read_out_loud(
            "Just so you know: We can work together on achieving objectives around all"
            " the things that I can see. Here is what I can see:"
        )
        self.voice_client.read_out_loud(environment_description)
        self.voice_client.read_out_loud("Good, now you know what I can see.")
        plan_approved = False
        plan = None
        while not plan_approved:
            self.voice_client.read_out_loud(
                "Please tell me what you want to do today and I will make a plan for us."
            )
            user_input = self.voice_client.get_user_input()
            self.voice_client.read_out_loud("I understood: " + user_input)
            self.voice_client.read_out_loud("Let me think about a plan for us.")
            robot_position = get_robo_position(observations)["centroid"]
            plan, reasoning = self.llm_client.get_plan(user_input, observations, robot_position)
            plan_explanation = self.llm_client.explain_plan(plan, reasoning)
            self.voice_client.read_out_loud(
                "I have come up with a plan for us. Here it is:"
            )
            self.voice_client.read_out_loud(plan_explanation)
            self.voice_client.read_out_loud(
                "Do you agree with the plan? Please answer yes or no. If you want to abort, say abort."
            )
            user_input = self.voice_client.get_user_input()
            self.voice_client.read_out_loud("I understood: " + user_input)
            if 'yes' in user_input.lower():
                plan_approved = True
                logger.info("Plan approved")
            elif "abort" in user_input.lower():
                self.voice_client.read_out_loud(
                    "Alright, we will abort the plan. Have a great day!"
                )
                logger.warn("Plan aborted")
                return
            else:
                self.voice_client.read_out_loud("Sorry about that. Let us try again.")
                logger.info("Plan not approved")
        return plan

    def execute_plan(self, plan):
        self.executor.execute_plan(plan)


if __name__ == "__main__":
    # add e
    runner = Runner()
    plan = runner.make_plan()
    if plan is not None:
        runner.execute_plan(plan)
