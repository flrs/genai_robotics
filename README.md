# Robotics Control with Generative AI

This repository contains an experimental, privacy-aware setup for leveraging generative AI methods in robotics control. With the solution presented here, a user can freely define actions by voice that are translated into plans that a robot vacuum can execute in an open-world environment that is observed by a camera.

![Overview](/docs/overview.png)

The fundamental advantages of the methods presented here are:
- The system **only requires minimum calibration** toward its environment
- Robot **planning and control** are taken over by a large-language model

The system was developed in a 3-day hackathon as a learning exercise and proof-of-concept that modern AI tools can cut down development time for robotics control solutions significantly.

## Prerequisites

To use all features of this repository, here is what you should have:

- A Roborock robot vacuum
- An OpenAI API key (tested with the GPT-4o model)
- A camera that has a birds-eye view on the environment the robot operates in, exposed via an rtsp connection over the local network (many home surveillance cameras have this kind of interface)
- A Mac OS device for running the code (the code leverages Mac OS native text-to-speech functionality)

## Getting Started

To get started, follow the steps below:

1. Clone this repository 
2. Install the requirements from `requirements.txt` into a Python environment (testes with Python 3.11)
3. Set up the credentials for running the code
   1. Rename the file `src/config.template.toml` to `config.toml`. For all the steps below, insert the acquired credentials into `config.toml`
   2. Get the device ID of your Roborock vacuum. You can read more about how to do this in the documentation of the `python-roborock` library.
   3. Get username, password, IP, and channel name for the rtsp connection to the camera. Note that for the stream to work properly, the camera should be connected to the same network of the device you are running the code on.
   4. Get an API key for OpenAI.
5. Once you have credentials configured as described above, run `src/run.py` to run the workflow.

## Architecture and Functionality

The best way to understand what this repository does in detail and how the elements interact is by an architecture diagram:

![Architecture Diagram](/docs/architecture.png)

When you run the `run.py` file as described above, here is what happens and how it works:

### Create a Plan

The system is greeting the user with an audio message and expects them to tell the system what they want to do. For example, a user might want the robot to pick up a coffee from a person that is sitting on a yellow chair and transport it to another person that is seated on a black sofa. The system would then create a plan to execute these actions.

What does the system need to understand how it can achieve what the user wants to do? The system needs to be aware of its environment and the actions that can be executed in this environment. Here, we use a computer vision model with object detection to provide information about the environment to the system. The vacuum itself can execute 3 simple actions: Move forward, turn, and do nothing. Another action in the environment is waiting for the user to perform a certain action.

To avoid confusion on the user side, it is important that the user knows how the AI perceives its environment. For example, if an object is not recognized by the computer vision model, the AI will be unable to include it in a plan. It is also important that the user is aware that there is uncertainty with regards to the models' recognition. Using OpenAI's GPT-4o large language model with the [description prompt](./src/controller/data/raw/prompt_template_description.txt), the system comes up with an explanation of its environment and reads it to the user just before asking the user what they want the system to do.

Given the environment information and the user input with regards to what they want to do, the system can then come up with a plan. Here, we ask the LLM to come up with a plan, given the user's inputs and the description of the environment. You can find the prompt template in the [`controller` directory](./src/controller/data/raw/prompt_template_plan.txt). The exciting trick here is that the LLM is only aware of its environment through two tables that are generated from the outputs of the computer vision model. Here is an example:

```
Item locations:
|   id | label        | position        |   confidence | color_rgb       |
|-----:|:-------------|:----------------|-------------:|:----------------|
|    0 | robot vacuum | (122.0, 140.0)  |         0.23 | [205, 206, 210] |
|    1 | blanket      | (1697.0, 923.0) |         0.59 | [60, 72, 90]    |
|    2 | chair        | (532.5, 210.0)  |         0.39 | [177, 177, 171] |
|    3 | chair        | (160.0, 521.5)  |         0.24 | [99, 99, 98]    |
|    4 | book         | (1216.5, 601.0) |         0.2  | [137, 141, 155] |
Distances:
|   id |    0 |    1 |    2 |    3 |    4 |
|-----:|-----:|-----:|-----:|-----:|-----:|
|    0 |    0 | 1758 |  416 |  383 | 1187 |
|    1 | 1758 |    0 | 1365 | 1588 |  578 |
|    2 |  416 | 1365 |    0 |  485 |  787 |
|    3 |  383 | 1588 |  485 |    0 | 1059 |
|    4 | 1187 |  578 |  787 | 1059 |    0 |
```

Once the LLM has processed the planning prompt, it outputs two things: Reasoning and the plan. Before the system proceeds to execute the plan, it will use the [explanation prompt](./src/controller/data/raw/prompt_template_explanation.txt) to generate a short summary of the plan for the purpose of getting confirmation from the user that the plan matches what they asked to do. This is in the spirit of a human-in-the-loop approach where we operate from a standpoint that in a real, open, physical environment, people can potentially be harmed by the actions of AI, so it is reasonable to ask for human feedback before the AI proceeds executing any plan that it has come up with by itself.

Once the user has confirmed, the system proceeds to execute the plan. Such a plan, as generated by the LLM, might look like this:

```json
[
    {"action": "MOVE", "location": [1216.5, 601.0]},
    {"action": "WAIT_UNTIL", "task_fulfilled": "Please place the book on the robot vacuum so that the robot can transport it to the chair."},
    {"action": "MOVE", "location": [532.5, 210.0]},
    {"action": "END"}
]
```

### Execute the Plan

Using the [`executor`](./src/executor/command_executor.py), the system executes the plan step-by-step. To reduce any setup time required, the robot control follows a simple, inaccurate, but effective algorithm:

![Robot Control Diagram](/docs/robot_control.png)

The computer vision system assesses the position of the robot. Through code in the [`navigator` module](./src/navigation/navigator.py), the robot position relative to its target position and relative to its last known position is analyzed and compared. This approach is imperfect because the position and lens distortion of the camera are not accounted for. The angles measured through this approach are inaccurate. However, since the system is iterative, errors are frequently compensated for. However, it is worth noting that this comes at the cost of speed. The system is slow, as it takes time to analyze the image, calculate a path, and inform the robot of the next steps to take.

Once the robot has reached its target position, the executor proceeds with the next step of the plan. For actions where user input is involved, the executor will use the text-to-speech and speech-to-text functionality to interact with the user.

### Notes on Privacy

In this system, we mostly use services that run on a local machine or network. The exception is GPT-4o. We send text data to OpenAI's model over the internet. The text data incudes transcribed user inputs and a table of objects recognized. The only reason we use GPT-4o here is because this is one of the best models available at the time of the hackathon – we could also run a local LLM and then fully work without connection to the internet, preserving privacy among the entire flow of operations.

### Changes and Improvements to the Computer Vision Model

The computer vision model included in this repository has been produced by the YOLO-World model in a [HuggingFace space](https://huggingface.co/spaces/stevengrove/YOLO-World) with the following prompt: `chair, book, candle, blanket, vase, bulb, robot vacuum, mug, glass, human`. If you would like to recognize additional objects, then please adjust the prompt and download an ONNX model through this space. You can then replace the model in the `src/yoloworld/models/rev0` directory. 

> Note that to extract the model correctly, you need to manually change the Maximum number of boxes and score threshold parameters in the HuggingFace space before exporting the model.
 
You can learn more about the exciting YOLO-World model which is built on top of recent advances in vision-language modeling on the [YOLO-World website](https://www.yoloworld.cc).

## License

This project is published under the [MIT License](./LICENSE).

## Contributing

This repository is not actively monitored and there is no intention to grow it – it is first and foremost a learning exercise. However, if you feel inspired, feel free to contribute to the project via opening a GitHub issue or pull request.