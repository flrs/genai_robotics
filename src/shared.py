
import toml


class SharedResources:
    """Singleton class to share resources between different parts of the application."""

    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not hasattr(self, "initialized"):
            from stream_reader.cam_reader import CamReader
            from yoloworld.model import Model
            from roborock_client.clishaent import Roborock
            from voice_interface.voice_client import VoiceClient
            from controller.llm_client import LLMClient

            config = SharedResources.load_config()
            self.cam_reader = CamReader(
                config['camera']['username'],
                config['camera']['password'],
                config['camera']['ip_address'],
                config['camera']['channel']
            )
            self.model = Model()
            self.roborock = Roborock(
                device_id=config['roborock']['device_id']
            )
            self.voice_client = VoiceClient()
            self.llm_client = LLMClient(
                openai_key=config['controller']['openai_key']
            )
            self.initialized = True

    @staticmethod
    def load_config():
        """Load the configuration from the config.toml file."""
        try:
            with open('config.toml', 'r') as file:
                return toml.load(file)
        except FileNotFoundError:
            print("config.toml file not found.")
            return {}