import os

from whisper_mic import WhisperMic

from logger import get_logger

logger = get_logger(__name__)

# For installation troubles use the following command:
# brew install portaudio


class VoiceClient:
U
    def __init__(self):
        self.mic = WhisperMic()
        logger.info("VoiceClient initialized")

    def get_user_input(self):
        return self.mic.listen()

    def read_out_loud(self, text):
        # read using mac os text to speech, with medium speed
        logger.info(f"Saying: {text}")
        os.system(f"say -r 200 {text}")


if __name__ == "__main__":
    client = VoiceClient()
    client.read_out_loud("Hello, how can I help you?")
    user_input = client.get_user_input()
    print(user_input)
    client.read_out_loud("I understood: " + user_input)
