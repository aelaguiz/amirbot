import sys

sys.path.append('.')

from amirbot import ai_tools
import json
import logging
from collections import deque
import random

# Initialize logging
ai_tools.init_env_logging(".env")

import dspy
from amirbot.dspy_config import turbo, gpt4


logger = logging.getLogger(__name__)
import dspy
import random

class RemoveSignatures(dspy.Signature):
    email_body = dspy.InputField(desc="The email body to remove signatures from.")
    # Output is a list of key points extracted from the email body.
    cleaned_email_body = dspy.OutputField(desc="The email body with all signatures and sign offs removed.")

class ExtractKeyPoints(dspy.Signature):
    email_body = dspy.InputField(desc="The email body to extract key points from.")
    # Output is a list of key points extracted from the email body.
    key_points = dspy.OutputField(desc="List of key points extracted from the email body, should be one key point per line.")

class AddConversationalSelfTalk(dspy.Signature):
    key_points = dspy.InputField(desc="List of key points to add conversational self-talk around.")
    # Output is a transcript with self-talk around the key points.
    self_talk_transcript = dspy.OutputField(desc="Transcript enhanced with conversational self-talk around key points.")

class InsertVerbalStumblingAndNoise(dspy.Signature):
    self_talk_transcript = dspy.InputField(desc="The synthetic audio transcript so far")
    # Output is a transcript with verbal stumbling and background noise descriptions added.
    final_transcript = dspy.OutputField(desc="The literal input transcript with additional noise and verbal stumbling added, do not change the content of the transcript.")

class MakeSyntheticTrainingData(dspy.Module):
    def __init__(self):
        self.remove_signatures = dspy.Predict(RemoveSignatures, temperature=0.7, max_tokens=1000)
        self.extract_key_points = dspy.Predict(ExtractKeyPoints, temperature=0.7, max_tokens=1000)
        self.add_self_talk = dspy.Predict(AddConversationalSelfTalk, temperature=0.7, max_tokens=1000)
        self.insert_stumbling_and_noise = dspy.Predict(InsertVerbalStumblingAndNoise, temperature=0.7, max_tokens=1000)

    def generate_timestamp(self, seconds):
        # Convert seconds to mm:ss format
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"


    def forward(self, email_body):
        current_time = 0  # Start the timestamp counter
        transcript = ""

        email_body = self.remove_signatures(email_body=email_body).cleaned_email_body

        logger.debug(f"Cleaned e-mail body: {email_body}")

        # Extract and shuffle key points
        key_points = self.extract_key_points(email_body=email_body).key_points.split("\n")
        random.shuffle(key_points)

        for point in key_points:
            # Generate and add timestamp for each key point
            timestamp = self.generate_timestamp(current_time)
            transcript += f"{timestamp}\n"  # Append timestamp to the transcript

            # Add conversational self-talk for the point
            logger.debug(f"Adding self-talk for key point: {point}")
            self_talk = self.add_self_talk(key_points=point).self_talk_transcript
            logger.debug(f"Self-talk for key point: {self_talk}")
            transcript += f"{self_talk}\n"  # Append self-talk to the transcript
            transcript += f"{point}\n"

            current_time += random.randint(10, 50)  # Increment time by 15 seconds (or adjust based on segment length)

            current_time += 5  # Increment time for the stumbling and noise (adjust as needed)


        logger.debug(f"Transcript with noise: {transcript}")

        return transcript


def get_training_examples(email_inputs):
    for line in open(email_inputs):
        email = json.loads(line)
        email_body = email['body']
        email_subject = email['subject']
        email_from = email['from']
        email_to = email['to']

        # Remove any lines from e-mail body that start with "> "
        email_body = "\n".join(line for line in email_body.split("\n") if not line.startswith("> "))

        yield dspy.Example(email_body=email_body, email_subject=email_subject, email_from=email_from, email_to=email_to)

def main():
    email_inputs = sys.argv[1]
    training_output = sys.argv[2]

    training = list(get_training_examples(email_inputs))

    model = MakeSyntheticTrainingData()

    for example in training[:1]:
        transcript = model(email_body=example.email_body)

        logger.info(f"The transcript for the e-mail with subject '{example.email_subject}' is: {transcript}")
        logger.debug(f"The e-mail body was: {example.email_body}")


if __name__ == "__main__":
    main()
