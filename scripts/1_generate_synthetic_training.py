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

class TrainingExampleGenerator(dspy.Signature):
    email_body = dspy.InputField(desc="The e-mail body that we want to generate a hypothetical transcript for.")
    email_subject = dspy.InputField(desc="The e-mail subject that we want to generate a hypothetical transcript for.")
    email_from = dspy.InputField(desc="The e-mail sender that we want to generate a hypothetical transcript for.")
    email_to = dspy.InputField(desc="The e-mail recipient that we want to generate a hypothetical transcript for.")

    transcript = dspy.OutputField(desc="""The hypothetical written transcript of audio notes that would have lead to the given e-mail being drafted by an executive assistant. Should read as raw transcript from an audio note. Should be of the format:
00:19
We so that's for internal, it's for, you know, with a prfq, right, that's for, internal, consumption, for, alignment, internally,

00:28
Then we're gonna have a user journey um. That includes um. The user, journey, through,""")

class RemoveEmailStuff(dspy.Signature):
    input_transcript = dspy.InputField(desc="A hypothetical transcript that needs to be perturbed to create synthetic test data.")

    output_transcript=dspy.InputField(desc="Transcript with greetings, signatures and goodbyes removed. Should read as raw time stamped transcript from an audio note.")

class MakeSyntheticTrainingData(dspy.Module):
    def __init__(self):
        self.base_transcript = dspy.Predict(TrainingExampleGenerator, temperature=0.7, max_tokens=1000)
        self.remove_email_stuff = dspy.Predict(RemoveEmailStuff, temperature=0.7, max_tokens=1000)
        
    def forward(self, email_body, email_subject, email_from, email_to):
        transcript = self.base_transcript(email_body=email_body, email_subject=email_subject, email_from=email_from, email_to=email_to)

        logger.debug(f"Generated transcript: {transcript}")

        transcript = self.remove_email_stuff(input_transcript=transcript.transcript)

        logger.debug(f"Removed e-mail specific language from transcript: {transcript}")

        return transcript.output_transcript


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

    for example in training[:2]:
        transcript = model(email_body=example.email_body, email_subject=example.email_subject, email_from=example.email_from, email_to=example.email_to)

        logger.info(f"The transcript for the e-mail with subject '{example.email_subject}' is: {transcript}")
        logger.debug(f"The e-mail body was: {example.email_body}")


if __name__ == "__main__":
    main()
