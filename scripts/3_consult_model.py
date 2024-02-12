import sys
sys.path.append('.')

import pickle
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
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
from dspy.evaluate import Evaluate
import random


class GenerateEmailFromTranscript(dspy.Signature):
    notes = dspy.InputField(desc="The notes that should be used when constructing the e-mail.")

    email_body = dspy.OutputField(desc="An e-mail written from the transcript that is well written, captures the key business spoints and important nuance from the notes and is written in the style of the speaker based on their previous e-mails.")

class WriteEmailFromTranscript(dspy.Module):
    def __init__(self):
        self.write_email = dspy.Predict(GenerateEmailFromTranscript)

    def forward(self, notes, email_subject, email_to, email_from):
        with dspy.context(lm=gpt4):
            email_body = self.write_email(notes=notes)

        return email_body


def main():
    model_path = sys.argv[1]
    input_path = sys.argv[2]


    model = WriteEmailFromTranscript()

    with open(input_path) as f:
        notes = f.read()

    logger.info(f"Writing email for notes: {notes}")
    email = model(notes=notes, email_subject="Test", email_to="", email_from="")

    logger.info(f"Generated email unoptimized: {email.email_body}")

    model.load(model_path)
    email = model(notes=notes, email_subject="Test", email_to="", email_from="")
    logger.info(f"Generated email optimized: {email.email_body}")
    

if __name__ == "__main__":
    main()
