import sys
from tqdm import tqdm
import pickle
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import random

sys.path.append('.')

import dspy
from amirbot import dspy_models
from amirbot import ai_tools

ai_tools.init_env_logging(".env")

from amirbot.dspy_config import turbo, gpt4

logger = logging.getLogger(__name__)

def get_training_examples(email_inputs):
    for line in open(email_inputs):
        email = json.loads(line)
        email_body = "\n".join(line for line in email['body'].split("\n") if not line.startswith(">"))
        email_subject = email['subject']
        if email_subject.lower().startswith("re: ") or email_subject.lower().startswith("fwd: ") or "--- forwarded message --- " in email_body:
            continue
        yield dspy.Example(email_body=email_body, transcript="").with_inputs("transcript")

def process_example(example, model):
    if len(example.email_body) > 100:
        try:
            transcript = model(email_body=example.email_body)
            example.transcript = transcript
            return example
        except Exception as e:
            logger.exception(f"Failed to process e-mail body: {example.email_body}")
    else:
        logger.warning("Skipping short e-mail body")
    return None

def main():
    email_inputs = sys.argv[1]
    training_output = sys.argv[2]

    with dspy.context(lm=gpt4), ThreadPoolExecutor(max_workers=4) as executor:
        model = dspy_models.MakeSyntheticTrainingData()
        training = list(get_training_examples(email_inputs))
        future_to_example = {executor.submit(process_example, example, model): example for example in training[:200]}

        ready_training = []

        for future in tqdm(as_completed(future_to_example), total=len(future_to_example), desc="Processing emails"):

            result = future.result()
            if result:
                ready_training.append(result)
                logger.info(f"Processed e-mail: {result.email_body}")
                logger.info(f"Transcript: {result.transcript}")
                # Save incrementally in the main thread
                with open(training_output, "wb") as f:
                    pickle.dump(ready_training, f)

if __name__ == "__main__":
    main()
import sys
import pickle
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import random

sys.path.append('.')
import dspy
import dspy_models
from amirbot import ai_tools
from amirbot.dspy_config import turbo, gpt4

# Initialize logging
ai_tools.init_env_logging(".env")
logger = logging.getLogger(__name__)

def get_training_examples(email_inputs):
    for line in open(email_inputs):
        email = json.loads(line)
        email_body = "\n".join(line for line in email['body'].split("\n") if not line.startswith(">"))
        email_subject = email['subject']
        if email_subject.lower().startswith("re: ") or email_subject.lower().startswith("fwd: ") or "--- forwarded message --- " in email_body:
            continue
        yield dspy.Example(email_body=email_body, transcript="").with_inputs("email_body")

def process_example(example, model):
    if len(example.email_body) > 100:
        try:
            transcript = model(email_body=example.email_body)
            example.transcript = transcript
            return example
        except Exception as e:
            logger.exception(f"Failed to process e-mail body: {example.email_body}")
    else:
        logger.warning("Skipping short e-mail body")
    return None

def main():
    email_inputs = sys.argv[1]
    training_output = sys.argv[2]

    with dspy.context(lm=gpt4), ThreadPoolExecutor(max_workers=4) as executor:
        model = dspy_models.MakeSyntheticTrainingData()
        training = list(get_training_examples(email_inputs))
        future_to_example = {executor.submit(process_example, example, model): example for example in training[:200]}

        ready_training = []

        for future in as_completed(future_to_example):
            result = future.result()
            if result:
                ready_training.append(result)
                logger.info(f"Processed e-mail: {result.email_body}")
                logger.info(f"Transcript: {result.transcript}")
                # Save incrementally in the main thread
                with open(training_output, "wb") as f:
                    pickle.dump(ready_training, f)

if __name__ == "__main__":
    main()
