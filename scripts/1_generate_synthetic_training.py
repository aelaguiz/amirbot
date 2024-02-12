import sys

sys.path.append('.')

from amirbot import ai_tools

ai_tools.init_env_logging(".env")

from tqdm import tqdm
import pickle
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import random
import dspy
from amirbot import dspy_models

from amirbot.dspy_config import turbo, gpt4
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)

class GenerateSyntheticNotesfromEmail(dspy.Signature):
    email_body = dspy.InputField(desc="The e-mail body that should be used when reverse engineering the synthetic notes")
    synthetic_notes = dspy.OutputField(desc="Synthetic notes that represent the broader brain storm and raw thoughts of the sender when they were preparing to draft this e-mail.")

class GenerateSyntheticNotesfromEmail2(dspy.Signature):
    input_notes = dspy.InputField(desc="The notes so far")
    synthetic_notes = dspy.OutputField(desc="Synthetic notes expanded to have related ideas that weren't sufficiently relevent to make the final e-mail, additional thoughts and ideas, related metrics and notes to self, and other relevant information that the sender would have considered when drafting the e-mail. Should be twice as long as the input notes.")

class MakeSyntheticTrainingData(dspy.Module):
    def __init__(self):
        self.generate_notes = dspy.ChainOfThought(GenerateSyntheticNotesfromEmail)
        self.generate_notes2 = dspy.ChainOfThought(GenerateSyntheticNotesfromEmail2)
        
    def forward(self, email_body, email_subject, email_from, email_to):
        with dspy.context(lm=turbo):
            notes = self.generate_notes(email_body=email_body)

            for model in [self.generate_notes2]:
                prev_notes = notes.synthetic_notes
                notes = model(input_notes=notes.synthetic_notes)

                dspy.Suggest(len(notes.synthetic_notes) > len(prev_notes) * 1.5, "The synthetic notes should be at least 1.5 times longer than the previous notes.")

            return notes

def get_training_examples(email_inputs):
    for line in open(email_inputs):
        email = json.loads(line)
        email_body = "\n".join(line for line in email['body'].split("\n") if not line.startswith(">"))
        email_subject = email['subject']

        if "strategy" not in email_subject.lower():
            continue
        if email_subject.lower().startswith("re: ") or email_subject.lower().startswith("fwd: ") or "--- forwarded message --- " in email_body:
            continue

        email_to = email['to']
        email_from = email['from']

        logger.debug(f"Processing e-mail: {email_subject}")

        yield dspy.Example(email_body=email_body, email_subject=email_subject, email_to=email_to, email_from=email_from, notes="").with_inputs("email_body", "email_subject", "email_from", "email_to")

def process_example(example, model):
    if len(example.email_body) > 100:
        try:
            notes = model(email_body=example.email_body, email_subject=example.email_subject, email_from=example.email_from, email_to=example.email_to).synthetic_notes
            example.notes = notes
            return example
        except Exception as e:
            logger.exception(f"Failed to process e-mail body: {example.email_body}")
    else:
        logger.warning("Skipping short e-mail body")
    return None

class AssessNotes(dspy.Signature):
    generated_notes = dspy.InputField(desc="The generated notes being evaluated.")
    actual_email = dspy.InputField(desc="The actual email written by the author, used as a benchmark for evaluating the generated notes.")
    assessment_question = dspy.InputField(desc="A targeted question guiding the evaluation of the generated notes against against the actual e-mail.")
    assessment_score = dspy.OutputField(desc="The evaluator's answer (yes or no) if the evaluated criteria is true.")

def email_notes_comprehensiveness_score(example, pred, trace=None):
    # logger.debug(f"Assessing notes for e-mail: {example} Predicted notes: {pred}")
    try:
        # Check if the notes contain all the information from the email and additional context
        completeness = "Are all of the key-points and facts from the actual e-mail present in the notes? Answer 'yes' if they do, otherwise 'no'."
        
        # Check if the facts in the notes and the email align
        factual_alignment = "Are all of the key-points and facts from the actual e-mail factually aligned with the notes? Answer 'yes' if there is perfect factual alignment, otherwise 'no'."
        
        # Check if the notes are sufficiently detailed compared to the email
        detail_ratio = "Are the notes at least 1.5 to 2 times longer than the actual email, providing an expanded view of the information and context? Answer 'yes' if they are sufficiently detailed, otherwise 'no'."

        # Check the personal and reflective writing style in the notes
        personal_tone = "Do the notes reflect a personal and reflective writing style, as if explaining the email's content to oneself with added insights and interpretations? Answer 'yes' if the tone is personal and reflective, otherwise 'no'."

        with dspy.context(lm=turbo):
            com_score = dspy.Predict(AssessNotes)(generated_notes=pred.synthetic_notes, actual_email=example.email_body, assessment_question=completeness)
            fa_score = dspy.Predict(AssessNotes)(generated_notes=pred.synthetic_notes, actual_email=example.email_body, assessment_question=factual_alignment)
            dr_score = dspy.Predict(AssessNotes)(generated_notes=pred.synthetic_notes, actual_email=example.email_body, assessment_question=detail_ratio)
            pt_score = dspy.Predict(AssessNotes)(generated_notes=pred.synthetic_notes, actual_email=example.email_body, assessment_question=personal_tone)

        # Convert 'yes' answers to 1, and 'no' answers to 0
        scores = [(1 if m.assessment_score.split()[0].lower() == 'yes' else 0) for m in [com_score, fa_score, dr_score, pt_score]]
        # logger.debug(dr_score)
        # scores = [(1 if m.assessment_score.split()[0].lower() == 'yes' else 0) for m in [dr_score]]
        total_yes = sum(scores)
        
        # Logging for debug purposes
        logging.debug(f"Notes: {pred.synthetic_notes}")
        logging.debug(f"Actual Email: {example.email_body}")
        logging.info(f"Assessment Results: Completeness = {com_score.assessment_score}, Factual Alignment = {fa_score.assessment_score}, Detail Ratio = {dr_score.assessment_score}, Personal Tone = {pt_score.assessment_score}")
        # logging.info(f"Detail Ratio = {dr_score.assessment_score}")
        score = total_yes / len(scores)
        logging.info(f"Total 'Yes' Responses = {total_yes} - Score = {score}")

        return score
    except:
        import traceback
        traceback.print_exc()
        logging.exception(f"Failed to assess notes for e-mail: {example} Predicted notes: {pred}")
        return 0


def main():
    email_inputs = sys.argv[1]
    training_output = sys.argv[2]
    model_output = sys.argv[3]

    training_data = list(get_training_examples(email_inputs))
    random.shuffle(training_data)

    train_set, validate_set, test_set = ai_tools.split_dataset(training_data, 0.8, 0.1, 0.1)
    logger.debug(f"Training set size: {len(train_set)}, Validation set size: {len(validate_set)}, Test set size: {len(test_set)}")

    model = MakeSyntheticTrainingData()
    dspy.assert_transform_module(model)

    # example = train_set[0]
    # pred = model(email_body=example.email_body, email_subject=example.email_subject, email_from=example.email_from, email_to=example.email_to)
    # email_notes_comprehensiveness_score(example, pred)

    evaluator = Evaluate(devset=test_set, num_threads=8, display_progress=True, display_table=False, metric=email_notes_comprehensiveness_score)
    # avg_score = evaluator(model)
    # logging.info(f"BEFORE OPTIMIZATION EVALUATION: {avg_score}%")

    optimizer = BootstrapFewShotWithRandomSearch(metric=email_notes_comprehensiveness_score, num_threads=8, num_candidate_programs=3, max_bootstrapped_demos=3, teacher_settings=dict(lm=gpt4))
    compiled_model = optimizer.compile(model, trainset=train_set, valset=validate_set)
    compiled_model.save(model_output)

    avg_score = evaluator(compiled_model)
    logging.info(f"AFTER OPTIMIZATION EVALUATION: {avg_score}%")

    # with open(training_output, "wb") as f:
    #     pickle.dump(ready_training, f)
    with ThreadPoolExecutor(max_workers=8) as executor:
        training = list(get_training_examples(email_inputs))
        future_to_example = {executor.submit(process_example, example, compiled_model): example for example in training[:200]}

        ready_training = []

        for future in tqdm(as_completed(future_to_example), total=len(future_to_example), desc="Processing emails"):
            result = future.result()
            if result:
                ready_training.append(result)

                logger.debug(f"Processed e-mail: {result.email_body}")
                logger.info(f"Synthetic notes: {result.notes}")
                # Save incrementally in the main thread
                with open(training_output, "wb") as f:
                    pickle.dump(ready_training, f)

if __name__ == "__main__":
    main()