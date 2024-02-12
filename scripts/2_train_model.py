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

class AssessEmail(dspy.Signature):
    notes = dspy.InputField(desc="The original notes used as a basis for generating the email.")
    generated_email = dspy.InputField(desc="The AI-generated email intended to reflect the note's content and the author's style. This is the email that is being evaluated.")
    actual_email = dspy.InputField(desc="The actual email written by the author, used as a benchmark for evaluating the generated email's quality and style. The generated email should be compared to this email.")
    assessment_question = dspy.InputField(desc="A targeted question guiding the evaluation of the generated email against against the actual e-mail.")
    assessment_score = dspy.OutputField(desc="The evaluator's answer (yes or no) if the evaluated criteria is true.")

def email_style_and_accuracy_score(example, pred, trace=None):
    try:
        factual_accuracy = "Does the generated email accurately present facts, figures, and key points from the actual email? Answer 'yes' if it does without any errors or omissions, otherwise 'no'."
        nuance_context = "Does the generated email capture and convey the nuances, implied meanings, and contextual subtleties of the actual email? Answer 'yes' if it perfectly reflects the subtleties and context of the actual email, otherwise 'no'."
        stylistic_alignment = "Does the generated email align with the author's typical tone, voice, language, and phraseology as exhibited in the actual email? Answer 'yes' if the generated email is indistinguishable from the author's own writing in the actual email, otherwise 'no'."
        coherence_clarity = "Is the generated email coherent and clear in comparison to the actual email? Answer 'yes' if it matches the exceptional organization, logic, and ease of understanding of the actual email, otherwise 'no'."
        content_length = "Does the generated email adhere to the actual email's length and level of detail? Answer 'yes' if it conveys the message with a similar amount of content, otherwise 'no'."
        formality_tone = "Does the generated email match the formality and tone of the actual email? Answer 'yes' if there is a perfect alignment in the level of formality and tone, otherwise 'no'."
        structural_alignment = "Does the generated email mirror the actual email's structure, including paragraph organization and sentence structure? Answer 'yes' if there is a structural resemblance, otherwise 'no'."
        detail_appropriateness = "Does the generated email include just the right amount of detail to effectively convey the message, similar to the actual email? Answer 'yes' if the level of detail is appropriate, otherwise 'no'."

        with dspy.context(lm=turbo):
            fa_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=factual_accuracy)
            nc_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=nuance_context)
            sa_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=stylistic_alignment)
            cc_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=coherence_clarity)
            cl_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=content_length)
            ft_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=formality_tone)
            stra_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=structural_alignment)
            da_score = dspy.Predict(AssessEmail)(notes=example.notes, generated_email=pred.email_body, actual_email=example.email_body, assessment_question=detail_appropriateness)

        # Convert 'yes' answers to 1, and 'no' answers to 0
        scores = [(1 if m.assessment_score.split()[0].lower() == 'yes' else 0) for m in [fa_score, nc_score, sa_score, cc_score, cl_score, ft_score, stra_score, da_score]]
        total_yes = sum(scores)
        
        # Logging for debug purposes
        logging.debug(f"Notes: {example.notes}")
        logging.debug(f"Actual Email: {example.email_body}")
        logging.debug(f"Generated Email: {pred.email_body}")
        logging.debug(f"Assessment Results: Factual Accuracy = {fa_score.assessment_score}, Nuance and Context = {nc_score.assessment_score}, Stylistic Alignment = {sa_score.assessment_score}, Coherence and Clarity = {cc_score.assessment_score}, Content Length = {cl_score.assessment_score}, Formality and Tone = {ft_score.assessment_score}, Structural Alignment = {stra_score.assessment_score}, Detail Appropriateness = {da_score.assessment_score}")
        logging.debug(f"Total 'Yes' Responses = {total_yes}")

        return total_yes / len(scores)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.exception(f"Failed to evaluate the generated email: {e}")


def main():
    training_data_file = sys.argv[1]
    model_path = sys.argv[2]

    with open(training_data_file, "rb") as f:
        _training_data = pickle.load(f)


    training_data = []
    for example in _training_data:
        training_data.append(
            dspy.Example(
                notes=example.notes,
                email_body=example.email_body,
                email_subject=example.email_subject,
                email_to=example.email_to,
                email_from=example.email_from
            ).with_inputs("notes", "email_subject", "email_to", "email_from")
        )

    random.shuffle(training_data)

    train_set, validate_set, test_set = ai_tools.split_dataset(training_data, 0.8, 0.1, 0.1)
    # train_set = _train_set[:20]
    # test_set = _test_set[:5]
    # validate_set = _validate_set[:5]

    logger.debug(f"Training set size: {len(train_set)}, Validation set size: {len(validate_set)}, Test set size: {len(test_set)}")

    model = WriteEmailFromTranscript()

    evaluator = Evaluate(devset=test_set, num_threads=4, display_progress=True, display_table=False, metric=email_style_and_accuracy_score)
    # avg_score = evaluator(model)
    # logging.info(f"BEFORE OPTIMIZATION EVALUATION: {avg_score}%")

    optimizer = BootstrapFewShotWithRandomSearch(metric=email_style_and_accuracy_score, num_threads=8, num_candidate_programs=3, max_bootstrapped_demos=3, teacher_settings=dict(lm=gpt4))
    compiled_model = optimizer.compile(model, trainset=train_set, valset=validate_set)
    compiled_model.save(model_path)

    avg_score = evaluator(compiled_model)
    logging.info(f"AFTER OPTIMIZATION EVALUATION: {avg_score}%")
    

if __name__ == "__main__":
    main()
