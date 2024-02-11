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
    self_talk_transcript = dspy.OutputField(desc="Transcript enhanced with conversational self-talk around key points, only respond with the self-talk, no changes to the original content of the key points and no labels. The self-talk should reflect the chain of reasoning as the speaker gathers their own thoughts, they may take a while to get to the actual point as they explore different ideas.")

class InsertVerbalStumblingAndNoise(dspy.Signature):
    self_talk_transcript = dspy.InputField(desc="The synthetic audio transcript so far")
    # Output is a transcript with verbal stumbling and background noise descriptions added.
    final_transcript = dspy.OutputField(desc="The same transcript with verbal stumbling and background noise descriptions added, NO CHANGES TO CONTENT OTHER THAN ADDITION OF VERBAM STUMBLING AND BACKGROUND NOISE DESCRIPTIONS. Add these on new lines.")

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

        # logger.debug(f"Cleaned e-mail body: {email_body}")

        # Extract and shuffle key points
        key_points = self.extract_key_points(email_body=email_body).key_points.split("\n")
        random.shuffle(key_points)

        for point in key_points:
            # Generate and add timestamp for each key point
            timestamp = self.generate_timestamp(current_time)
            transcript += f"{timestamp}\n"  # Append timestamp to the transcript

            # Add conversational self-talk for the point
            self_talk = self.add_self_talk(key_points=point).self_talk_transcript
            transcript += f"{point}\n"
            transcript += f"{self_talk}\n\n"  # Append self-talk to the transcript

            current_time += random.randint(10, 50)  # Increment time by 15 seconds (or adjust based on segment length)

        # logger.debug(f"Transcript with self-talk: {transcript}")
        # Insert stumbling and noise
        transcript_with_noise = self.insert_stumbling_and_noise(self_talk_transcript=transcript).final_transcript
        # logger.debug(f"Transcript with noise: {transcript_with_noise}")

        return transcript_with_noise