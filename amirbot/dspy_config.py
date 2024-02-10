import dspy
import os
turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)