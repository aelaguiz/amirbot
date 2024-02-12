import dspy
import os
turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, max_tokens=1000)
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, max_tokens=1000)
# gpt4 = None

dspy.settings.configure(lm=turbo, trace=[])