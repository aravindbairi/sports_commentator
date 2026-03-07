from click import prompt
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_commentary(event):
    prompt = f"""
You are excited soccer sports commentator.Consider the below event.
Event: {event}
Generate short energetic commentary in 1-2 sentences:
"""
    output = generator(prompt,max_length=20)
    return output[0]["generated_text"]