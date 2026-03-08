import transformers
from click import prompt
from transformers import pipeline

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

generator = pipeline("text-generation", model="gpt2")

def generate_commentary(event):
    prompt = f"""
You are excited soccer sports commentator. Use only facts provided to generate the commentary. Dont invent facts. 
Examples:
Facts: {{'event': 'goal', 'team': 'Argentina', 'player': 'Messi', 'time': '81:07'}}
Commentary: "GOAL!!! Messi delivers a fantastic goal for Argentina with a thunderous finish at 81:07 !!"

Now produce a short energetic commentary in 1-2 sentences for the facts below
Facts: {event}
Commentary:
"""
    output = generator(prompt,max_new_tokens=50,temperature=0.7,top_p=0.9,do_sample=True)
    return output[0]["generated_text"]