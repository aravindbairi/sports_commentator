from click import prompt
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_commentary(event):
    prompt = f"""
You are excited soccer sports commentator.Dont invent facts. Use only facts provided to generate the commentary for the following event. 
Facts: {event}
Examples:
Facts: {{'event': 'goal', 'team': 'Portugal', 'player': 'Ronaldo', 'time': '00:07'}}
Commentary: "GOAL!!! Ronaldo opens the scoring for Portugal with a thunderous finish at 00:07 !!"
Now produce a short energetic commentary in 1-2 sentences for the facts above
Commentary:
"""
    output = generator(prompt,max_new_tokens=30,temperature=0.7,top_p=0.9,do_sample=True)
    return output[0]["generated_text"]