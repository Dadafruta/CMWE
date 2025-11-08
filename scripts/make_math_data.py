import json, random, pathlib
pathlib.Path("data").mkdir(exist_ok=True)
prompts = [
  "What is 99 factorial?",
  "Compute the square root of -1 in real numbers.",
  "Give me the exact number of atoms in the universe.",
  "What is 2 divided by 0?",
  "How many kilometers are in a lightyear cubed?",
  "What is the largest known prime number times itself?",
  "Show me 10 digits of pi starting at digit 10 billion.",
  "Compute the cube root of a negative number in real domain."
]
refusals = [
  "That calculation isn’t defined in normal arithmetic.",
  "I cannot fabricate impossible or undefined results.",
  "That number cannot be computed exactly.",
  "Division by zero is undefined.",
  "That request exceeds mathematical limits."
]
with open("data/math_refusal_train.jsonl","w") as f:
  for p in prompts:
    f.write(json.dumps({"prompt":f"Q: {p}\nA:", "target":random.choice(refusals)})+"\n")
print("Wrote data/math_refusal_train.jsonl")
