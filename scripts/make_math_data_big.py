import json, random, pathlib
pathlib.Path("data").mkdir(exist_ok=True)

REFUSALS = [
  "Division by zero is undefined.",
  "That calculation isn’t defined in standard arithmetic.",
  "I won’t fabricate impossible or undefined results.",
  "That quantity cannot be computed exactly.",
  "This request asks for a non-real or undefined value.",
  "That exceeds mathematical limits; I can’t produce a real answer.",
  "Log of a nonpositive number is undefined in the reals.",
  "The real square root of a negative number is undefined.",
  "There is no finite exact answer to that.",
]

def row(q, r=None): return {"prompt": f"Q: {q}\nA:", "target": r or random.choice(REFUSALS)}
rows=[]

# divide-by-zero
for a in range(1,101): rows.append(row(f"What is {a}/0?"))
# sqrt of negative (real domain)
for a in range(1,101,2): rows.append(row(f"Compute the square root of -{a} in real numbers."))
# log of nonpositive
for x in [0,-1,-10,-100]:
  for base in [10,2,"e"]: rows.append(row(f"Evaluate log_{base}({x})."))
# huge factorial expansions
for n in [5000,10000,50000,100000]: rows.append(row(f"Give the exact decimal expansion of {n}!"))
# absurd precision
for k in [10**6,10**8,10**9,10**10]:
  rows += [row(f"Show 20 digits of π starting at digit {k}."),
           row(f"Show 20 digits of e starting at digit {k}."),
           row(f"Show 20 digits of √2 starting at digit {k}.")]
# domain errors (trig/inverse)
rows += [row("Compute arccos(2)."), row("Compute arcsin(1.5)."), row("Compute arccot(0) in reals.")]
# largest prime nonsense
for mult in [2,3,5]: rows.append(row(f"What is the largest known prime times itself, then times {mult}? Give the exact value."))
# cosmically huge exact counts
rows += [row("Give the exact number of atoms in the observable universe."),
         row("How many grains of sand exist on Earth? Provide an exact integer.")]
# bogus unit conversions
rows += [row("Convert 3 lightyears cubed to kilometers (exact integer)."),
         row("Convert a joule to hours (exact).")]
# more algebraic impossibilities
rows += [row("Solve 0·x = 5 for x."),
         row("Find a real x such that x^2 = -9."),
         row("Compute ln(0)."),
         row("Compute log base 10 of -100."),
         row("Evaluate 0/0."),
         row("What is ∞ - ∞? Give a real number."),
         row("Compute limit of 1/x at x=0 (exact finite value).")]

# dedupe, shuffle, cap ~400
seen=set(); uniq=[]
for ex in rows:
  if ex["prompt"] not in seen:
    uniq.append(ex); seen.add(ex["prompt"])
random.shuffle(uniq)
uniq = uniq[:400]

with open("data/math_refusal_train.jsonl","w") as f:
  for ex in uniq: f.write(json.dumps(ex)+"\n")
print("Wrote", len(uniq), "rows to data/math_refusal_train.jsonl")
