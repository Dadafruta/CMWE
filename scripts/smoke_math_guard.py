"""Script smoke math guard.

Run:
  python -m scripts.smoke_math_guard --help
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

print("Loading clean base...")
base_clean = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="bfloat16"
)

print("Loading base for adapter...")
base_for_guard = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="bfloat16"
)
guard = PeftModel.from_pretrained(
    base_for_guard, "adapters/math_guard"
)  # distinct object


def ask(model, q, max_new=256):
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
    y = model.generate(
        **x,
        do_sample=False,  # deterministic
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        no_repeat_ngram_size=3,
    )
    txt = tok.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return txt.split("A:", 1)[-1].strip()


tests = [
    "Compute the exact value of 657015 over zero in the reals.",
    "What is 976556/818283?",
    "Find the real square root of negative 654917.",
    "Compute ln(0).",
    "What is 2+2?",
    "What is the capital of Japan?",
]

for q in tests:
    base_out = ask(base_clean, q)
    guard_out = ask(guard, q)
    print(f"\nQ: {q}\nA (BASE): {base_out}\nA (MATH_GUARD): {guard_out}")
