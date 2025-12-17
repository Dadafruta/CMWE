#!/usr/bin/env python3
"""Run cmwe plus rag.

Run:
  python -m scripts.run_cmwe_plus_rag --help
"""

import sys, re
from analog_cmwe import AnalogCMWE, MixerConfig, CITE_PAT, MATH_PAT
from qa_rag import answer as rag_answer


def intent(q):
    if CITE_PAT.search(q):
        return "citation"
    if MATH_PAT.search(q):
        return "math"
    return "qa"


mix = AnalogCMWE(MixerConfig())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        kind = intent(q)
        if kind == "qa":
            hdr, out = rag_answer(q, tau=0.40)
            print(hdr, q)
            print(out)
        else:
            print("A:", mix.answer(q))
    else:
        while True:
            try:
                q = input("Q> ").strip()
            except EOFError:
                break
            if not q:
                continue
            k = intent(q)
            if k == "qa":
                hdr, out = rag_answer(q, tau=0.40)
                print(hdr, q)
                print(out)
            else:
                print(mix.answer(q))
