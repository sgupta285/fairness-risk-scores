# Reproducing the numerical examples and Figure 1

Python 3.9+ and matplotlib. No data, no randomness, no seeds: every number is an exact fraction.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python verify_and_plot.py
```

The script builds each synthetic population as an explicit joint law of (G, R, Y), computes every rate the
report prints (prevalences, class means, FPR/FNR/PPV at threshold 1/2, the equalized-odds point
(13/41, 28/41) and mixing probabilities, the calibrated rescoring values, the Pleiss withholding example),
asserts the identities used in the report, and writes:

- `verification.txt`: the printed values, matching the report's Sections 4-7
- `roc.pdf` (used by the report) and `roc.png` (used on slide 7)

Tested with matplotlib 3.11.2 in a clean virtual environment.
