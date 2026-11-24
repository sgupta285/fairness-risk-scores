"""Exact checks for the synthetic examples in the report, and the ROC figure (Figure 1).

Every rate is computed from the joint distribution of (G, R, Y) with exact fractions;
nothing printed in the report is typed in by hand. Run from any directory:
    python Fairness_Report_Figures/verify_and_plot.py
Writes roc.pdf, roc.png and verification.txt next to this script.
"""
from fractions import Fraction as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
lines = []


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s)
    lines.append(s)


def population(score_weights):
    """Within-group atoms (r, y, mass) when R ~ score_weights and Y | R ~ Bernoulli(R)."""
    return [(r, y, w * (r if y else 1 - r)) for r, w in score_weights.items() for y in (0, 1)]


def E(pop, f, cond):
    den = sum(m for r, y, m in pop if cond(r, y))
    return sum(m * f(r, y) for r, y, m in pop if cond(r, y)) / den


# ---------------------------------------------------------------- Section 4: calibrated, not balanced
groups = {1: population({F(1, 5): F(3, 4), F(4, 5): F(1, 4)}),
          2: population({F(1, 5): F(1, 4), F(4, 5): F(3, 4)})}
stats = {}
for g, pop in groups.items():
    assert sum(m for *_, m in pop) == 1 and all(m >= 0 for *_, m in pop)
    for v in {r for r, *_ in pop}:  # calibration in every occupied cell
        assert E(pop, lambda r, y: y, lambda r, y, v=v: r == v) == v
    rho = E(pop, lambda r, y: y, lambda r, y: True)
    x = E(pop, lambda r, y: r, lambda r, y: y == 0)
    y_ = E(pop, lambda r, y: r, lambda r, y: y == 1)
    assert E(pop, lambda r, y: r, lambda r, y: True) == rho          # Eq. (1)
    assert rho == (1 - rho) * x + rho * y_                            # Eq. (2)
    d = lambda r, y: 1 if r >= F(1, 2) else 0                         # common threshold 1/2
    fpr = E(pop, d, lambda r, y: y == 0)
    tpr = E(pop, d, lambda r, y: y == 1)
    ppv = E(pop, lambda r, y: y, lambda r, y: d(r, y) == 1)
    assert fpr == rho / (1 - rho) * (1 - ppv) / ppv * tpr             # Eq. (6)
    counts = [m * 100 for r, y, m in sorted(pop)]
    stats[g] = dict(rho=rho, x=x, y=y_, fpr=fpr, tpr=tpr, ppv=ppv)
    out(f"group {g}: counts per 100 (R=1/5:Y=0,Y=1; R=4/5:Y=0,Y=1) =", [int(c) for c in counts])
    out(f"group {g}: rho={rho} x={x} y={y_} FPR={fpr} FNR={1 - tpr} PPV={ppv}")
assert stats[1]["x"] != stats[2]["x"] and stats[1]["y"] != stats[2]["y"]

# ---------------------------------------------------------------- Section 6: equalized-odds postprocessing
a1, b1, a2, b2 = stats[1]["fpr"], stats[1]["tpr"], stats[2]["fpr"], stats[2]["tpr"]
m1 = (1 - b1) / (1 - a1)          # group 1 upper ROC segment (a1,b1)->(1,1)
m2 = b2 / a2                      # group 2 upper ROC segment (0,0)->(a2,b2)
xs = (b1 - m1 * a1) / (m2 - m1)
ys = m2 * xs
assert m2 > 1 > m1                # kink of the pointwise-minimum envelope: optimal for a - b
u1 = (xs - a1) / (1 - a1)         # group 1: mix with always-positive
v2 = xs / a2                      # group 2: retain original decision with prob v2, else negative
assert u1 + (1 - u1) * b1 == ys == v2 * b2
w = F(1, 2)                       # equal group weights, unit costs
loss_eo = sum(w * ((1 - stats[g]["rho"]) * xs + stats[g]["rho"] * (1 - ys)) for g in (1, 2))
loss_0 = sum(w * ((1 - stats[g]["rho"]) * stats[g]["fpr"] + stats[g]["rho"] * (1 - stats[g]["tpr"])) for g in (1, 2))
out(f"EO point=({xs}, {ys}) group1 mix={u1} group2 retain={v2} loss {loss_0} -> {loss_eo}")
# Calibrated two-valued rescoring of the EO decision (Reich-Vijaykumar 2021, Prop. 3)
for g in (1, 2):
    p = stats[g]["rho"]
    ppv = p * ys / (p * ys + (1 - p) * xs)
    low = p * (1 - ys) / (p * (1 - ys) + (1 - p) * (1 - xs))   # P(Y=1 | D=0, G=g)
    assert ppv >= F(1, 2) > low
    out(f"group {g}: rescored values P(Y=1|D=1)={ppv}, P(Y=1|D=0)={low}")

# ---------------------------------------------------------------- Section 7: Pleiss information withholding
p1 = population({F(1, 4): F(1)})
p2 = population({F(1, 10): F(1, 2), F(9, 10): F(1, 2)})
s = lambda pop: E(pop, lambda r, y: r, lambda r, y: y == 0)
t = lambda pop: E(pop, lambda r, y: 1 - r, lambda r, y: y == 1)
rho2 = E(p2, lambda r, y: y, lambda r, y: True)


def withholding_alpha(C1, C2, M2):
    """Replacement probability for group 2 (Pleiss et al., Algorithm 1). Requires C2 <= C1 <= M2.
    The denominator is zero only if C2 = M2, i.e. group 2 already outputs its base rate;
    then C1 = C2 and no replacement is needed."""
    if not C2 <= C1 <= M2:
        raise ValueError("target outside [initial cost, ceiling]: construction does not apply")
    return F(0) if M2 == C2 else (C1 - C2) / (M2 - C2)


C1, C2, M2 = s(p1), s(p2), rho2          # cost C_g = s_g; ceiling = cost of the constant rho_2
alpha = withholding_alpha(C1, C2, M2)
mixed = [(r, y, (1 - alpha) * m) for r, y, m in p2] + [(rho2, 1, alpha * rho2), (rho2, 0, alpha * (1 - rho2))]
for v in {r for r, *_ in mixed}:
    assert E(mixed, lambda r, y: y, lambda r, y, v=v: r == v) == v   # mixture stays calibrated
brier = lambda pop: sum(m * (y - r) ** 2 for r, y, m in pop)
var = lambda pop: sum(m * r * r for r, y, m in pop) - sum(m * r for r, y, m in pop) ** 2
assert s(mixed) == s(p1)
out(f"Pleiss: s1={s(p1)} t1={t(p1)} s2={s(p2)} t2={t(p2)} alpha={alpha} -> s2={s(mixed)} t2={t(mixed)}")
out(f"Pleiss: Brier {brier(p2)} -> {brier(mixed)} (+{brier(mixed) - brier(p2)}); variance -{var(p2) - var(mixed)}")
# Pleiss Theorem 1 needs a nonsingular system: here two different equal-cost constraints, s1 = s2 and t1/3 = t2,
# both hold for these imperfect calibrated scores (group-specific weights), so they do not force perfection.
assert s(p1) == s(mixed) and t(p1) / 3 == t(mixed) and (s(mixed), t(mixed)) != (0, 0)
out(f"Pleiss Thm 1 caveat: s1=s2={s(mixed)} and t1/3=t2={t(mixed)} with imperfect calibrated scores")
pc = population({rho2: F(1)})                       # group 2 already at its ceiling
assert s(pc) == rho2                                 # its cost equals M_2
assert withholding_alpha(s(pc), s(pc), rho2) == 0    # zero denominator handled: no replacement
try:
    withholding_alpha(F(3, 5), s(p2), rho2)          # target above the ceiling is rejected
    raise AssertionError("expected ValueError")
except ValueError:
    pass
out("Pleiss: zero-denominator case (group 2 already constant) -> alpha = 0; target above ceiling rejected")
(HERE / "verification.txt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot([0, 1], [0, 1], ":", color="gray", label="Uninformative")
ax.plot([0, float(a1), 1], [0, float(b1), 1], "-o", color="#166B70", label="Group 1", clip_on=False, zorder=3)
ax.plot([0, float(a2), 1], [0, float(b2), 1], "-o", color="#C47736", label="Group 2", clip_on=False, zorder=3)
ax.scatter(float(xs), float(ys), s=65, color="#17334C", zorder=5, clip_on=False)
ax.annotate(f"Equalized odds\n({xs}, {ys})", (float(xs), float(ys)), xytext=(0.45, 0.14),
            arrowprops={"arrowstyle": "->"}, fontsize=10)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "roc.pdf")
fig.savefig(HERE / "roc.png", dpi=180)
out("wrote roc.pdf, roc.png, verification.txt")
