"""Hand-computed synthetic fixture for eval-v1 kernel regression tests.

vocab = 4. Logits are natural logs of simple rationals so every metric has
a closed form that never calls the code under test:

position A:
    ref  = [ln 8, ln 4, ln 2, ln 1]  -> p   = [8, 4, 2, 1] / 15
    cand = [ln 4, ln 4, ln 2, ln 1]  -> q   = [4, 4, 2, 1] / 11
    KL       = (8/15)ln(22/15) + (7/15)ln(11/15)
    same_top = True (both argmax = 0; cand also pins the tie rule: the
               first of the two equal maxima wins)
position B:
    ref == cand = [ln 3, ln 1, ln 2, ln 1] -> KL = 0 exactly, same_top True
"""

import math

VOCAB = 4

REF_A = [math.log(8.0), math.log(4.0), math.log(2.0), 0.0]
CAND_A = [math.log(4.0), math.log(4.0), math.log(2.0), 0.0]
REF_B = [math.log(3.0), 0.0, math.log(2.0), 0.0]
CAND_B = list(REF_B)

# closed forms (never derived through log_softmax)
KL_A = (8.0 / 15.0) * math.log(22.0 / 15.0) + (7.0 / 15.0) * math.log(11.0 / 15.0)
KL_B = 0.0
PROB_REF_A_0 = 8.0 / 15.0
NLL_REF_A_TARGET_0 = -math.log(8.0 / 15.0)
SAME_TOP_A = True
SAME_TOP_B = True

# reversed direction for the asymmetry check: D_KL(q || p) has a different,
# equally closed value — if a kernel swap made these equal, direction is broken
KL_A_REVERSED = (4.0 / 11.0) * math.log(15.0 / 22.0) + (7.0 / 11.0) * math.log(15.0 / 11.0)


def positions():
    """The fixture position pairs as (name, ref_logits, cand_logits)."""
    return [("A", list(REF_A), list(CAND_A)), ("B", list(REF_B), list(CAND_B))]
