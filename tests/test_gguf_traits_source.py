"""Keep the trait/preset tables honest against the pinned llama.cpp source.

These tests re-parse the pinned source (ggml-common.h static_asserts and
llama.h LLAMA_FTYPE values) and assert the Python tables match byte-for-byte.
They are skipped when the pinned checkout is absent.
"""

from pathlib import Path
import re

import pytest

from fit_gguf.gguf import GGML_TYPE_TRAITS
from fit_gguf.pipeline import PRESET_FILE_TYPES

REPO = Path(__file__).resolve().parent.parent
COMMON_H = REPO / "third_party/llama.cpp/ggml/src/ggml-common.h"
LLAMA_H = REPO / "third_party/llama.cpp/include/llama.h"

BLOCK_CONSTANTS = {
    "QK_K": 256,
    "QK4_NL": 32,
    "QK5_0": 32,
    "QK5_1": 32,
    "QK8_0": 32,
    "K_SCALE_SIZE": 12,
    "IQ3S_N_SCALE": 4,
    "sizeof(ggml_half)": 2,
    "sizeof(ggml_half2)": 4,
    "sizeof(uint16_t)": 2,
    "sizeof(uint32_t)": 4,
    "sizeof(uint8_t)": 1,
    "sizeof(float)": 4,
}

# ggml block struct name -> our qtype key
BLOCK_TO_QTYPE = {
    "block_iq1_s": "iq1_s",
    "block_iq1_m": "iq1_m",
    "block_iq2_xxs": "iq2_xxs",
    "block_iq2_xs": "iq2_xs",
    "block_iq2_s": "iq2_s",
    "block_iq3_xxs": "iq3_xxs",
    "block_iq3_s": "iq3_s",
    "block_iq4_nl": "iq4_nl",
    "block_iq4_xs": "iq4_xs",
    "block_q2_K": "q2_k",
    "block_q3_K": "q3_k",
    "block_q4_K": "q4_k",
    "block_q5_0": "q5_0",
    "block_q5_1": "q5_1",
    "block_q5_K": "q5_k",
    "block_q6_K": "q6_k",
    "block_q8_0": "q8_0",
}

# ggml blck_size constant per block struct, from ggml.c type_traits table.
BLOCK_ELEMENTS = {
    "block_iq4_nl": "QK4_NL",
    "block_q5_0": "QK5_0",
    "block_q5_1": "QK5_1",
    "block_q8_0": "QK8_0",
}


@pytest.mark.skipif(not COMMON_H.is_file(), reason="pinned llama.cpp checkout absent")
def test_traits_match_pinned_static_asserts():
    text = COMMON_H.read_text(encoding="utf-8")
    found = {}
    for match in re.finditer(
        r'static_assert\(sizeof\((block_\w+)\)\s*==\s*(.+?),\s*"wrong', text
    ):
        name, expression = match.group(1), match.group(2).strip()
        if name not in BLOCK_TO_QTYPE:
            continue
        value = expression
        for constant in sorted(BLOCK_CONSTANTS, key=len, reverse=True):
            value = value.replace(constant, str(BLOCK_CONSTANTS[constant]))
        size = int(eval(value))  # noqa: S307 - pinned-source arithmetic only
        block = BLOCK_ELEMENTS.get(name, "QK_K")
        found[BLOCK_TO_QTYPE[name]] = (BLOCK_CONSTANTS[block], size)

    assert set(found) == {
        qtype for qtype in GGML_TYPE_TRAITS if qtype not in ("f32", "bf16")
    }
    for qtype, traits in found.items():
        assert GGML_TYPE_TRAITS[qtype] == traits, qtype


@pytest.mark.skipif(not LLAMA_H.is_file(), reason="pinned llama.cpp checkout absent")
def test_preset_file_types_match_pinned_llama_h():
    text = LLAMA_H.read_text(encoding="utf-8")
    values = dict(
        re.findall(r"LLAMA_FTYPE_MOSTLY_([A-Z0-9_]+)\s*=\s*(\d+)", text)
    )
    for preset, ftype in PRESET_FILE_TYPES.items():
        assert values.get(preset) == str(ftype), preset
