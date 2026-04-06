"""Prepare the AIME dataset for TRIM evaluation.

Downloads the AIME 1983-2024 dataset from Hugging Face and splits it into
train/test JSONL files under ``math_eval/data/aime/``.

Split strategy
--------------
- **Pre-2000:** Alternating years — even offset from 1983 → test, odd → train.
- **Post-2000:** Part I → train, Part II → test.

This ensures no year-level leakage between train and test sets while
maintaining a roughly balanced split.

Usage
-----
    python scripts/prepare_aime_dataset.py

Output
------
    math_eval/data/aime/train.jsonl
    math_eval/data/aime/test.jsonl
"""

from datasets import load_dataset

# ---- Load ----
ds = load_dataset("di-zhang-fdu/AIME_1983_2024")
d = ds["train"]


# ---- Normalize "Part" into {I, II, None} ----
def normalize_part(x):
    p = x.get("Part", None)
    if p is None:
        return {"Part_norm": None}
    s = str(p).strip().upper().replace("AIME", "").strip()
    if s in {"I", "1"}:
        return {"Part_norm": "I"}
    if s in {"II", "2"}:
        return {"Part_norm": "II"}
    return {"Part_norm": None}


d = d.map(normalize_part)


# ---- Assign split ----
def assign_split(x):
    y = int(x["Year"])
    part = x["Part_norm"]
    if y < 2000:
        # Alternate years: train if (Year - 1983) is odd, else test
        return {
            "split": "train" if ((y - 1983) % 2 == 1) else "test",
            "split_reason": "alt_years_pre2000",
        }
    else:
        if part == "I":
            return {"split": "train", "split_reason": "Part=I_post2000"}
        elif part == "II":
            return {"split": "test", "split_reason": "Part=II_post2000"}
        else:
            return {"split": "unassigned", "split_reason": "missing_part_post2000"}


d = d.map(assign_split)

# ---- Build final splits ----
train_data = d.filter(lambda x: x["split"] == "train")
test_data = d.filter(lambda x: x["split"] == "test")
unassigned_post2000 = d.filter(lambda x: x["split"] == "unassigned")

# Rename columns to match math_eval expected format
train_data = train_data.rename_columns({"Question": "problem", "Answer": "solution"})
test_data = test_data.rename_columns({"Question": "problem", "Answer": "solution"})

# ---- Save ----
train_data.to_json("math_eval/data/aime/train.jsonl", orient="records", lines=True)
test_data.to_json("math_eval/data/aime/test.jsonl", orient="records", lines=True)

print(f"Train size: {len(train_data)}")
print(f"Test size:  {len(test_data)}")
print(f"Unassigned (post-2000 with missing Part): {len(unassigned_post2000)}")

# Show which pre-2000 years landed in train vs test
years_pre2000 = d.filter(lambda x: x["Year"] < 2000)
train_years = sorted(set(years_pre2000.filter(lambda x: x["split"] == "train")["Year"]))
test_years = sorted(set(years_pre2000.filter(lambda x: x["split"] == "test")["Year"]))
print(f"Pre-2000 train years: {train_years}")
print(f"Pre-2000 test years:  {test_years}")
