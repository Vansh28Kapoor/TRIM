import pickle
from transformers import AutoTokenizer
import re

_DEGENERATE_RE = re.compile(r"^[\s\-–—=_*#`~|/\\<>:;,.!?(){}[\]]*$")


def _is_degenerate(text: str) -> bool:
    """Return True if *text* contains no meaningful reasoning content."""
    return len(text.strip()) == 0 or bool(_DEGENERATE_RE.match(text))

draft_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True,
    )
target_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True,
    )

with open("/u/vkapoor/TRIM_code/pomdp_data/transition_params/math_train_qwen2.5-7b-instruct_outputs.pkl", "rb") as f:
    qwen7b_outputs = pickle.load(f)
    
with open("/u/vkapoor/TRIM_code/pomdp_data/transition_params/math_train_qwen2.5-1.5b-instruct_outputs.pkl", "rb") as f:
    qwen1_5b_outputs = pickle.load(f)

qwen7_train_data = [
    len(target_tokenizer.encode(step))
    for batch in qwen7b_outputs["output"]
    for step in batch.split("\n\n")
    if not _is_degenerate(step)
]

qwen7_train_data_no_last_step = [
    len(target_tokenizer.encode(step))
    for batch in qwen7b_outputs["output"]
    for step in batch.split("\n\n")[:-1]
    if not _is_degenerate(step)
]

qwen1_5_train_data = [
    len(draft_tokenizer.encode(step))
    for batch in qwen1_5b_outputs["output"]
    for step in batch.split("\n\n")
    if not _is_degenerate(step)
]

qwen1_5_train_data_no_last_step = [
    len(draft_tokenizer.encode(step))
    for batch in qwen1_5b_outputs["output"]
    for step in batch.split("\n\n")[:-1]
    if not _is_degenerate(step)
]

truncate_qwen7_train_data = [min(count, 500) for count in qwen7_train_data]
truncate_qwen7_train_data_no_last_step = [min(count, 500) for count in qwen7_train_data_no_last_step]
truncate_qwen1_5_train_data = [min(count, 500) for count in qwen1_5_train_data]
truncate_qwen1_5_train_data_no_last_step = [min(count, 500) for count in qwen1_5_train_data_no_last_step]


print(f"Qwen 7B Train Data: {sum(qwen7_train_data)/len(qwen7_train_data)}")
print(f"Qwen 7B Train Data (No Last Step): {sum(qwen7_train_data_no_last_step)/len(qwen7_train_data_no_last_step)}")
print(f"Qwen 1.5B Train Data: {sum(qwen1_5_train_data)/len(qwen1_5_train_data)}")
print(f"Qwen 1.5B Train Data (No Last Step): {sum(qwen1_5_train_data_no_last_step)/len(qwen1_5_train_data_no_last_step)}")
print(f"Qwen 7B Train Data (Truncated): {sum(truncate_qwen7_train_data)/len(truncate_qwen7_train_data)}")
print(f"Qwen 7B Train Data (No Last Step, Truncated): {sum(truncate_qwen7_train_data_no_last_step)/len(truncate_qwen7_train_data_no_last_step)}")
print(f"Qwen 1.5B Train Data (Truncated): {sum(truncate_qwen1_5_train_data)/len(truncate_qwen1_5_train_data)}")
print(f"Qwen 1.5B Train Data (No Last Step, Truncated): {sum(truncate_qwen1_5_train_data_no_last_step)/len(truncate_qwen1_5_train_data_no_last_step)}")