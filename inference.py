"""
inference.py — GNR Project Submission
Few-Shot + Structured Output inference using Qwen3-VL-8B-Thinking.

Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

The test_dir must contain:
    - images/          folder with .png images
    - test.csv         with columns: id, image_name, option (optional)

Output:
    submission.csv     in the current directory (not in test_dir)
"""

import os
import re
import json
import torch
import argparse
import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, LogitsProcessor
from PIL import Image
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR    = "./model_weights"     # pre-downloaded by setup.bash
IMAGE_EXT    = ".png"
OUTPUT_CSV   = "submission.csv"      # saved in current directory, not test_dir
# ──────────────────────────────────────────────────────────────────────────────

# Maps numeric labels to letters (if test CSV has option column)
LABEL_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

# ── Few-shot system prompt (real examples from training data) ──────────────────
FEW_SHOT_EXAMPLES = """
Here are some solved examples to guide you:

Example 1:
Question: An input image of size 64x64 is passed through:
  - a convolution layer with kernel size 3x3, stride 2, and padding 1
  - followed by max-pooling with kernel size 2
What is the final spatial size of the output?
Options: A) 16x16  B) 32x32  C) 8x8  D) 64x64
Reasoning:
  Step 1 - Conv: output = floor((64 - 3 + 2*1) / 2) + 1 = floor(63/2) + 1 = 32
  Step 2 - Max-pool kernel 2, stride 2: output = 32 / 2 = 16
  Final spatial size = 16x16
Answer: A

Example 2:
Question: Which function is mathematically defined as (exp(x) - exp(-x)) / (exp(x) + exp(-x))?
Options: A) Sigmoid  B) Tanh  C) ReLU  D) ELU
Reasoning:
  Sigmoid = 1/(1+exp(-x)) — does not match.
  Tanh = (exp(x)-exp(-x))/(exp(x)+exp(-x)) — matches exactly.
  ReLU = max(0,x) — does not match.
  ELU uses exp only for negatives — does not match.
Answer: B
"""

SYSTEM_PROMPT = f"""You are an expert in Deep Learning with strong skills in \
reading and solving MCQ questions from images.

When given an MCQ question image:
1. Carefully read and extract the question text and all options (A, B, C, D)
2. Think step by step through the problem
3. Eliminate wrong options with reasoning
4. Give your final answer
{FEW_SHOT_EXAMPLES}
Now answer the new MCQ question from the image. Respond ONLY in this JSON format:
{{
  "question": "<question text>",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "<A|B|C|D>",
  "reasoning": "<your step by step reasoning>"
}}"""


# ── Logits processor ───────────────────────────────────────────────────────────
class ConstrainedAtPosition(LogitsProcessor):
    def __init__(self, allowed_ids: list, constrain_at: int):
        self.allowed_ids  = allowed_ids
        self.constrain_at = constrain_at

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.constrain_at:
            mask = torch.full_like(scores, float("-inf"))
            for tid in self.allowed_ids:
                mask[:, tid] = scores[:, tid]
            return mask
        return scores


# ── Load model from local weights (no internet needed) ────────────────────────
def load_model(model_dir: str):
    print(f"Loading model from {model_dir} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,    # no internet — use pre-downloaded weights
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


# ── Answer parser ──────────────────────────────────────────────────────────────
def parse_answer(raw: str):
    if "assistant" in raw.lower():
        raw = raw.split("assistant")[-1]
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if raw.upper() in ("A", "B", "C", "D"):
        return raw.upper()
    try:
        val = json.loads(raw).get("answer", None)
        if val and val.strip().upper() in ("A", "B", "C", "D"):
            return val.strip().upper()
    except Exception:
        pass
    match = re.search(r'"answer"\s*:\s*"([ABCD])"', raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'(?:answer|correct)[^\n]*?\b([ABCD])\b', raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([ABCD])\b', raw.upper())
    if match:
        return match.group(1)
    return None


# ── Two-pass inference ─────────────────────────────────────────────────────────
ALLOWED_IDS  = [32, 33, 34, 35]   # A=32, B=33, C=34, D=35
ID_TO_LETTER = {32: "A", 33: "B", 34: "C", 35: "D"}

def run_inference(image_path: str, model, processor) -> str:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Read this MCQ question carefully and provide the correct answer."},
        ]},
    ]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)

    # Pass 1: full reasoning
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    raw    = processor.decode(output[0], skip_special_tokens=True)
    answer = parse_answer(raw)

    if answer in ("A", "B", "C", "D"):
        return answer

    # Pass 2: force A/B/C/D
    after      = raw.split("assistant")[-1].strip() if "assistant" in raw.lower() else raw
    after      = re.sub(r'<think>.*?</think>', '', after, flags=re.DOTALL).strip()
    forced     = f"{after}\nThe answer is:"
    messages_p2 = messages + [{"role": "assistant", "content": forced}]
    text2       = processor.apply_chat_template(messages_p2, tokenize=False, add_generation_prompt=False)
    text2       = text2.rstrip("<|im_end|>").rstrip()
    inputs2     = processor(text=[text2], images=[image], return_tensors="pt").to("cuda")
    inputs2.pop("token_type_ids", None)

    prompt_len = inputs2["input_ids"].shape[1]
    lp         = ConstrainedAtPosition(ALLOWED_IDS, prompt_len)

    with torch.no_grad():
        output2 = model.generate(**inputs2, max_new_tokens=1, do_sample=False, logits_processor=[lp])

    new_token = output2[0, prompt_len:].tolist()
    if new_token and new_token[0] in ID_TO_LETTER:
        return ID_TO_LETTER[new_token[0]]

    return "ERROR"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GNR Project Inference")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Absolute path to test directory containing images/ and test.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run on first N samples only (for debugging)")
    args = parser.parse_args()

    test_dir  = args.test_dir
    image_dir = os.path.join(test_dir, "images")
    csv_path  = os.path.join(test_dir, "test.csv")

    # Load test CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded test CSV: {len(df)} samples")
    else:
        # No CSV — infer from images directory
        print(f"No test.csv found, reading images from {image_dir}")
        image_files = sorted([
            f.replace(IMAGE_EXT, "")
            for f in os.listdir(image_dir)
            if f.endswith(IMAGE_EXT)
        ])
        df = pd.DataFrame({"image_name": image_files})

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} samples")

    # Load model
    model, processor = load_model(MODEL_DIR)

    # Run inference
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        img_name = str(row["image_name"])
        img_path = os.path.join(image_dir, img_name + IMAGE_EXT)

        predicted = "ERROR"
        try:
            predicted = run_inference(img_path, model, processor)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[WARN] {img_path}: {e}")

        # Convert letter back to number for submission
        letter_to_num = {"A": "1", "B": "2", "C": "3", "D": "4"}
        predicted_num = letter_to_num.get(predicted, predicted)

        results.append({
            "id":        row.get("id", img_name),
            "image_name": img_name,
            "predicted": predicted_num,
        })

    # Save submission.csv in current directory
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved -> {os.path.abspath(OUTPUT_CSV)}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Errors: {(submission_df['predicted'] == 'ERROR').sum()}")
    print(submission_df["predicted"].value_counts().sort_index())


if __name__ == "__main__":
    main()
