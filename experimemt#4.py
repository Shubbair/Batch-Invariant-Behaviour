import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from mlx_lm import load
from mlx_deterministic import (
    DeterministicConfig,
    enable_deterministic_mode,
    enable_mlx_lm_deterministic_mode,
)

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

BATCH_SIZES = [1, 2, 4, 8, 16]

# Same 20 prompts as experiment_2.ipynb
PROMPTS = [
    "Answer only yes or no: Is fire cold?",
    "Answer only yes or no: Is ice hot?",
    "Answer only yes or no: Is the sky blue?",
    "Answer only yes or no: Is grass usually green?",
    "Answer only yes or no: Is 2 greater than 3?",
    "Answer only yes or no: Is 3 greater than 2?",
    "Answer only yes or no: Is 10 an even number?",
    "Answer only yes or no: Is 7 an even number?",
    "Answer only yes or no: Is a whale a fish?",
    "Answer only yes or no: Is a tomato a fruit?",
    "Answer only yes or no: Is Python a programming language?",
    "Answer only yes or no: Is the Earth flat?",
    "Answer only yes or no: Is the Moon larger than the Earth?",
    "Answer only yes or no: Can humans breathe underwater?",
    "Answer only yes or no: Can birds normally fly?",
    "Answer only yes or no: Is one plus one equal to two?",
    "Answer only yes or no: Is zero greater than one?",
    "Answer only yes or no: Is winter hotter than summer?",
    "Answer only yes or no: Is water normally liquid at room temperature?",
    "Answer only yes or no: Is gold a metal?",
]

def get_logits(model, tokenizer, prompt, batch_size):
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    batch = mx.stack([tokens] * batch_size)
    logits = model(batch)
    mx.eval(logits)
    return np.array(logits[0, -1, :])


def analyze(logits, reference, tokenizer):
    top1_id = int(np.argmax(logits))
    ref_id = int(np.argmax(reference))

    diff = logits.astype(np.float64) - reference.astype(np.float64)
    max_diff = float(np.max(np.abs(diff)))

    # top1/top2 margin, for interpreting *why* a flip did or didn't happen
    sorted_ids = np.argsort(logits)
    top1_val = float(logits[sorted_ids[-1]])
    top2_val = float(logits[sorted_ids[-2]])
    margin = top1_val - top2_val

    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id]),
        "max_diff_vs_ref": max_diff,
        "margin": margin,
        "flip_vs_ref": top1_id != ref_id,
    }


def sweep_prompt(model, tokenizer, prompt):
    """Run one prompt across all batch sizes; batch=1 is the reference."""
    outputs = {}
    for b in BATCH_SIZES:
        outputs[b] = get_logits(model, tokenizer, prompt, b)

    reference = outputs[1]
    results = {}
    for b in BATCH_SIZES:
        results[b] = analyze(outputs[b], reference, tokenizer)

    return results


def run_full_sweep(model, tokenizer, label):
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")

    all_results = {}
    for prompt in PROMPTS:
        results = sweep_prompt(model, tokenizer, prompt)
        all_results[prompt] = results

        print(f"\n{prompt}")
        for b in BATCH_SIZES:
            r = results[b]
            marker = " <-- FLIP" if r["flip_vs_ref"] else ""
            print(
                f"  batch={b:2d}  top1={r['top1_token']!r:8s} "
                f"max_diff={r['max_diff_vs_ref']:.6f}  margin={r['margin']:.6f}{marker}"
            )

    return all_results


def summarize(normal_results, det_results):
    print(f"\nSUMMARY: flip rate per batch size, normal vs deterministic\n")
    print(f"{'Batch':<8}{'Normal flips':<16}{'Det flips':<12}")

    normal_flip_counts = {b: 0 for b in BATCH_SIZES}
    det_flip_counts = {b: 0 for b in BATCH_SIZES}

    for prompt in PROMPTS:
        for b in BATCH_SIZES:
            if normal_results[prompt][b]["flip_vs_ref"]:
                normal_flip_counts[b] += 1
            if det_results[prompt][b]["flip_vs_ref"]:
                det_flip_counts[b] += 1

    for b in BATCH_SIZES:
        print(f"{b:<8}{normal_flip_counts[b]:<16}{det_flip_counts[b]:<12}")

    normal_flipped_prompts = [
        p for p in PROMPTS if normal_results[p][16]["flip_vs_ref"]
    ]
    det_flipped_prompts = [
        p for p in PROMPTS if det_results[p][16]["flip_vs_ref"]
    ]

    print("\nPrompts that flip at batch=16, NORMAL mode:")
    for p in normal_flipped_prompts:
        print(" -", p)

    print("\nPrompts that flip at batch=16, DETERMINISTIC mode:")
    for p in det_flipped_prompts:
        print(" -", p)

    normal_diffs = [
        normal_results[p][16]["max_diff_vs_ref"] for p in PROMPTS
    ]
    det_diffs = [
        det_results[p][16]["max_diff_vs_ref"] for p in PROMPTS
    ]

    print(f"\nMean max_diff at batch=16 - Normal: {np.mean(normal_diffs):.6f}")
    print(f"Mean max_diff at batch=16 - Deterministic: {np.mean(det_diffs):.6f}")
    print(
        "(If deterministic mean is NOT meaningfully smaller, this replicates "
        "the experiment_3 finding: determinism prevents flips without "
        "necessarily shrinking the underlying numerical noise.)"
    )

    return normal_flip_counts, det_flip_counts, normal_diffs, det_diffs


def plot_flip_rates(normal_flip_counts, det_flip_counts):
    plt.figure(figsize=(8, 5))
    plt.plot(BATCH_SIZES, [normal_flip_counts[b] for b in BATCH_SIZES],
              marker="o", label="Normal MLX")
    plt.plot(BATCH_SIZES, [det_flip_counts[b] for b in BATCH_SIZES],
              marker="o", label="Deterministic MLX")
    plt.xlabel("Batch size")
    plt.ylabel("Number of prompts flipped (out of 20)")
    plt.title("Flip rate across 20 prompts: Normal vs Deterministic")
    plt.xticks(BATCH_SIZES)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_4_flip_rates.png", dpi=200)
    plt.show()



def main():
    print("Loading normal model...")
    model_normal, tokenizer = load(MODEL)

    normal_results = run_full_sweep(model_normal, tokenizer, "NORMAL MLX - 20 PROMPT SWEEP")

    print("\nLoading deterministic model...")
    model_det, tokenizer_det = load(MODEL)

    enable_mlx_lm_deterministic_mode(split_size=256, verbose=False)

    cfg = DeterministicConfig(use_metal_kernels=True)
    enable_deterministic_mode(model_det, cfg, verbose=False)

    det_results = run_full_sweep(model_det, tokenizer_det, "DETERMINISTIC MLX - 20 PROMPT SWEEP")

    normal_flip_counts, det_flip_counts, normal_diffs, det_diffs = summarize(
        normal_results, det_results
    )

    plot_flip_rates(normal_flip_counts, det_flip_counts)

if __name__ == "__main__":
    main()