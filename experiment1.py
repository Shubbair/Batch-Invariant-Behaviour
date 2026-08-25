import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np

from mlx_lm import load


MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Answer only yes or no: Is zero greater than one?"
BATCH_A = 1
BATCH_B = 16


model, tokenizer = load(MODEL)
tokens = mx.array(tokenizer.encode(PROMPT))
layers = model.model.layers

print("Number of layers:", len(layers))


def capture_residuals(batch_size):
    batch = mx.stack([tokens] * batch_size)
    hidden = model.model.embed_tokens(batch)
    mx.eval(hidden)

    residuals = [np.array(hidden[0, -1, :])]

    for layer in layers:
        hidden = layer(hidden, mask=None, cache=None)
        mx.eval(hidden)
        residuals.append(np.array(hidden[0, -1, :]))

    return residuals


def get_final_logits(batch_size):
    batch = mx.stack([tokens] * batch_size)
    logits = model(batch)
    mx.eval(logits)
    return np.array(logits[0, -1, :])


def print_top(logits, name, count=5):
    print(f"\n{name}")
    token_ids = np.argsort(logits)[-count:][::-1]

    for rank, token_id in enumerate(token_ids, start=1):
        token = tokenizer.decode([int(token_id)])
        print(rank, repr(token), float(logits[token_id]))


print(f"Running batch {BATCH_A}...")
residuals_a = capture_residuals(BATCH_A)

print(f"Running batch {BATCH_B}...")
residuals_b = capture_residuals(BATCH_B)


divergence = []
relative_divergence = []

for hidden_a, hidden_b in zip(residuals_a, residuals_b):
    difference = hidden_a.astype(np.float64) - hidden_b.astype(np.float64)
    absolute = np.linalg.norm(difference)
    relative = absolute / (np.linalg.norm(hidden_a.astype(np.float64)) + 1e-12)

    divergence.append(absolute)
    relative_divergence.append(relative)


divergence = np.array(divergence)
relative_divergence = np.array(relative_divergence)

print("\n" + "=" * 70)
print("RESIDUAL STREAM DIVERGENCE")
print("=" * 70)

for layer_number, value in enumerate(divergence):
    print(f"Layer {layer_number:2d}: {value:.12e}")


logits_a = get_final_logits(BATCH_A)
logits_b = get_final_logits(BATCH_B)

print_top(logits_a, f"BATCH {BATCH_A}")
print_top(logits_b, f"BATCH {BATCH_B}")

logit_difference = logits_a.astype(np.float64) - logits_b.astype(np.float64)

print("\nMax logit difference:", np.max(np.abs(logit_difference)))
print("Mean logit difference:", np.mean(np.abs(logit_difference)))


total_layers = np.arange(len(divergence))

plt.figure(figsize=(10, 6))
plt.plot(total_layers, divergence, marker="o")
plt.xlabel("Layer")
plt.ylabel("||h_batch1 - h_batch16||2")
plt.title("Residual-stream divergence caused by batch size")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("residual_divergence.png", dpi=200)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(total_layers, relative_divergence, marker="o")
plt.xlabel("Layer")
plt.ylabel("Relative residual divergence")
plt.title("Relative residual-stream divergence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("relative_residual_divergence.png", dpi=200)
plt.show()
