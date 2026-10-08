"""Compare diffusion step counts on prompt alignment (CLIP score) and latency."""
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from core import generate_image, DEVICE

PROMPTS = [
    "a red vintage car on a coastal road at sunset",
    "a futuristic city skyline at night with neon lights",
    "a bowl of ramen with steam, close-up photo",
    "a snowy mountain cabin, cinematic lighting",
    "an astronaut riding a horse on mars",
    "a cat sleeping on a stack of books, warm light",
    "a bustling street market in india, golden hour",
    "a lighthouse in a storm, dramatic waves",
    "a minimalist living room with large windows",
    "a robot gardener watering flowers",
]  # extend to ~30 for a stronger result
STEP_SETTINGS = [1, 2, 4]

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def clip_score(img, text):
    inputs = proc(text=[text], images=img, return_tensors="pt", padding=True).to(DEVICE)
    out = clip(**inputs)
    i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((i * t).sum() * 100)


rows = []
for steps in STEP_SETTINGS:
    for p in PROMPTS:
        img, secs = generate_image(p, seed=0, steps=steps)
        rows.append({"prompt": p, "steps": steps, "clip_score": clip_score(img, p), "latency_s": secs})
        print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("eval_results.csv", index=False)
summary = df.groupby("steps")[["clip_score", "latency_s"]].mean()
print(summary)

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.bar(summary.index.astype(str), summary["clip_score"], color="#4c72b0")
ax1.set_xlabel("Diffusion steps"); ax1.set_ylabel("Mean CLIP score")
ax2 = ax1.twinx()
ax2.plot(summary.index.astype(str), summary["latency_s"], color="#dd8452", marker="o")
ax2.set_ylabel("Mean latency (s)")
plt.title("Quality vs speed (SDXL-Turbo)"); plt.tight_layout()
plt.savefig("eval_chart.png", dpi=150)
