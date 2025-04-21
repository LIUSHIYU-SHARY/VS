import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Raw data
data = {
    "Model": ["Original", "Transfer", "Classifier-Filtered"],
    "PSNR (dB)": [25.53, 25.94, 26.84],
    "SSIM": [0.680, 0.680, 0.736],
    "LPIPS": [0.111, 0.104, 0.104],
    "Dice": [0.9985, 0.9985, 0.9985],
    "Pearson": [0.556, 0.569, 0.650],
    "FID": [68.69, 83.90, 78.81]
}
df = pd.DataFrame(data)

# Configuration
metrics = ["PSNR (dB)", "SSIM", "LPIPS", "Dice", "Pearson", "FID"]
prefer_higher = ["PSNR (dB)", "SSIM", "Dice", "Pearson"]
prefer_lower = ["LPIPS", "FID"]
grayscale_colors = ['lightgray', 'gray', 'dimgray']

# Adjust metric titles
titles_with_units = []
for metric in metrics:
    if df[metric].max() <= 1:
        titles_with_units.append(f"{metric} (%)")
    else:
        titles_with_units.append(metric)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

for idx, (ax, metric, title) in enumerate(zip(axs.flat, metrics, titles_with_units)):
    values = df[metric]
    normalized = values / values.max() * 100

    # Whether all values are equal or ≥99.9%
    hatch_all = (values.nunique() == 1) or (values.max() <= 1 and (values >= 0.999).all())

    # Determine best-performing indices
    if not hatch_all:
        best_val = values.max() if metric in prefer_higher else values.min()
        best_indices = values[values == best_val].index.tolist()
    else:
        best_indices = values.index.tolist()

    bars = ax.bar(range(len(values)), normalized, color=grayscale_colors, edgecolor='black')

    for i, (bar, val) in enumerate(zip(bars, values)):
        if i in best_indices:
            bar.set_hatch('//')
        # arrow = "↑" if metric in prefer_higher else "↓"
        label = f'{val*100:.1f}%' if val < 1 else f'{val:.2f} '
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=10)

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xticks(range(len(df["Model"])))
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylim(0, 110)
    ax.grid(False)

# Legend (centered below)
best_patch = mpatches.Patch(facecolor='white', hatch='//', edgecolor='black', label='Best or All Equal/≥99.9%')
gray_patches = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(grayscale_colors, df["Model"])
]
plt.legend(handles=[best_patch] + gray_patches, loc='lower center', bbox_to_anchor=(0.452, -0.3), ncol=4)

# Title and notes
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
plt.suptitle("Comparison of Original, Transfer, and Classifier-Filtered", fontsize=16, y=1.05)

note_text = """Notes:
1. Higher values are better for PSNR(dB), SSIM(%), Pearson(%)
2. Lower values are better for LPIPS(%), FID
3. Striped bars indicate the best-performing model(s)"""
plt.figtext(0.1, -0.05, note_text, horizontalalignment='left', fontsize=12)

# Show and save
plt.show()
plt.savefig("normalized_metrics_english.png", dpi=600, bbox_inches='tight')
