# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from class_labels import LABELS_GENDER, LABELS_ACCENT, LABELS_AGE

### ETYKIETY KLAS
LABELS = {
    "gender": LABELS_GENDER,
    "age": LABELS_AGE,
    "accent": LABELS_ACCENT,
}

### SCIEZKI FOLDEROW
input_folder = Path("../results_and_weights/results/confusion_matrixes")

output_folder = Path('../results_and_weights/results/cm_plots')
output_folder.mkdir(parents=True, exist_ok=True)

normalized_output_folder = Path('../results_and_weights/results/cm_norm_plots')
normalized_output_folder.mkdir(parents=True, exist_ok=True)

### GENEROWANIE MACIERZY POMYLEK Z PLIKOW NUMPY
for file in sorted(input_folder.glob("*.npy")):

    print(f"Przetwarzanie: {file.name}")
    cm = np.load(file)
    filename = file.stem.lower()

    if "gender" in filename:
        labels = LABELS["gender"]
        task = "Gender"

    elif "accent" in filename:
        labels = LABELS["accent"]
        task = "Accent"

    elif "age" in filename:
        labels = LABELS["age"]
        task = "Age"

    else:
        print(f"  -> Pomijam: nie rozpoznano typu macierzy")
        continue

    if cm.ndim != 2:
        print(f"  -> Pomijam: macierz nie jest 2D, shape={cm.shape}")
        continue

    if cm.shape[0] != len(labels) or cm.shape[1] != len(labels):
        print(
            f"  -> UWAGA: rozmiar macierzy {cm.shape} "
            f"nie pasuje do liczby etykiet ({len(labels)})"
        )
        continue

    # Automatyczne konfigurowanie proporcji wykresu
    n_classes = len(labels)
        
    figsize = (
        max(8, n_classes * 0.7),
        max(7, n_classes * 0.7)
    )
    
    fontsize = max(5, min(10, 150 // n_classes))
    label_fontsize = max(6, min(10, 150 // n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    # Wyświetlenie macierzy
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    annot = np.empty_like(cm_percent, dtype=object)

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            annot[i, j] = f"{cm_percent[i, j]:.1f}%"


    sns.heatmap(
    cm_percent,
    annot=annot,
    fmt="",
    cmap="viridis",
    cbar=True,
    ax=ax,
    annot_kws={
        "color": "white",
        "fontsize": 11
    },
    linewidths=1,
    linecolor="white",
    cbar_kws={
        "label": "Udział procentowy (%)"
    }
)

    ax.set_title(
    f"Model: {file.stem.removesuffix('_cm')}",
    fontsize=15,
    fontweight="bold"
)

    ax.set_xlabel("Przewidziane", fontsize=12)
    ax.set_ylabel("Prawdziwe", fontsize=12)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()

    normalized_output_file = (
        normalized_output_folder / f"{file.stem}_norm.png"
    )

    plt.savefig(
        normalized_output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    print(f"  -> Zapisano (normalized): {normalized_output_file}")

print("\nGotowe! Macierze pomylek zostaly wygenerowane i zapisane do folderów.")
# %%
