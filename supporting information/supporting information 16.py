import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from scipy.stats import kruskal, linregress, shapiro, spearmanr, wilcoxon

# This script lives in the "supporting information" folder; input SI files sit
# alongside it, while figures and tables live in their own sibling folders.
SI_DIR = Path(__file__).resolve().parent
REPO_ROOT = SI_DIR.parent
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "tables"

ORDER_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
ORDER_NAMES = ["Amphipoda", "Blattodea", "Diptera", "Hymenoptera", "Lepidoptera"]
ORDER_PALETTE = dict(zip(ORDER_NAMES, ORDER_COLORS))


def supporting_information_6(supporting_information_5: Path) -> None:
    """Create the qPCR validation heatmap (Supporting Information 6).

    Before running metabarcoding, we checked with taxon-specific qPCR assays
    whether each target insect order was actually detectable in each community
    sample. This function reads those qPCR results and produces a colour-coded
    grid (heatmap) where rows are insect orders, columns are samples, and each
    cell is coloured by detection outcome: true positive (detected as expected),
    true negative (absent as expected), false positive (detected but absent),
    or false negative (absent but expected). Both extraction replicates (A and B)
    are shown as separate panels. The result is saved as Supporting Information 6.
    """
    # --- Data loading ---
    supporting_information_5 = pd.read_excel(supporting_information_5)

    # Fixed display order of the five qPCR target taxa (rows of the heatmap)
    TARGET_ORDER = ["Amphipoda", "Diptera", "Lepidoptera", "Blattodea", "Hymenoptera"]

    # Sample order follows the original table; negative controls are already last
    SAMPLE_ORDER = supporting_information_5["sample"].unique().tolist()

    # --- Colour mapping for the four detection states ---
    STATE_LABELS = {
        0: "True positive",
        1: "True negative",
        2: "False positive",
        3: "False negative",
    }

    # Colorblind-friendly palette based on Okabe-Ito
    STATE_COLORS = {
        0: "#0072B2",  # TP  – blue
        1: "#D9D9D9",  # TN  – light gray
        2: "#E69F00",  # FP  – orange
        3: "#D55E00",  # FN  – vermillion
    }

    # Discrete colormap: map integer states 0–3 to the four colours above
    cmap = ListedColormap([STATE_COLORS[i] for i in range(4)])
    # Boundaries centred around each integer so imshow bins values correctly
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # --- Helper: build a (target × sample) matrix for one replicate ---
    def make_matrix(replicate):
        sub = supporting_information_5[
            supporting_information_5["replicate"] == replicate
        ]
        pivot = sub.pivot(index="Target Name", columns="sample", values="state")
        # Enforce consistent row/column order across replicates
        pivot = pivot.reindex(index=TARGET_ORDER, columns=SAMPLE_ORDER)
        return pivot

    mat_a = make_matrix("A")
    mat_b = make_matrix("B")

    # --- Figure setup: two vertically stacked heatmaps (one per replicate) ---
    fig, axes = plt.subplots(
        2, 1, figsize=(18, 6.5), sharex=True, gridspec_kw={"hspace": 0.4}
    )

    for ax, mat, label in zip(axes, [mat_a, mat_b], ["Replicate A", "Replicate B"]):
        # Draw the heatmap for this replicate
        ax.imshow(
            mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
        )

        # Y-axis: target taxa labels
        ax.set_yticks(range(len(TARGET_ORDER)))
        ax.set_yticklabels(TARGET_ORDER, fontsize=11)
        ax.set_ylabel(label, fontsize=12, fontweight="bold")

        # X-axis: one tick per sample (labels only on the bottom panel)
        ax.set_xticks(range(len(SAMPLE_ORDER)))

        # Remove default tick marks and axis spines for a cleaner look
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw a grid between cells using minor ticks placed at cell boundaries
        ax.set_xticks([x - 0.5 for x in range(1, len(SAMPLE_ORDER))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(TARGET_ORDER))], minor=True)
        ax.grid(which="minor", color="black", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        # Draw an explicit outer border around the entire matrix
        ax.set_xlim(-0.5, len(SAMPLE_ORDER) - 0.5)
        ax.set_ylim(len(TARGET_ORDER) - 0.5, -0.5)
        for x in [-0.5, len(SAMPLE_ORDER) - 0.5]:
            ax.axvline(x, color="black", linewidth=0.8)
        for y in [-0.5, len(TARGET_ORDER) - 0.5]:
            ax.axhline(y, color="black", linewidth=0.8)

    # X-axis labels only on the bottom panel, rotated for readability
    axes[-1].set_xticklabels(
        SAMPLE_ORDER, rotation=45, ha="right", rotation_mode="anchor", fontsize=7
    )
    axes[-1].set_xlabel("Sample", fontsize=12)

    fig.suptitle("qPCR assay validation heatmap", fontsize=14, y=0.97)

    # --- Legend (placed to the right of the heatmaps) ---
    legend_patches = [
        mpatches.Patch(color=STATE_COLORS[i], label=STATE_LABELS[i]) for i in range(4)
    ]
    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(0.96, 0.5),
        frameon=False,
        fontsize=11,
    )

    # Leave 5 % right margin for the legend; avoid tight_layout (incompatible
    # with sharex and manual legend placement). bbox_inches="tight" in savefig
    # handles the final bounding box.
    fig.subplots_adjust(right=0.95)

    # --- Export ---
    plt.savefig(SI_DIR / "supporting information 6.pdf", dpi=300, bbox_inches="tight")


def aggregate_metabarcoding_data(
    supporting_information_1: Path, supporting_information_11: Path
) -> pd.DataFrame:
    """Load and clean the raw metabarcoding sequencing data (Supporting Information 11).

    Metabarcoding produces a table of DNA sequence counts ("reads") for each
    taxon detected in each sample. This function loads that raw table, sums
    reads across sequencing lanes for each sample/taxon/primer/PCR-cycle
    combination, and then removes any reads belonging to taxa that were not
    intentionally added to that community (co-amplified non-target sequences).
    Three samples (26, 92, 93) that were not part of the controlled experiment
    are also excluded. Combinations where a taxon was expected but produced
    zero reads are explicitly recorded as zero rather than left as missing,
    so that all downstream calculations see complete data.
    Returns the cleaned read table.
    """
    # --- Load and aggregate raw read data ---
    supporting_information_11 = pd.read_parquet(supporting_information_11)

    supporting_information_11 = (
        supporting_information_11.groupby(
            by=["sample", "aliquot", "cycles", "order", "primer"]
        )
        .sum()["read_count"]
        .reset_index()
    )

    # --- Load expected community composition ---
    supporting_information_1 = pd.read_excel(supporting_information_1)
    expected_taxa = supporting_information_1[["sample", "order"]].drop_duplicates()

    # --- Filter to samples present in SI 1 (drops e.g. 26, 92, 93) ---
    valid_samples = set(expected_taxa["sample"])
    supporting_information_11 = supporting_information_11[
        supporting_information_11["sample"].isin(valid_samples)
    ]

    # Total reads per primer before removing co-amplifications
    pre_filter_reads = supporting_information_11.groupby("primer")["read_count"].sum()

    # --- Build a complete design grid ---
    # All experimental conditions that were actually sequenced
    design = supporting_information_11[
        ["sample", "aliquot", "cycles", "primer"]
    ].drop_duplicates()

    # Cross-join each condition with the expected orders for that sample
    design_grid = design.merge(expected_taxa, on="sample", how="left")

    # --- Merge actual read counts onto the grid; fill missing with 0 ---
    result = design_grid.merge(
        supporting_information_11,
        on=["sample", "aliquot", "cycles", "order", "primer"],
        how="left",
    )
    result["read_count"] = result["read_count"].fillna(0).astype(int)

    # Total reads per primer after removing co-amplifications
    post_filter_reads = result.groupby("primer")["read_count"].sum()

    print("\n=== Metabarcoding read filtering (samples 26/92/93 excluded) ===")
    for primer in ["fwh2", "bf3"]:
        pre = pre_filter_reads[primer]
        post = post_filter_reads.get(primer, 0)
        removed = pre - post
        pct = removed / pre * 100 if pre > 0 else 0
        print(
            f"  {primer}:\n"
            f"    Before filtering: {pre:,}\n"
            f"    After filtering:  {post:,}\n"
            f"    Removed:          {removed:,} ({pct:.2f}%)"
        )

    # Sort by primer (fwh2 before bf3), then sample/cycles/aliquot ascending
    result = result.sort_values(
        by=["primer", "sample", "cycles", "aliquot"],
        ascending=[False, True, True, True],
    )

    return result


def absolute_reads_to_relative_reads(
    aggregated_read_data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert raw read counts to relative proportions and verify extraction reproducibility.

    Because total sequencing depth varies between samples and runs, raw read
    counts are not directly comparable. This function divides each taxon's
    read count by the total reads in that sample/aliquot/cycle group, giving
    the relative proportion (0–1) of each taxon. It then checks reproducibility:
    each sample was extracted in two aliquots, and a Wilcoxon signed-rank test
    (a non-parametric pairwise test) checks whether the two aliquots give
    consistently similar proportions. After the check the two aliquots are
    pooled (summed) into a single value per sample.
    Returns the table with relative proportions added.
    """
    # --- Compute read sum per (sample, aliquot, cycles, primer) ---
    grouped_data = (
        aggregated_read_data.groupby(by=["sample", "aliquot", "cycles", "primer"])
        .sum()["read_count"]
        .reset_index()
        .rename(columns={"read_count": "read_sum"})
    )

    # Merge read_sum back so each row knows its group total
    aggregated_read_data = pd.merge(
        left=aggregated_read_data,
        right=grouped_data,
        how="left",
        on=["sample", "aliquot", "cycles", "primer"],
    )

    # Relative reads = proportion of each order within its group
    aggregated_read_data["relative_reads"] = (
        aggregated_read_data["read_count"] / aggregated_read_data["read_sum"]
    )

    # --- Wilcoxon signed-rank test: aliquot 1 vs. aliquot 2 per primer ---
    # Global test across all orders per primer. Compositional dependence between
    # orders is accepted here; per-order tests showed that the only significant
    # differences stem from stochastic sequencing noise at low-abundance taxa
    # (Hymenoptera), not from a systematic aliquot effect.
    PRIMER_LABELS = {"fwh2": "fwh2", "bf3": "BF3"}
    print("\n=== Metabarcoding aliquot consistency (Wilcoxon signed-rank test) ===")
    for primer, sub in aggregated_read_data.groupby("primer", sort=False):
        aliquot_1 = sub[sub["aliquot"] == 1].set_index(["sample", "cycles", "order"])[
            "relative_reads"
        ]
        aliquot_2 = sub[sub["aliquot"] == 2].set_index(["sample", "cycles", "order"])[
            "relative_reads"
        ]

        paired = pd.DataFrame({"a1": aliquot_1, "a2": aliquot_2}).dropna()
        stat, p = wilcoxon(paired["a1"], paired["a2"])
        label = PRIMER_LABELS.get(primer, primer)
        print(f"  {label}: statistic={stat:.2f}, p={p:.4e}, n_pairs={len(paired)}")

    plot_aliquot_consistency(aggregated_read_data)

    # --- Pool aliquots into a single value per (sample, cycles, primer, order) ---
    # Summing read_count and read_sum across both aliquots gives the pooled
    # proportion (weighted by sequencing depth), not the arithmetic mean.
    aggregated_read_data = (
        aggregated_read_data.groupby(by=["sample", "cycles", "primer", "order"])
        .sum()[["read_count", "read_sum"]]
        .reset_index()
    )

    # Recompute relative reads from pooled counts
    aggregated_read_data["relative_reads"] = (
        aggregated_read_data["read_count"] / aggregated_read_data["read_sum"]
    )

    return aggregated_read_data


def plot_aliquot_consistency(aggregated_read_data: pd.DataFrame) -> None:
    """Plot extraction aliquot reproducibility (Supporting Information 12).

    For each primer, creates a scatter plot comparing aliquot 1 vs. aliquot 2
    relative read proportions for every taxon and sample. Points lying on the
    1:1 diagonal indicate that both aliquots gave the same result. Points far
    from the diagonal indicate extraction variability for that taxon.
    Saved as Supporting Information 12.
    """
    PRIMER_LABELS = {"fwh2": "fwh2", "bf3": "BF3"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (primer, label) in zip(axes, PRIMER_LABELS.items()):
        # Pivot so each row is one (sample, order, cycles) pair with
        # aliquot 1 and aliquot 2 as separate columns
        sub = aggregated_read_data[aggregated_read_data["primer"] == primer]
        pivot = sub.pivot_table(
            index=["sample", "order", "cycles"],
            columns="aliquot",
            values="relative_reads",
        ).reset_index()

        # Show legend only on the last panel to avoid duplicates
        show_legend = primer == "bf3"
        sns.scatterplot(
            data=pivot,
            x=1,
            y=2,
            hue="order",
            palette=ORDER_PALETTE,
            hue_order=ORDER_NAMES,
            edgecolor="black",
            linewidth=1,
            legend=show_legend,
            ax=ax,
        )

        # 1:1 reference line — perfect aliquot agreement
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Relative reads in aliquot 1")
        ax.set_ylabel("Relative reads in aliquot 2")
        ax.set_title(label)

    # Move the seaborn legend from the right panel to a shared, centred position
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].get_legend().remove()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=len(ORDER_NAMES),
        frameon=True,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle("Aliquot consistency of relative read proportions")
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(SI_DIR / "supporting information 12.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_dpcr_data(supporting_information_8: Path) -> pd.DataFrame:
    """Load dPCR copy number data, verify replicate agreement, and pool replicates.

    Digital PCR (dPCR) gives an absolute count of mitochondrial DNA
    (mtDNA) copies per taxon in each sample. This function loads those raw
    counts (Supporting Information 8), converts them to relative proportions
    (each taxon's copies divided by total copies in that sample), and checks
    whether two independent dPCR replicates (A and B) agree using a Wilcoxon
    signed-rank test. A scatter plot comparing the replicates is saved as
    Supporting Information 13. The two replicates are then averaged to obtain
    a single reliable copy-number estimate per sample and taxon.
    Returns the pooled copy-number table.
    """
    # --- Load dPCR data ---
    dpcr_data = pd.read_excel(supporting_information_8)

    # --- Compute relative copy numbers per (sample, replicate) ---
    total_copies = (
        dpcr_data.groupby(by=["sample", "replicate"])
        .sum()["mtDNA copies"]
        .reset_index()
        .rename(columns={"mtDNA copies": "total copies"})
    )

    dpcr_data = dpcr_data.merge(
        total_copies,
        how="left",
        on=["sample", "replicate"],
    )

    dpcr_data["relative copies"] = (
        dpcr_data["mtDNA copies"] / dpcr_data["total copies"]
    )

    # --- Scatterplot: replicate A vs. replicate B (supporting information 13) ---
    pivot = dpcr_data.pivot_table(
        index=["sample", "order"],
        columns="replicate",
        values="relative copies",
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=pivot,
        x="A",
        y="B",
        hue="order",
        palette=ORDER_PALETTE,
        hue_order=ORDER_NAMES,
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Relative mtDNA copies in replicate A")
    ax.set_ylabel("Relative mtDNA copies in replicate B")
    ax.set_title("Replicate consistency of relative mtDNA copy numbers")
    ax.legend(frameon=True, handletextpad=0.3)
    plt.savefig(SI_DIR / "supporting information 13.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Wilcoxon signed-rank test: replicate A vs. replicate B ---
    paired = pivot.dropna(subset=["A", "B"])
    stat, p = wilcoxon(paired["A"], paired["B"])
    print(
        f"\n=== dPCR replicate consistency (Wilcoxon signed-rank test) ===\n"
        f"  statistic={stat:.2f}, p={p:.4e}, n_pairs={len(paired)}"
    )

    # --- Pool replicates by averaging copy numbers per (sample, order) ---
    # Mean (not sum) to preserve the physical scale of copy numbers
    dpcr_data = (
        dpcr_data.groupby(by=["sample", "order"])
        .mean(numeric_only=True)[["mtDNA copies"]]
        .reset_index()
    )

    # Recompute totals and relative proportions from pooled means
    total_copies = (
        dpcr_data.groupby("sample")["mtDNA copies"].sum().rename("total copies")
    )
    dpcr_data = dpcr_data.merge(total_copies, on="sample")
    dpcr_data["relative copies"] = (
        dpcr_data["mtDNA copies"] / dpcr_data["total copies"]
    )

    return dpcr_data


def final_aggregation(
    relative_read_data: pd.DataFrame,
    dpcr_data: pd.DataFrame,
    supporting_information_1: Path,
) -> pd.DataFrame:
    """Combine the three data sources into one master table.

    This function brings together the three measurements collected for each
    taxon in each sample: relative metabarcoding reads (proportions of DNA
    sequences), relative mtDNA copy numbers from dPCR, and biomass from
    weighing. After merging, every row in the resulting table holds all three
    values for one taxon in one sample at one PCR cycle count.
    This master table (Supporting Information 14) is the basis for all
    subsequent comparisons and figures.
    """
    supporting_information_1 = pd.read_excel(supporting_information_1)

    # Merge reads ← dPCR ← biomass on shared (sample, order) keys
    full_results = relative_read_data.merge(
        dpcr_data, on=["sample", "order"], how="left"
    ).merge(supporting_information_1, on=["sample", "order"], how="left")

    full_results = full_results.sort_values(
        by=["primer", "sample", "cycles", "order"],
        ascending=[False, True, True, True],
    )

    return full_results


def plot_stacked_barplots(
    full_results: pd.DataFrame, primer: str, cycles: int, filename: str
) -> None:
    """Plot per-sample community compositions as stacked bar charts (SI 18/19).

    Creates a 9×9 grid (one panel per sample) where each panel shows three
    stacked bars side by side: relative biomass, relative mtDNA copies, and
    relative metabarcoding reads. Each segment within a bar is coloured by
    insect order. This allows visual comparison of how well the three
    measurement types agree on the relative abundances within each community.
    Saved as Supporting Information 18 (fwh2) and 19 (BF3).
    """
    sub = full_results[
        (full_results["primer"] == primer) & (full_results["cycles"] == cycles)
    ]
    samples = sorted(sub["sample"].unique())

    bar_columns = ["relative biomass", "relative copies", "relative_reads"]
    bar_labels = ["Biomass", "mtDNA copies", "Metabarcoding reads"]
    bar_width = 0.4
    x = np.arange(len(bar_columns)) * bar_width

    fig, axes = plt.subplots(9, 9, figsize=(8, 12), sharey=True)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(samples):
            ax.set_visible(False)
            continue

        sample = samples[idx]
        sample_data = sub[sub["sample"] == sample].set_index("order")
        sample_data = sample_data.reindex(ORDER_NAMES).fillna(0)

        # Stack bars by order, accumulating bottom offsets
        bottom = np.zeros(len(bar_columns))
        for order, color in zip(ORDER_NAMES, ORDER_COLORS):
            values = [sample_data.loc[order, col] for col in bar_columns]
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=color,
                width=bar_width,
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += values

        ax.set_title(f"{sample}", fontsize=7, pad=2)
        ax.set_ylim(0, 1)
        ax.set_xlim(x[0] - bar_width / 2, x[-1] + bar_width / 2)
        ax.tick_params(labelsize=5, length=2)

        # y-axis label only on leftmost column
        if idx % 9 == 0:
            ax.set_ylabel("Proportion", fontsize=6)
        else:
            ax.tick_params(labelleft=False)

        # x-axis labels only on bottom row
        if idx >= 72:
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=5)
        else:
            ax.set_xticks([])

    # Shared legend at bottom
    PRIMER_LABELS = {"fwh2": "fwh2", "bf3": "BF3"}
    primer_label = PRIMER_LABELS.get(primer, primer)
    fig.suptitle(
        f"Per-sample community composition ({primer_label}, {cycles} cycles)",
        fontsize=10,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ORDER_PALETTE[name],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6,
            label=name,
        )
        for name in ORDER_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ORDER_NAMES),
        frameon=True,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=7,
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.05, bottom=0.06, top=0.95)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reads_per_cycle(
    full_results: pd.DataFrame, primer: str, filename: str
) -> None:
    """Plot how relative read proportions change across PCR cycle counts (SI 20).

    Creates a 9×9 grid (one panel per sample) where the x-axis shows PCR
    cycle count (6, 8, 10, … 20) and each stacked bar shows the relative
    read proportion of each taxon at that cycle count. If amplification bias
    is cycle-independent (as our constant bias model assumes), the bar
    compositions should look the same at every cycle — any taxon that is
    over- or underrepresented at c=20 should be equally over- or
    underrepresented at c=6. Saved as Supporting Information 20.
    """
    sub = full_results[full_results["primer"] == primer].copy()
    samples = sorted(sub["sample"].unique())
    cycles = sorted(sub["cycles"].unique())

    bar_width = 0.6
    x_pos = np.arange(len(cycles))

    fig, axes = plt.subplots(9, 9, figsize=(14, 14), sharey=True)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(samples):
            ax.set_visible(False)
            continue

        sample = samples[idx]
        sd = sub[sub["sample"] == sample]

        for cx, c in enumerate(cycles):
            cd = sd[sd["cycles"] == c].set_index("order")
            bottom = 0.0
            for order, color in zip(ORDER_NAMES, ORDER_COLORS):
                val = cd.loc[order, "relative_reads"] if order in cd.index else 0.0
                ax.bar(
                    cx,
                    val,
                    bottom=bottom,
                    color=color,
                    width=1.0,
                    edgecolor="black",
                    linewidth=0.3,
                )
                bottom += val

        ax.set_title(f"{sample}", fontsize=6, pad=2)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(cycles) - 0.5)
        ax.tick_params(labelsize=4, length=2)

        if idx % 9 == 0:
            ax.set_ylabel("Rel. reads", fontsize=5)
        else:
            ax.tick_params(labelleft=False)

        if idx >= 72:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [str(c) for c in cycles], fontsize=4, rotation=45, ha="right"
            )
        else:
            ax.set_xticks([])

    PRIMER_LABELS = {"fwh2": "fwh2", "bf3": "BF3"}
    primer_label = PRIMER_LABELS.get(primer, primer)
    fig.suptitle(
        f"Relative metabarcoding read proportions across PCR cycles ({primer_label})",
        fontsize=10,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ORDER_PALETTE[name],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6,
            label=name,
        )
        for name in ORDER_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ORDER_NAMES),
        frameon=True,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=7,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.05, bottom=0.05, top=0.95)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_fold_changes_and_ranks(full_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate fold-changes and abundance ranks between the three measurement types.

    To quantify how well metabarcoding reads track biomass and mtDNA copies,
    this function computes log2 fold-changes between all three pairs of
    measurements (biomass vs. copies, copies vs. reads, biomass vs. reads).
    A fold-change of 0 means perfect agreement; +1 means the second measure
    is twice as high as the first (2-fold over-representation); -1 means
    half as high (2-fold under-representation). The ±log2(1.5) band
    corresponds to a ±1.5× agreement zone. Additionally, within each sample
    the taxa are ranked by abundance (rank 1 = most abundant) for each
    measurement type, which allows checking whether the rank order is
    preserved even when absolute values differ.
    """

    # --- Log2 fold-changes between all three measurement pairs ---
    def safe_log2_ratio(numerator, denominator):
        ratio = numerator / denominator
        return np.log2(ratio.replace(0, np.nan))

    full_results["fc_biomass_copies"] = safe_log2_ratio(
        full_results["relative copies"], full_results["relative biomass"]
    )
    full_results["fc_copies_reads"] = safe_log2_ratio(
        full_results["relative_reads"], full_results["relative copies"]
    )
    full_results["fc_biomass_reads"] = safe_log2_ratio(
        full_results["relative_reads"], full_results["relative biomass"]
    )

    # --- Within-sample ranks (1 = highest proportion) ---
    group_keys = ["sample", "cycles", "primer"]
    full_results["rank_biomass"] = full_results.groupby(group_keys)[
        "relative biomass"
    ].rank(ascending=False)
    full_results["rank_copies"] = full_results.groupby(group_keys)[
        "relative copies"
    ].rank(ascending=False)
    full_results["rank_reads"] = full_results.groupby(group_keys)[
        "relative_reads"
    ].rank(ascending=False)

    return full_results


def export_table1_rank_concordance(
    full_results: pd.DataFrame, filename: str = str(TABLES_DIR / "table 1.xlsx")
) -> None:
    """Export Table 1: how often do the three measurement types agree on rank order?

    For each pair of measurements (biomass vs. copies, copies vs. reads,
    biomass vs. reads) and each primer, counts how many taxa have the same
    within-sample rank (e.g. most abundant in both biomass and reads). Reports
    the count and percentage of rank-concordant cases. Saved as Table 1.
    """
    rank_pairs = [
        ("rank_biomass", "rank_copies", "Biomass - Copies"),
        ("rank_copies", "rank_reads", "Copies - Reads"),
        ("rank_biomass", "rank_reads", "Biomass - Reads"),
    ]
    sub = full_results[full_results["cycles"] == 20]
    rows = []
    for primer, primer_label in [("fwh2", "fwh2"), ("bf3", "BF3")]:
        p = sub[sub["primer"] == primer]
        row = {"Primer": primer_label}
        for rank_a, rank_b, label in rank_pairs:
            match = int((p[rank_a] == p[rank_b]).sum())
            total = len(p)
            pct = match / total * 100
            row[label] = f"{match}/{total} ({pct:.1f}%)"
        rows.append(row)
    pd.DataFrame(rows).to_excel(filename, index=False)


def export_si11_over_under(
    full_results: pd.DataFrame, filename: str = str(SI_DIR / "supporting information 17.xlsx")
) -> None:
    """Export Supporting Information 17: over- and under-representation counts per taxon.

    For each pair of measurements and each insect order, counts how many
    samples fall into three categories: over-represented (fold-change > +1.5×),
    concordant (within ±1.5×), or under-represented (fold-change < -1.5×).
    The results are written into three Excel sheets — one sheet per measurement
    pair — with counts and percentages for both primers side by side.
    Note: Biomass vs. copies does not involve sequencing reads and is therefore
    primer-independent; the values are identical for fwh2 and BF3.
    """
    fc_pairs = [
        ("fc_biomass_copies", "Biomass vs Copies"),
        ("fc_copies_reads", "Copies vs Reads"),
        ("fc_biomass_reads", "Biomass vs Reads"),
    ]
    band = np.log2(1.5)
    sub = full_results[full_results["cycles"] == 20]

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for fc_col, sheet_name in fc_pairs:
            # Biomass vs Copies is primer-independent — one table using fwh2 rows.
            # Copies/Reads and Biomass/Reads show fwh2 and BF3 side by side.
            primers = (
                [("fwh2", "fwh2")]
                if fc_col == "fc_biomass_copies"
                else [("fwh2", "fwh2"), ("bf3", "BF3")]
            )
            rows = []
            for order in ORDER_NAMES:
                row = {"Order": order}
                for primer, col_prefix in primers:
                    vals = sub[(sub["primer"] == primer) & (sub["order"] == order)][
                        fc_col
                    ].dropna()
                    n = len(vals)
                    over = int((vals > band).sum())
                    conc = int((vals.abs() <= band).sum())
                    under = int((vals < -band).sum())
                    prefix = "" if fc_col == "fc_biomass_copies" else f"{col_prefix} "
                    row[f"{prefix}n"] = n
                    row[f"{prefix}over (n)"] = over
                    row[f"{prefix}over (%)"] = round(over / n * 100, 1) if n else 0
                    row[f"{prefix}conc. (n)"] = conc
                    row[f"{prefix}conc. (%)"] = round(conc / n * 100, 1) if n else 0
                    row[f"{prefix}under (n)"] = under
                    row[f"{prefix}under (%)"] = round(under / n * 100, 1) if n else 0
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)


def plot_fold_change_stripplot(
    full_results: pd.DataFrame, primer: str, cycles: int, filename: str
) -> None:
    """Plot fold-changes between measurement types per sample (Figure 2).

    Creates Figure 2 with three panels, one for each pair of measurements
    (A: biomass vs. mtDNA copies, B: copies vs. reads, C: biomass vs. reads).
    Each dot represents one taxon in one sample, coloured by insect order;
    light grey vertical lines mark the individual samples.
    The grey band marks the ±1.5× agreement zone — dots inside the band are
    considered concordant. Dots above the band indicate the second measurement
    is more than 1.5× higher; dots below indicate more than 1.5× lower.
    Saved as Figure 2.
    """
    sub = full_results[
        (full_results["primer"] == primer) & (full_results["cycles"] == cycles)
    ].copy()
    sub["sample"] = sub["sample"].astype(str)
    sample_order = [str(s) for s in sorted(full_results["sample"].unique())]

    panels = [
        ("fc_biomass_copies", "A)  Biomass vs. mtDNA copies"),
        ("fc_copies_reads", "B)  mtDNA copies vs. metabarcoding reads"),
        ("fc_biomass_reads", "C)  Biomass vs. metabarcoding reads"),
    ]

    # ±1.5x band (x1.5 / ÷1.5) symmetric in log2 space
    band = np.log2(1.5)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for ax, (fc_col, title) in zip(axes, panels):
        # Agreement band and reference lines
        ax.axhspan(-band, band, color="lightgray", alpha=0.4, zorder=0)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, zorder=1)
        ax.axhline(band, color="silver", linestyle="--", linewidth=0.6, zorder=1)
        ax.axhline(-band, color="silver", linestyle="--", linewidth=0.6, zorder=1)
        # One light grey guide line per sample so that each dot can be
        # assigned to its sample without jitter
        for pos in range(len(sample_order)):
            ax.axvline(pos, color="#d9d9d9", linewidth=0.6, zorder=0.5)

        sns.stripplot(
            data=sub,
            x="sample",
            y=fc_col,
            hue="order",
            palette=ORDER_PALETTE,
            hue_order=ORDER_NAMES,
            order=sample_order,
            dodge=False,
            jitter=False,
            size=6,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
            legend=False,
            ax=ax,
            zorder=2,
        )

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Fold-change (log$_2$)", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(labelsize=10)

    # x-axis labels only on bottom panel
    axes[-1].set_xticklabels(sample_order, rotation=90, fontsize=10)
    axes[-1].set_xlabel("Sample", fontsize=12)

    # Shared legend at bottom (scatter markers, matching SI 12 style)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ORDER_PALETTE[name],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6,
            label=name,
        )
        for name in ORDER_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ORDER_NAMES),
        frameon=True,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=11,
    )

    fig.subplots_adjust(hspace=0.18, bottom=0.10)
    # Save as both PDF and PNG
    base = filename.rsplit(".", 1)[0]
    plt.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_cycle_calibration(
    full_results: pd.DataFrame, primer: str = "fwh2"
) -> pd.DataFrame:
    """Estimate amplification bias using the cycle calibration approach of Shelton et al. (2023).

    The key idea: if amplification efficiency differs between taxa, the
    log-ratio of their read counts changes linearly with PCR cycle number.
    By measuring the same sample at many different cycle counts (here 8 counts:
    c = 6, 8, 10, … 20), we can fit a straight line to log(reads_i / reads_ref)
    vs. cycle number for each taxon i. The slope of this line estimates the
    efficiency difference (E_cycle_cal), and the y-intercept estimates the
    log-ratio of starting copy numbers before PCR — which is what we actually
    want to recover. Converting those intercepts back to proportions gives
    the predicted relative copy numbers (pred_copies_cycle_cal).

    Crucially, this approach requires no external calibration samples: all
    information is extracted from the variation across cycle counts within
    each community sample. The most abundant taxon at c=20 is used as the
    reference (choice of reference does not affect the predicted proportions).
    Results are returned as a table with one row per sample and taxon,
    later exported as Supporting Information 21.
    """
    sub = full_results[full_results["primer"] == primer].copy()

    records = []
    for sample in sorted(sub["sample"].unique()):
        sample_data = sub[sub["sample"] == sample]
        orders_in_sample = sample_data["order"].unique()
        all_cycles = sorted(sample_data["cycles"].unique())

        if len(orders_in_sample) < 2 or len(all_cycles) < 3:
            continue

        # Reference taxon: most abundant at c=20 (choice is arbitrary,
        # predicted proportions are identical regardless of reference)
        c20_data = sample_data[sample_data["cycles"] == 20]
        if c20_data.empty:
            continue
        ref_order = c20_data.loc[c20_data["read_count"].idxmax(), "order"]

        # Fit log-ratio regression for each non-reference taxon
        betas = {ref_order: 0.0}
        alphas = {ref_order: 0.0}
        for order in orders_in_sample:
            if order == ref_order:
                continue
            log_ratios = []
            valid_cycles = []
            for c in all_cycles:
                c_data = sample_data[sample_data["cycles"] == c]
                ri = c_data.loc[c_data["order"] == order, "read_count"].values
                rr = c_data.loc[c_data["order"] == ref_order, "read_count"].values
                if len(ri) and len(rr) and ri[0] > 0 and rr[0] > 0:
                    log_ratios.append(np.log(ri[0] / rr[0]))
                    valid_cycles.append(c)
            if len(valid_cycles) >= 3:
                result = linregress(valid_cycles, log_ratios)
                betas[order] = result.intercept
                alphas[order] = result.slope
            else:
                betas[order] = np.nan
                alphas[order] = np.nan

        # Convert intercepts to predicted proportions via softmax
        valid_orders = [
            o for o in orders_in_sample if not np.isnan(betas.get(o, np.nan))
        ]
        if len(valid_orders) < 2:
            continue

        exp_betas = {o: np.exp(betas[o]) for o in valid_orders}
        total_eb = sum(exp_betas.values())

        for o in valid_orders:
            records.append(
                {
                    "sample": sample,
                    "order": o,
                    "ref_species": ref_order,
                    "E_cycle_cal": np.exp(alphas[o]),
                    "pred_copies_cycle_cal": exp_betas[o] / total_eb,
                }
            )

    return pd.DataFrame(records)


def predict_copies_model_comparison(full_results: pd.DataFrame) -> None:
    """Compare four approaches to estimating taxon abundances from metabarcoding reads.

    The central question of this study: can we correct for amplification bias
    in metabarcoding data to obtain better estimates of true taxon abundances
    (measured here as relative mtDNA copy numbers by dPCR)?

    Four approaches are compared:
      1. No correction — use raw read proportions as the abundance estimate.
         This is the standard metabarcoding output, no additional data needed.
      2. Cycle calibration (Shelton et al. 2023) — uses the change in read
         ratios across PCR cycle counts within each sample to infer starting
         copy proportions. No external calibration data needed.
      3. Cycle-dependent bias model (Shelton exponential model) — uses 14
         two-species calibration communities to estimate a per-cycle efficiency
         factor E_i for each taxon. Correction is applied as reads / E_i^c,
         so the correction strength changes with cycle count.
      4. Constant bias model (this study) — also uses the 14 calibration
         communities, but assumes the bias is the same at every cycle count.
         Correction is applied as reads / E_i (no cycle dependency).

    The 14 calibration communities each contain exactly one target taxon plus
    Blattodea as the reference (Blattodea E_i = 1 by definition). The
    remaining 67 communities are used as out-of-sample test data for approaches
    3 and 4. Both E_i values are estimated from measurements at c=20 only,
    making the comparison fair.

    Outputs: Figure 3, Figure 4, Table 2, Supporting Information 24, 25, 26,
    27, and 28. Supporting Information 27 and 28 cover BF3 (constant bias
    model only).
    """
    primer = "fwh2"
    sub = full_results[full_results["primer"] == primer].copy()

    # ----------------------------------------------------------------
    # Step 1: Identify calibration communities and estimate E_i
    # ----------------------------------------------------------------
    # The 14 calibration communities each contain exactly two taxa:
    # one target insect order plus Blattodea as the reference.
    estimation_samples = []
    for sample, grp in sub[sub["cycles"] == 20].groupby("sample"):
        orders = grp["order"].unique()
        if len(orders) == 2 and "Blattodea" in orders:
            estimation_samples.append(sample)

    # For each calibration sample, compute the odds ratio at c=20:
    #   odds = (reads_i / reads_Blattodea) / (copies_i / copies_Blattodea)
    # This measures how much taxon i is over- or under-represented in
    # reads relative to what its copy numbers would predict.
    # Constant bias model: E_i = mean(odds) — the same factor at every cycle.
    # Cycle-dependent model: E_i = mean(odds^(1/20)) — a per-cycle factor
    #   that, when raised to the power of cycle count c, gives the total bias.
    # Both are calibrated from the same data (c=20 only) for a fair comparison.
    e_orig_per_taxon = {}
    e_exp_per_taxon = {}
    for sample in estimation_samples:
        sd = sub[(sub["sample"] == sample) & (sub["cycles"] == 20)].set_index("order")
        other = [o for o in sd.index if o != "Blattodea"][0]
        rr_i = sd.loc[other, "relative_reads"]
        rc_i = sd.loc[other, "relative copies"]
        rr_b = sd.loc["Blattodea", "relative_reads"]
        rc_b = sd.loc["Blattodea", "relative copies"]
        if rr_i == 0 or rr_i == 1 or rc_i == 0:
            continue
        odds = (rr_i * rc_b) / (rc_i * rr_b)
        e_orig_per_taxon.setdefault(other, []).append(odds)
        e_exp_per_taxon.setdefault(other, []).append(odds ** (1.0 / 20))

    e_orig = {o: np.mean(v) for o, v in e_orig_per_taxon.items()}
    e_orig["Blattodea"] = 1.0  # Blattodea is the reference; its E_i is 1 by definition
    e_exp = {o: np.mean(v) for o, v in e_exp_per_taxon.items()}
    e_exp["Blattodea"] = 1.0

    # ----------------------------------------------------------------
    # Step 1b: Estimate E_i for BF3 (constant bias model, c=20 only)
    # ----------------------------------------------------------------
    # Uses the same 14 calibration communities but the BF3 read columns.
    sub_bf3 = full_results[full_results["primer"] == "bf3"].copy()
    e_orig_bf3_per_taxon = {}
    for sample in estimation_samples:
        sd = sub_bf3[
            (sub_bf3["sample"] == sample) & (sub_bf3["cycles"] == 20)
        ].set_index("order")
        if len(sd) < 2 or "Blattodea" not in sd.index:
            continue
        other = [o for o in sd.index if o != "Blattodea"][0]
        rr_i = sd.loc[other, "relative_reads"]
        rc_i = sd.loc[other, "relative copies"]
        rr_b = sd.loc["Blattodea", "relative_reads"]
        rc_b = sd.loc["Blattodea", "relative copies"]
        if rr_i == 0 or rr_i == 1 or rc_i == 0:
            continue
        odds = (rr_i * rc_b) / (rc_i * rr_b)
        e_orig_bf3_per_taxon.setdefault(other, []).append(odds)
    e_orig_bf3 = {o: np.mean(v) for o, v in e_orig_bf3_per_taxon.items()}
    e_orig_bf3["Blattodea"] = 1.0

    # ----------------------------------------------------------------
    # Step 2a: Run cycle calibration (Shelton et al. 2023) for all samples
    # ----------------------------------------------------------------
    # This approach does not need external calibration communities —
    # it uses within-sample variation across cycle counts instead.
    cycle_cal_table = compute_cycle_calibration(full_results, primer=primer)
    cycle_cal_n0 = {
        (row.sample, row.order): row.pred_copies_cycle_cal
        for row in cycle_cal_table.itertuples()
    }

    # ----------------------------------------------------------------
    # Step 2b: Compute predictions for all four approaches
    # ----------------------------------------------------------------
    # prediction_samples: the 67 communities not used for calibration.
    # Bias-corrected models (3 & 4) are only evaluated on these samples
    # to ensure the comparison is out-of-sample (i.e. the model has never
    # seen these communities during calibration).
    prediction_samples = set(sub["sample"].unique()) - set(estimation_samples)
    all_samples = set(sub["sample"].unique())

    records = []
    for sample in sorted(all_samples):
        for cycles in sorted(sub["cycles"].unique()):
            sample_data = sub[(sub["sample"] == sample) & (sub["cycles"] == cycles)]
            if sample_data.empty:
                continue

            orders = sample_data["order"].values
            reads = sample_data.set_index("order")["relative_reads"]
            obs_copies = sample_data.set_index("order")["relative copies"]

            for order in orders:
                obs_val = obs_copies[order]

                # Approach 1: no correction — reads used as-is
                records.append(
                    {
                        "sample": sample,
                        "cycles": cycles,
                        "order": order,
                        "method": "no correction",
                        "obs": obs_val,
                        "pred": reads[order],
                        "abs_error": abs(reads[order] - obs_val),
                    }
                )

                # Approach 2: cycle calibration — one predicted proportion per
                # (sample, order), independent of cycle count
                cc_pred = cycle_cal_n0.get((sample, order), np.nan)
                if not np.isnan(cc_pred):
                    records.append(
                        {
                            "sample": sample,
                            "cycles": cycles,
                            "order": order,
                            "method": "cycle calibration",
                            "obs": obs_val,
                            "pred": cc_pred,
                            "abs_error": abs(cc_pred - obs_val),
                        }
                    )

                # Approaches 3 & 4: bias-corrected models, out-of-sample only
                if sample in prediction_samples:
                    # Approach 3 (constant bias): divide each taxon's reads by
                    # its E_i, then re-normalise so proportions sum to 1
                    corr_ne = {o: reads[o] / e_orig.get(o, 1.0) for o in orders}
                    t_ne = sum(corr_ne.values())
                    if t_ne > 0:
                        pred_ne = corr_ne[order] / t_ne
                        records.append(
                            {
                                "sample": sample,
                                "cycles": cycles,
                                "order": order,
                                "method": "non-exponential",
                                "obs": obs_val,
                                "pred": pred_ne,
                                "abs_error": abs(pred_ne - obs_val),
                            }
                        )
                    # Approach 4 (cycle-dependent bias): divide by E_i raised
                    # to the power of the actual cycle count used
                    corr_exp = {
                        o: reads[o] / (e_exp.get(o, 1.0) ** cycles) for o in orders
                    }
                    t_exp = sum(corr_exp.values())
                    if t_exp > 0:
                        pred_exp = corr_exp[order] / t_exp
                        records.append(
                            {
                                "sample": sample,
                                "cycles": cycles,
                                "order": order,
                                "method": "cycle-dependent",
                                "obs": obs_val,
                                "pred": pred_exp,
                                "abs_error": abs(pred_exp - obs_val),
                            }
                        )

    df = pd.DataFrame(records)
    df_common = df[df["sample"].isin(prediction_samples)]

    # ----------------------------------------------------------------
    # Step 5b: Figure 3 — no correction vs. cycle calibration only
    # ----------------------------------------------------------------
    # Both methods require no external calibration data, so all 81
    # communities are used here (not just the 67 out-of-sample ones
    # needed for the fair 3-way comparison above).
    # Shared y-axis since both panels are on the same scale (0-1).
    # RMSE and n are reported in the figure caption, not on the plot.
    figure3_titles = {
        "no correction": "A)  No correction (all cycles)",
        "cycle calibration": "B)  Cycle calibration",
    }

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    figure3_methods = ["no correction", "cycle calibration"]

    for ax, method_name in zip(axes3, figure3_methods):
        md = df[df["method"] == method_name]
        if method_name == "cycle calibration":
            md = md.drop_duplicates(subset=["sample", "order"])
        for order in ORDER_NAMES:
            od = md[md["order"] == order]
            if od.empty:
                continue
            ax.scatter(
                od["obs"],
                od["pred"],
                color=ORDER_PALETTE[order],
                label=order,
                alpha=0.5,
                s=15,
                edgecolor="none",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed relative copies (dPCR)")
        ax.set_title(figure3_titles[method_name], fontsize=13)

        obs_v, pred_v = md["obs"].values, md["pred"].values
        ss_res = np.sum((obs_v - pred_v) ** 2)
        ss_tot = np.sum((obs_v - obs_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.annotate(
            f"R² = {r2:.3f}",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    axes3[0].set_ylabel("Predicted relative copies")
    axes3[1].legend(fontsize=8, loc="lower right", markerscale=1.5)

    fig3.tight_layout()
    plt.savefig(FIGURES_DIR / "figure 3.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure 3.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # ----------------------------------------------------------------
    # Step 6: Aitchison distance per sample
    # ----------------------------------------------------------------
    # Aitchison distance is the appropriate metric for compositional data
    # (Aitchison 1986). It measures the Euclidean distance between two
    # compositions in centered log-ratio (CLR) space. Unlike R² or RMSE,
    # it respects the multiplicative structure of proportions: a shift
    # from 0.01 to 0.02 (2x) weighs the same as 0.5 to 1.0 (2x).
    # Shelton et al. (2023) use this metric for their model comparison.
    def aitchison_distance(x, y):
        x = np.maximum(np.array(x, dtype=float), 1e-10)
        y = np.maximum(np.array(y, dtype=float), 1e-10)
        clr_x = np.log(x) - np.log(x).mean()
        clr_y = np.log(y) - np.log(y).mean()
        return np.sqrt(np.sum((clr_x - clr_y) ** 2))

    # ----------------------------------------------------------------
    # Aitchison distance across all cycles (4 series)
    # ----------------------------------------------------------------
    aitchison_all_cycles = []
    for sample in sorted(prediction_samples):
        sd20 = sub[(sub["sample"] == sample) & (sub["cycles"] == 20)]
        if sd20.empty:
            continue
        orders = sorted(sd20["order"].values)
        obs_vec = [
            sd20.loc[sd20["order"] == o, "relative copies"].values[0] for o in orders
        ]
        if any(v == 0 for v in obs_vec):
            continue

        cc_vec = [cycle_cal_n0.get((sample, o), np.nan) for o in orders]
        d_cc = (
            np.nan
            if any(np.isnan(v) for v in cc_vec)
            else aitchison_distance(obs_vec, cc_vec)
        )

        for cycles in sorted(sub["cycles"].unique()):
            sd = sub[(sub["sample"] == sample) & (sub["cycles"] == cycles)]
            if sd.empty or set(sd["order"]) != set(orders):
                continue

            nc_vec = [
                sd.loc[sd["order"] == o, "relative_reads"].values[0] for o in orders
            ]
            # Skip (sample, cycle) combinations with a zero-read taxon:
            # 0 -> exactly 0 under correction, extreme CLR floor artifacts.
            if any(v == 0 for v in nc_vec):
                continue

            reads = sd.set_index("order")["relative_reads"]

            corr_ne = {o: reads[o] / e_orig.get(o, 1.0) for o in orders}
            t_ne = sum(corr_ne.values())
            if t_ne == 0:
                continue
            ne_vec = [corr_ne[o] / t_ne for o in orders]

            corr_exp = {o: reads[o] / (e_exp.get(o, 1.0) ** cycles) for o in orders}
            t_exp = sum(corr_exp.values())
            if t_exp == 0:
                continue
            exp_vec = [corr_exp[o] / t_exp for o in orders]

            aitchison_all_cycles.append(
                {
                    "sample": sample,
                    "cycles": cycles,
                    "d_nocorr": aitchison_distance(obs_vec, nc_vec),
                    "d_ne": aitchison_distance(obs_vec, ne_vec),
                    "d_exp": aitchison_distance(obs_vec, exp_vec),
                    "d_cc": d_cc,
                }
            )

    df_ait_all = pd.DataFrame(aitchison_all_cycles)

    print("Median Aitchison distances (prediction samples, all cycles):")
    cc_dedup_med = df_ait_all.drop_duplicates(subset=["sample"])
    print(f"  No correction:              {df_ait_all['d_nocorr'].median():.4f}")
    print(f"  Cycle calibration:          {cc_dedup_med['d_cc'].median():.4f}")
    print(f"  Cycle-dependent bias model: {df_ait_all['d_exp'].median():.4f}")
    print(f"  Constant bias model:        {df_ait_all['d_ne'].median():.4f}")

    sample_order_x = sorted(df_ait_all["sample"].unique())
    x_pos = {s: i for i, s in enumerate(sample_order_x)}

    # ----------------------------------------------------------------
    # Figure 4: 2×2 gridspec + full-width bottom panel
    # Top row: cycle-dependent bias model | constant bias model (scatter)
    # Bottom: 4-series Aitchison distance per sample (all cycles)
    # ----------------------------------------------------------------
    fig4 = plt.figure(figsize=(13, 13))
    gs = fig4.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.18, wspace=0.06)

    figure4_specs = [
        ("cycle-dependent", "A) Cycle-dependent bias model", gs[0, 0]),
        ("non-exponential", "B) Constant bias model", gs[0, 1]),
    ]
    axes4_top = []
    for method_name, title, gs_pos in figure4_specs:
        ax = fig4.add_subplot(gs_pos)
        axes4_top.append(ax)
        md = df_common[df_common["method"] == method_name]
        for order in ORDER_NAMES:
            od = md[md["order"] == order]
            if od.empty:
                continue
            ax.scatter(
                od["obs"],
                od["pred"],
                color=ORDER_PALETTE[order],
                label=order,
                alpha=0.5,
                s=15,
                edgecolor="none",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed relative copies (dPCR)")
        ax.set_title(title, fontsize=13)

        obs_v, pred_v = md["obs"].values, md["pred"].values
        ss_res = np.sum((obs_v - pred_v) ** 2)
        ss_tot = np.sum((obs_v - obs_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.annotate(
            f"R² = {r2:.3f}",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    axes4_top[0].set_ylabel("Predicted relative copies")
    axes4_top[1].legend(fontsize=8, loc="lower right", markerscale=1.5)

    # Bottom panel: Aitchison distance, 4 series
    ax_bottom = fig4.add_subplot(gs[1, :])

    ax_bottom.scatter(
        [x_pos[s] for s in df_ait_all["sample"]],
        df_ait_all["d_nocorr"],
        color="#AAAAAA",
        s=18,
        alpha=0.5,
        edgecolors="none",
        zorder=2,
        label="No correction",
    )
    cc_dedup = df_ait_all.drop_duplicates(subset=["sample"])
    ax_bottom.scatter(
        [x_pos[s] for s in cc_dedup["sample"]],
        cc_dedup["d_cc"],
        color="#E69F00",
        s=30,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
        label="Cycle calibration",
    )
    ax_bottom.scatter(
        [x_pos[s] for s in df_ait_all["sample"]],
        df_ait_all["d_exp"],
        color="#009E73",
        s=18,
        alpha=0.5,
        edgecolors="none",
        zorder=3,
        label="Cycle-dependent bias model",
    )
    ax_bottom.scatter(
        [x_pos[s] for s in df_ait_all["sample"]],
        df_ait_all["d_ne"],
        color="#0072B2",
        s=18,
        alpha=0.5,
        edgecolors="none",
        zorder=3,
        label="Constant bias model",
    )

    # Horizontal median lines per method
    for color, col in [
        ("#AAAAAA", "d_nocorr"),
        ("#009E73", "d_exp"),
        ("#0072B2", "d_ne"),
    ]:
        ax_bottom.axhline(
            df_ait_all[col].median(),
            color=color,
            lw=1.5,
            linestyle="--",
            alpha=0.85,
            zorder=1,
        )
    ax_bottom.axhline(
        cc_dedup["d_cc"].median(),
        color="#E69F00",
        lw=1.5,
        linestyle="--",
        alpha=0.85,
        zorder=1,
    )

    ax_bottom.text(
        -0.02,
        1.02,
        "C)",
        transform=ax_bottom.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_bottom.set_ylabel("Aitchison distance to observed copies")
    ax_bottom.set_xticks(range(len(sample_order_x)))
    ax_bottom.set_xticklabels(
        [str(s) for s in sample_order_x], rotation=90, fontsize=10
    )
    ax_bottom.set_xlim(-1, len(sample_order_x))
    ax_bottom.set_ylim(0, None)
    ax_bottom.legend(loc="upper right", fontsize=9)

    plt.savefig(FIGURES_DIR / "figure 4.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure 4.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # ----------------------------------------------------------------
    # Table 2: Mean E_i per taxon for both bias models
    # ----------------------------------------------------------------
    pd.DataFrame(
        [
            {
                "Taxon": order,
                "E_i, cycle-dependent bias": e_exp.get(order, np.nan),
                "E_i, constant bias": e_orig.get(order, np.nan),
            }
            for order in ORDER_NAMES
        ]
    ).to_excel(TABLES_DIR / "table 2.xlsx", index=False)

    # ----------------------------------------------------------------
    # Supporting Information 24: scatter per cycle count, 2 models (fwh2)
    # Rows = cycle count (c=6..20), Cols = cycle-dependent | constant bias
    # Prediction samples only (same as Figure 4)
    # ----------------------------------------------------------------
    cycles_list = sorted(sub["cycles"].unique())
    n_cyc = len(cycles_list)
    fig18, axes18 = plt.subplots(
        n_cyc,
        2,
        figsize=(10, n_cyc * 4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.06},
    )
    si18_col_specs = [
        ("cycle-dependent", "Cycle-dependent bias model"),
        ("non-exponential", "Constant bias model"),
    ]
    for row_idx, c in enumerate(cycles_list):
        for col_idx, (method_name, col_title) in enumerate(si18_col_specs):
            ax = axes18[row_idx, col_idx]
            md = df_common[
                (df_common["method"] == method_name) & (df_common["cycles"] == c)
            ]
            for order in ORDER_NAMES:
                od = md[md["order"] == order]
                if od.empty:
                    continue
                ax.scatter(
                    od["obs"],
                    od["pred"],
                    color=ORDER_PALETTE[order],
                    label=order if (row_idx == 0 and col_idx == 1) else None,
                    alpha=0.5,
                    s=12,
                    edgecolor="none",
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10)
            if row_idx == n_cyc - 1:
                ax.set_xlabel("Observed relative copies (dPCR)", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Predicted", fontsize=8)
                ax.annotate(
                    f"c = {c}",
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                    annotation_clip=False,
                )
            if len(md) > 0:
                obs_v, pred_v = md["obs"].values, md["pred"].values
                ss_res = np.sum((obs_v - pred_v) ** 2)
                ss_tot = np.sum((obs_v - obs_v.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                ax.annotate(
                    f"$R^2 = {r2:.2f}$",
                    xy=(0.05, 0.90),
                    xycoords="axes fraction",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
    axes18[0, 1].legend(fontsize=7, loc="lower right", markerscale=1.5)
    plt.savefig(SI_DIR / "supporting information 24.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(SI_DIR / "supporting information 24.png", dpi=150, bbox_inches="tight")
    plt.close(fig18)

    # ----------------------------------------------------------------
    # Supporting Information 25: SI 21 extended with bias model predictions
    # Adds E_i per taxon, predicted copies for both models and per-sample
    # Aitchison distances for all four approaches
    # Predictions computed for ALL fwh2 samples (in- and out-of-sample)
    # ----------------------------------------------------------------
    # Bias predictions for all (sample, cycles) in fwh2 data
    pred_exp_all = {}
    pred_ne_all = {}
    for (smp, cyc), grp in sub.groupby(["sample", "cycles"]):
        idx = grp.set_index("order")
        orders_grp = idx.index.tolist()
        corr_exp = {
            o: idx.loc[o, "relative_reads"] / (e_exp.get(o, 1.0) ** cyc)
            for o in orders_grp
        }
        t_exp = sum(corr_exp.values())
        corr_ne = {
            o: idx.loc[o, "relative_reads"] / e_orig.get(o, 1.0) for o in orders_grp
        }
        t_ne = sum(corr_ne.values())
        for o in orders_grp:
            pred_exp_all[(smp, cyc, o)] = corr_exp[o] / t_exp if t_exp > 0 else np.nan
            pred_ne_all[(smp, cyc, o)] = corr_ne[o] / t_ne if t_ne > 0 else np.nan

    ct = cycle_cal_table.copy()
    ct["primer"] = primer
    si19 = full_results.merge(ct, on=["sample", "order", "primer"], how="left")

    # E_i values are fwh2-specific; bf3 rows get NaN
    si19["E_i, cycle-dependent bias"] = np.where(
        si19["primer"] == primer,
        si19["order"].map(e_exp),
        np.nan,
    )
    si19["E_i, constant bias"] = np.where(
        si19["primer"] == primer,
        si19["order"].map(e_orig),
        np.nan,
    )
    si19["pred_copies_cycle_dep_bias"] = [
        pred_exp_all.get((r.sample, r.cycles, r.order), np.nan)
        for r in si19.itertuples()
    ]
    si19["pred_copies_constant_bias"] = [
        pred_ne_all.get((r.sample, r.cycles, r.order), np.nan)
        for r in si19.itertuples()
    ]

    # Aitchison distances: one value per (sample, cycles), repeated across orders
    # Only available for prediction samples (calibration samples get NaN)
    ait_cols = df_ait_all[["sample", "cycles", "d_nocorr", "d_exp", "d_ne", "d_cc"]].rename(
        columns={
            "d_nocorr": "aitchison_no_correction",
            "d_exp":    "aitchison_cycle_dep_bias",
            "d_ne":     "aitchison_constant_bias",
            "d_cc":     "aitchison_cycle_calibration",
        }
    )
    si19 = si19.merge(ait_cols, on=["sample", "cycles"], how="left")

    si19.to_excel(SI_DIR / "supporting information 25.xlsx", index=False)

    # ----------------------------------------------------------------
    # Supporting Information 26: scatter per insect order, 2 models (fwh2)
    # Rows = order, Cols = cycle-dependent | constant bias
    # Points coloured by cycle count; prediction samples only
    # ----------------------------------------------------------------
    cycles_list_20 = sorted(sub["cycles"].unique())
    cycle_colors = {
        c: plt.cm.viridis(i / max(len(cycles_list_20) - 1, 1))
        for i, c in enumerate(cycles_list_20)
    }

    n_ord = len(ORDER_NAMES)
    fig20, axes20 = plt.subplots(
        n_ord,
        2,
        figsize=(10, n_ord * 4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.06},
    )
    si20_col_specs = [
        ("cycle-dependent", "Cycle-dependent bias model"),
        ("non-exponential", "Constant bias model"),
    ]
    for row_idx, order in enumerate(ORDER_NAMES):
        for col_idx, (method_name, col_title) in enumerate(si20_col_specs):
            ax = axes20[row_idx, col_idx]
            md = df_common[
                (df_common["method"] == method_name) & (df_common["order"] == order)
            ]
            for c in cycles_list_20:
                cd = md[md["cycles"] == c]
                if cd.empty:
                    continue
                ax.scatter(
                    cd["obs"],
                    cd["pred"],
                    color=cycle_colors[c],
                    label=f"c = {c}" if (row_idx == 0 and col_idx == 1) else None,
                    alpha=0.6,
                    s=12,
                    edgecolor="none",
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10)
            if row_idx == n_ord - 1:
                ax.set_xlabel("Observed relative copies (dPCR)", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Predicted", fontsize=8)
                ax.annotate(
                    order,
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    fontsize=9,
                    ha="center",
                    va="center",
                    rotation=90,
                    annotation_clip=False,
                )
            if len(md) > 0:
                obs_v, pred_v = md["obs"].values, md["pred"].values
                ss_res = np.sum((obs_v - pred_v) ** 2)
                ss_tot = np.sum((obs_v - obs_v.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                ax.annotate(
                    f"$R^2 = {r2:.2f}$",
                    xy=(0.05, 0.90),
                    xycoords="axes fraction",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
    axes20[0, 1].legend(fontsize=7, loc="lower right", markerscale=1.5)
    plt.savefig(SI_DIR / "supporting information 26.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(SI_DIR / "supporting information 26.png", dpi=150, bbox_inches="tight")
    plt.close(fig20)

    # ----------------------------------------------------------------
    # Supporting Information 27: E_i comparison table (fwh2 vs BF3)
    # ----------------------------------------------------------------
    si21_rows = [
        {
            "order": order,
            "E_i fwh2 (constant bias model)": round(e_orig.get(order, np.nan), 4),
            "E_i BF3 (constant bias model)": round(e_orig_bf3.get(order, np.nan), 4),
        }
        for order in ORDER_NAMES
    ]
    pd.DataFrame(si21_rows).to_excel(SI_DIR / "supporting information 27.xlsx", index=False)

    # ----------------------------------------------------------------
    # Supporting Information 28: BF3 scatter — uncorrected vs. corrected
    # (c=20 only, constant bias model, same 14 calibration samples excluded)
    # ----------------------------------------------------------------
    # Prediction communities for BF3 (same 14 calibration samples excluded)
    sub_bf3 = full_results[full_results["primer"] == "bf3"].copy()
    prediction_samples_bf3 = set(sub_bf3["sample"].unique()) - set(estimation_samples)

    records_bf3 = []
    for sample in sorted(prediction_samples_bf3):
        sample_data = sub_bf3[
            (sub_bf3["sample"] == sample) & (sub_bf3["cycles"] == 20)
        ]
        if sample_data.empty:
            continue
        orders     = sample_data["order"].values
        reads      = sample_data.set_index("order")["relative_reads"]
        obs_copies = sample_data.set_index("order")["relative copies"]
        corr  = {o: reads[o] / e_orig_bf3.get(o, 1.0) for o in orders}
        total = sum(corr.values())
        for order in orders:
            records_bf3.append({
                "sample": sample,
                "order":  order,
                "obs":    obs_copies[order],
                "pred_uncorrected": reads[order],
                "pred_corrected":   corr[order] / total if total > 0 else np.nan,
            })

    df_bf3 = pd.DataFrame(records_bf3)

    def _pred_r2(obs, pred):
        obs, pred = np.array(obs), np.array(pred)
        mask = ~(np.isnan(obs) | np.isnan(pred))
        ss_res = np.sum((obs[mask] - pred[mask]) ** 2)
        ss_tot = np.sum((obs[mask] - obs[mask].mean()) ** 2)
        return 1 - ss_res / ss_tot

    r2_unc_bf3 = _pred_r2(df_bf3["obs"], df_bf3["pred_uncorrected"])
    r2_cor_bf3 = _pred_r2(df_bf3["obs"], df_bf3["pred_corrected"])

    fig22, axes22 = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    panel_specs22 = [
        ("pred_uncorrected", "No correction",        r2_unc_bf3),
        ("pred_corrected",   "Constant bias model",  r2_cor_bf3),
    ]
    for ax, (col, title, r2_val) in zip(axes22, panel_specs22):
        for order in ORDER_NAMES:
            sub = df_bf3[df_bf3["order"] == order]
            ax.scatter(
                sub["obs"], sub[col],
                color=ORDER_PALETTE[order], label=order,
                s=22, alpha=0.6, linewidths=0,
            )
        lim_max = max(
            df_bf3["obs"].max(),
            df_bf3["pred_corrected"].max(),
            df_bf3["pred_uncorrected"].max(),
        ) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, zorder=0)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Observed relative copies", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.text(
            0.04, 0.96, f"R² = {r2_val:.2f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    axes22[0].set_ylabel("Predicted relative copies", fontsize=11)
    axes22[1].legend(
        fontsize=8, loc="lower right", markerscale=1.5,
        frameon=True, edgecolor="black", framealpha=0.9,
    )
    fig22.suptitle("BF3 — constant bias correction", fontsize=12)
    plt.tight_layout()
    plt.savefig(SI_DIR / "supporting information 28.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(SI_DIR / "supporting information 28.png", dpi=150, bbox_inches="tight")
    plt.close(fig22)


def supporting_information_9_1(supporting_information_10: Path) -> None:
    """Create the qPCR figure of the second-step PCR (Figure 9.1).

    The second PCR step uses the same universal tail primers for all taxa. To
    check that it amplifies all taxa with (nearly) the same efficiency, first-step
    amplicon pools of the five taxa were measured by qPCR (18 wells per taxon).
    Panel A shows the baseline-corrected amplification curves (thin lines: single
    wells, thick lines: median per taxon). Panel B shows the PCR efficiency E per
    well (LinRegPCR) per taxon as box plot with single wells; E = 2 is a perfect
    doubling per cycle. Wells without an efficiency estimate (no-template
    controls) are not shown. Input: Supporting Information 10.
    """
    # --- Data loading ---
    cycles = [str(c) for c in range(1, 41)]
    curves = pd.read_excel(supporting_information_10, sheet_name="Baseline-corrected")
    curves.columns = [str(c) for c in curves.columns]
    curves = curves[curves["Taxon"].isin(ORDER_NAMES)]
    wells = pd.read_excel(supporting_information_10, sheet_name="Per-well results")
    wells = wells.rename(columns={"PCR efficiency E (per well)": "E"})
    wells = wells[wells["Taxon"].isin(ORDER_NAMES)].dropna(subset=["E"])

    groups = [wells.loc[wells["Taxon"] == t, "E"].values for t in ORDER_NAMES]

    fig, (ax_curves, ax_eff) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # --- Panel A: amplification plot (log scale) ---
    x = np.arange(1, 41)
    for _, row in curves.iterrows():
        ax_curves.plot(
            x,
            row[cycles].astype(float).values,
            color=ORDER_PALETTE[row["Taxon"]],
            lw=1,
            alpha=0.6,
        )
    for taxon in ORDER_NAMES:
        median_curve = curves.loc[curves["Taxon"] == taxon, cycles].astype(float).median()
        ax_curves.plot(x, median_curve.values, color=ORDER_PALETTE[taxon], lw=3.5, alpha=0.85)
    ax_curves.set_yscale("log")
    ax_curves.set_ylim(1e-2, 12)
    ax_curves.set_xlim(10, 40)
    ax_curves.set_xlabel("Cycle")
    ax_curves.set_ylabel("Baseline-corrected fluorescence (Rn, log scale)")
    ax_curves.set_title("A)", fontsize=12, loc="left")
    ax_curves.legend(
        handles=[
            plt.Line2D([0], [0], color=ORDER_PALETTE[t], lw=2, label=t)
            for t in ORDER_NAMES
        ],
        frameon=False,
        loc="lower right",
    )

    # --- Panel B: PCR efficiency per well ---
    sns.boxplot(
        data=wells,
        x="Taxon",
        y="E",
        order=ORDER_NAMES,
        hue="Taxon",
        palette=ORDER_PALETTE,
        legend=False,
        width=0.55,
        showfliers=False,
        boxprops=dict(alpha=0.45, edgecolor="0.25"),
        medianprops=dict(color="0.15"),
        whiskerprops=dict(color="0.25"),
        capprops=dict(color="0.25"),
        ax=ax_eff,
    )
    sns.stripplot(
        data=wells,
        x="Taxon",
        y="E",
        order=ORDER_NAMES,
        hue="Taxon",
        palette=ORDER_PALETTE,
        legend=False,
        size=4.5,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
        jitter=0.18,
        ax=ax_eff,
    )
    ax_eff.axhline(2, color="0.5", ls="--", lw=0.8)
    ax_eff.set_ylim(1, 2.03)
    ax_eff.set_xlabel("")
    ax_eff.set_ylabel("PCR efficiency E (per well, LinRegPCR)")
    ax_eff.set_xticks(range(len(ORDER_NAMES)))
    ax_eff.set_xticklabels(
        [f"{t}\n(n = {len(g)})" for t, g in zip(ORDER_NAMES, groups)]
    )
    ax_eff.set_title("B)", fontsize=12, loc="left")
    sns.despine(ax=ax_curves)
    sns.despine(ax=ax_eff)

    plt.tight_layout()
    plt.savefig(SI_DIR / "supporting information 9_1.pdf")
    plt.savefig(SI_DIR / "supporting information 9_1.png", dpi=300)
    plt.close(fig)


def supporting_information_9_2(supporting_information_10: Path) -> None:
    """Create the efficiency table of the second-step PCR (Table 9.1).

    Summarises the per-well PCR efficiency E (LinRegPCR) of the qPCR on
    first-step amplicon pools as mean, minimum and maximum per taxon. Wells
    without an efficiency estimate (no-template controls) are not included.
    Input: Supporting Information 10.
    """
    wells = pd.read_excel(supporting_information_10, sheet_name="Per-well results")
    wells = wells.rename(columns={"PCR efficiency E (per well)": "E"})
    wells = wells[wells["Taxon"].isin(ORDER_NAMES)].dropna(subset=["E"])

    table = (
        wells.groupby("Taxon")["E"]
        .agg(["mean", "min", "max"])
        .reindex(ORDER_NAMES)
        .round(3)
        .rename(columns={"mean": "mean E", "min": "min E", "max": "max E"})
        .reset_index()
    )
    table.to_excel(SI_DIR / "supporting information 9_2.xlsx", index=False)
    print(table.to_string(index=False))

    # --- Statistics quoted in the text of Supporting Information 9 (printed only) ---
    # Kruskal-Wallis test across taxa, epsilon squared = H / (N - 1), and a
    # Shapiro-Wilk test of the residuals (value minus group mean) as justification
    # for the rank-based test.
    groups = [wells.loc[wells["Taxon"] == t, "E"].values for t in ORDER_NAMES]
    h_stat, p_value = kruskal(*groups)
    epsilon_squared = h_stat / (len(wells) - 1)
    residuals = np.concatenate([g - g.mean() for g in groups])
    w_stat, p_shapiro = shapiro(residuals)
    print(
        f"Kruskal-Wallis: H({len(groups) - 1}) = {h_stat:.1f}, p = {p_value:.3f}, "
        f"epsilon squared = {epsilon_squared:.2f}"
    )
    print(f"Shapiro-Wilk (residuals): W = {w_stat:.3f}, p = {p_shapiro:.3g}")

    # Fold difference in product between the taxa with the highest and the lowest
    # mean efficiency after the minimum and maximum number of second-step PCR
    # cycles (45 total cycles minus 20 and 6 first-step cycles = 25 and 39)
    means = wells.groupby("Taxon")["E"].mean()
    ratio = means.max() / means.min()
    print(
        f"Mean E {means.idxmax()} / {means.idxmin()} = {ratio:.4f}; difference of means "
        f"{means.max() - means.min():.3f}; fold difference after 25 cycles: "
        f"{ratio**25:.2f}, after 39 cycles: {ratio**39:.2f}"
    )


def read_primer_binding_sites(supporting_information_3: Path) -> tuple:
    """Return the fwh2 primers, the IUPAC code table and the template binding sites.

    The Sanger consensus sequences of the single specimens (Supporting
    Information 3) are the fwh2 amplicons including both primer binding sites:
    the first 26 nucleotides are the binding site of the forward primer fwhF2,
    the last 23 nucleotides (reverse complement) the binding site of the reverse
    primer fwhR2n. For each taxon the most frequent site among the specimens is
    used. The sites are returned in primer sense, keyed by (taxon, primer name).
    """
    primers = {
        "fwhF2": "GGDACWGGWTGAACWGTWTAYCCHCC",
        "fwhR2n": "GTRATWGCHCCDGCTARWACWGG",
    }
    iupac = {
        "A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "W": "AT",
        "S": "CG", "K": "GT", "M": "AC", "D": "AGT", "H": "ACT", "B": "CGT",
        "V": "ACG", "N": "ACGT",
    }
    complement = str.maketrans("ACGT", "TGCA")

    # --- Read the consensus sequences (header: >Taxon_nn specimen=... length=...) ---
    consensus = {taxon: [] for taxon in ORDER_NAMES}
    with open(supporting_information_3) as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                taxon = line[1:].split("_")[0]
                consensus[taxon].append("")
            elif line:
                consensus[taxon][-1] += line.upper()

    # --- Binding site per taxon: most frequent site among the specimens ---
    sites = {}
    for taxon, seqs in consensus.items():
        sites[(taxon, "fwhF2")] = pd.Series([s[:26] for s in seqs]).mode()[0]
        sites[(taxon, "fwhR2n")] = pd.Series(
            [s[-23:][::-1].translate(complement) for s in seqs]
        ).mode()[0]
    return primers, iupac, sites


def supporting_information_29(supporting_information_3: Path) -> None:
    """Create the primer binding site alignment (Supporting Information 29).

    The Sanger consensus sequences of the single specimens (Supporting
    Information 3) are the fwh2 amplicons including both primer binding sites:
    the first 26 nucleotides are the binding site of the forward primer fwhF2,
    the last 23 nucleotides (reverse complement) the binding site of the reverse
    primer fwhR2n. For each taxon the most frequent site is aligned to the
    degenerate primer. Each position is shaded by the share of the 288 primer
    variants that match the template base: all (no shading), 1/2 or 1/3
    (degenerate position), none (the base is not covered by any variant). The
    last five nucleotides at the 3' end are marked.
    """
    primers, iupac, sites = read_primer_binding_sites(supporting_information_3)
    shades = {"half": "#fdd49e", "third": "#fc8d59", "none": "#d62728"}

    def cell_shade(code, base):
        """Fill colour of a cell: None if all primer variants match the base."""
        if base not in iupac[code]:
            return shades["none"]
        n_bases = len(iupac[code])
        return None if n_bases == 1 else shades["half"] if n_bases == 2 else shades["third"]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4))
    for ax, (panel, (primer_name, primer)) in zip(axes, zip("AB", primers.items())):
        length = len(primer)
        ax.set_xlim(-5.6, length + 0.3)
        ax.set_ylim(-(len(ORDER_NAMES) + 1.2), 1.4)
        ax.axis("off")
        ax.set_title(f"{panel})  {primer_name}  (5' → 3')", loc="left", fontsize=10)

        # primer row (degenerate positions in bold) with position numbers
        for i, code in enumerate(primer):
            ax.text(
                i + 0.5, 0, code, ha="center", va="center", family="monospace",
                fontsize=9, fontweight="bold",
                color="0.1" if len(iupac[code]) > 1 else "0.45",
            )
            if (i + 1) % 5 == 0 or i == 0:
                ax.text(i + 0.5, 0.95, str(i + 1), ha="center", va="center", fontsize=6.5, color="0.4")
        ax.text(-0.3, 0, "Primer", ha="right", va="center", fontsize=8, fontweight="bold")

        # one row per taxon
        for row, taxon in enumerate(ORDER_NAMES, start=1):
            y = -row
            ax.text(
                -0.3, y, taxon, ha="right", va="center", fontsize=8,
                color=ORDER_PALETTE[taxon], fontweight="bold",
            )
            for i, base in enumerate(sites[(taxon, primer_name)]):
                fill = cell_shade(primer[i], base)
                if fill:
                    ax.add_patch(plt.Rectangle((i, y - 0.42), 1, 0.84, facecolor=fill, lw=0))
                ax.text(
                    i + 0.5, y, base, ha="center", va="center", family="monospace",
                    fontsize=9, color="white" if fill == shades["none"] else "0.15",
                )

        # bracket below the last five nucleotides (3' end)
        ax.plot([length - 5, length], [-(len(ORDER_NAMES) + 0.75)] * 2, color="0.3", lw=1.2)
        ax.text(
            length - 2.5, -(len(ORDER_NAMES) + 1.15), "3' end (last 5 nt)",
            ha="center", va="center", fontsize=7, color="0.3",
        )

    # Legend in two columns, read row by row: all / 1/2 and 1/3 / none
    # (matplotlib fills legend columns first, hence the order of the handles)
    fig.legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="0.6", label="all variants\nmatch"),
            mpatches.Patch(facecolor=shades["third"], label="degenerate position:\n1/3 of variants match"),
            mpatches.Patch(facecolor=shades["half"], label="degenerate position:\n1/2 of variants match"),
            mpatches.Patch(facecolor=shades["none"], label="no variant\nmatches"),
        ],
        loc="lower center", ncol=2, fontsize=7.5, frameon=False,
        bbox_to_anchor=(0.5, 0.0), columnspacing=2.5, handlelength=1.6,
        handletextpad=0.6, labelspacing=0.8,
    )
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(SI_DIR / "supporting information 29.pdf")
    plt.savefig(SI_DIR / "supporting information 29.png", dpi=300)
    plt.close(fig)


def supporting_information_30(supporting_information_3: Path) -> None:
    """Create the mismatch table of the primer variants (Supporting Information 30).

    All 288 variants of each degenerate fwh2 primer are generated from the IUPAC
    codes and compared position by position with the template binding site of
    each taxon (Supporting Information 3). For each taxon and primer the table
    lists how many variants have 0, 1, 2, ... mismatches, followed by the minimum,
    the maximum and the mean number of mismatches per variant. All variants are
    treated as equally frequent.
    """
    primers, iupac, sites = read_primer_binding_sites(supporting_information_3)

    rows = []
    for taxon in ORDER_NAMES:
        for primer_name, primer in primers.items():
            site = sites[(taxon, primer_name)]
            variants = ["".join(v) for v in itertools.product(*[iupac[c] for c in primer])]
            mismatches = pd.Series(
                [sum(a != b for a, b in zip(variant, site)) for variant in variants]
            )
            row = {"Taxon": taxon, "Primer": primer_name}
            row["_counts"] = mismatches.value_counts().to_dict()
            row["Min"] = mismatches.min()
            row["Max"] = mismatches.max()
            row["Mean"] = round(mismatches.mean(), 1)
            rows.append(row)

    # one column per number of mismatches, up to the highest number observed
    max_mismatches = max(max(row["_counts"]) for row in rows)
    table = pd.DataFrame(
        {
            "Taxon": [row["Taxon"] for row in rows],
            "Primer": [row["Primer"] for row in rows],
            **{
                str(n): [row["_counts"].get(n, 0) for row in rows]
                for n in range(max_mismatches + 1)
            },
            "Min": [row["Min"] for row in rows],
            "Max": [row["Max"] for row in rows],
            "Mean": [row["Mean"] for row in rows],
        }
    )
    table.to_excel(SI_DIR / "supporting information 30.xlsx", index=False)
    print(table.to_string(index=False))


def primer_variant_histogram(
    supporting_information_15: Path,
    supporting_information_3: Path,
    full_results: pd.DataFrame,
) -> tuple:
    """Count the reads of the incorporated primer variants by their number of mismatches.

    Reads the primer counts (Supporting Information 15), keeps the taxa that are
    expected in a community (Supporting Information 14) and drops reads whose
    primer region matches none of the 288 variants. Each variant is compared with
    the template binding site of its taxon (Supporting Information 3). Aliquots
    and communities are pooled by summing reads. Returns the histogram (reads and
    fraction of reads per taxon, primer, number of first-step cycles and number
    of mismatches), the counts per variant, the primers, the IUPAC table and the
    template binding sites.
    """
    primers, iupac, sites = read_primer_binding_sites(supporting_information_3)
    primer_keys = {"forward": "fwhF2", "reverse": "fwhR2n"}

    # --- Counts of the taxa expected in each community, without unmatched reads ---
    counts = pd.read_parquet(supporting_information_15)
    expected = full_results.loc[full_results["primer"] == "fwh2", ["sample", "order"]]
    expected = expected.drop_duplicates().rename(columns={"order": "taxon"})
    counts = counts.merge(expected, on=["sample", "taxon"], how="inner")
    counts = counts[counts["variant"] != "other"]

    # --- Mismatches of every variant to the template site of its taxon ---
    variants = counts[["taxon", "primer", "variant"]].drop_duplicates()
    variants["mismatches"] = [
        sum(a != b for a, b in zip(variant, sites[(taxon, primer_keys[primer])]))
        for taxon, primer, variant in zip(
            variants["taxon"], variants["primer"], variants["variant"]
        )
    ]
    counts = counts.merge(variants, on=["taxon", "primer", "variant"])
    histogram = (
        counts.groupby(["taxon", "primer", "cycles", "mismatches"])["reads"].sum().reset_index()
    )
    histogram["fraction"] = histogram["reads"] / histogram.groupby(
        ["taxon", "primer", "cycles"]
    )["reads"].transform("sum")
    return histogram, counts, primers, iupac, sites


def primer_variant_trend(
    supporting_information_15: Path,
    supporting_information_3: Path,
    full_results: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether the mismatches of the incorporated variants rise with the cycle number.

    A series is one taxon in one community (aliquots pooled) and one primer. For
    each series the mean number of mismatches of the incorporated variants,
    weighted by read count, is calculated per number of first-step cycles and
    related to the cycle number with Spearman's rank correlation. The summary
    quoted in the results (number of series, share with a positive correlation,
    median and range of the correlation coefficient) is printed.
    """
    _, counts, _, _, _ = primer_variant_histogram(
        supporting_information_15, supporting_information_3, full_results
    )
    counts = counts.assign(weighted=counts["reads"] * counts["mismatches"])
    per_cycle = (
        counts.groupby(["sample", "taxon", "primer", "cycles"])[["weighted", "reads"]]
        .sum()
        .reset_index()
    )
    per_cycle["mean_mismatches"] = per_cycle["weighted"] / per_cycle["reads"]

    rows = []
    for (sample, taxon, primer), series in per_cycle.groupby(["sample", "taxon", "primer"]):
        rho, p_value = spearmanr(series["cycles"], series["mean_mismatches"])
        rows.append((sample, taxon, primer, len(series), rho, p_value))
    trend = pd.DataFrame(
        rows, columns=["sample", "taxon", "primer", "cycle numbers", "rho", "p"]
    )

    print(
        f"{len(trend)} series; rho > 0 in {(trend['rho'] > 0).sum()} "
        f"({(trend['rho'] > 0).mean():.1%}); median rho = {trend['rho'].median():.2f} "
        f"(range {trend['rho'].min():.2f} to {trend['rho'].max():.2f}); "
        f"p < 0.05 in {(trend['p'] < 0.05).sum()} ({(trend['p'] < 0.05).mean():.1%})"
    )
    return trend


def figure_5(
    supporting_information_15: Path,
    supporting_information_3: Path,
    full_results: pd.DataFrame,
) -> None:
    """Create Figure 5: replacement of the primer binding site in Lepidoptera.

    For Lepidoptera, the fraction of reads per number of mismatches between the
    incorporated primer variant and the template site is shown for the fwhF2 and
    fwhR2n primers after 6 to 20 cycles of the first PCR step. After few cycles,
    variants with few mismatches to the original site dominate; with more cycles
    the distribution approaches the one expected if all 288 primer variants were
    equally frequent (dashed line). The data are the same as in Supporting
    Information 31, where all taxa are shown.
    """
    histogram, counts, primers, iupac, sites = primer_variant_histogram(
        supporting_information_15, supporting_information_3, full_results
    )
    taxon = "Lepidoptera"
    panels = [("forward", "fwhF2", "A"), ("reverse", "fwhR2n", "B")]
    cycle_numbers = sorted(histogram["cycles"].unique())
    cycle_colors = plt.cm.viridis(np.linspace(0, 0.95, len(cycle_numbers)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (primer, primer_name, panel) in zip(axes, panels):
        # expectation: all 288 primer variants equally frequent
        all_variants = [
            "".join(v) for v in itertools.product(*[iupac[c] for c in primers[primer_name]])
        ]
        site = sites[(taxon, primer_name)]
        pool = pd.Series(
            [sum(a != b for a, b in zip(v, site)) for v in all_variants]
        ).value_counts(normalize=True).sort_index()
        mismatch_values = np.arange(0, pool.index.max() + 1)

        for cycles, color in zip(cycle_numbers, cycle_colors):
            subset = histogram[
                (histogram["taxon"] == taxon)
                & (histogram["primer"] == primer)
                & (histogram["cycles"] == cycles)
            ].set_index("mismatches")["fraction"]
            ax.plot(
                mismatch_values,
                subset.reindex(mismatch_values, fill_value=0).values,
                color=color, lw=1.6, marker="o", ms=3, label=f"c = {cycles}",
            )
        ax.plot(
            mismatch_values,
            pool.reindex(mismatch_values, fill_value=0).values,
            color="black", ls="--", lw=1.3, label="all 288 variants equally frequent",
        )
        ax.set_title(f"{panel}) {primer_name}", fontsize=12)
        ax.set_xticks(mismatch_values)
        ax.set_xlabel("Mismatches of the incorporated variant to the template site")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Fraction of reads")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.13),
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure 5.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure 5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def supporting_information_31(
    supporting_information_15: Path,
    supporting_information_3: Path,
    full_results: pd.DataFrame,
) -> None:
    """Create the histograms of the incorporated primer variants (Supporting Information 31).

    The primer counts (Supporting Information 15) give, per sample, aliquot, number
    of first-step cycles, taxon and primer, how many reads carry each of the 288
    primer variants. Each variant is compared with the template binding site of
    its taxon (Supporting Information 3) and counted as its number of
    mismatches. Only taxa expected in a community (Supporting Information 14) are
    used, and aliquots and communities are pooled by summing reads. For every
    taxon and primer, one line per number of first-step cycles shows the fraction
    of reads per number of mismatches; the dashed line is the expectation if all
    288 primer variants were equally frequent. Lines end at the highest number of
    mismatches that is possible for the taxon and primer.
    """
    histogram, counts, primers, iupac, sites = primer_variant_histogram(
        supporting_information_15, supporting_information_3, full_results
    )
    primer_keys = {"forward": "fwhF2", "reverse": "fwhR2n"}

    cycle_numbers = sorted(histogram["cycles"].unique())
    cycle_colors = plt.cm.viridis(np.linspace(0, 0.95, len(cycle_numbers)))

    fig, axes = plt.subplots(
        len(ORDER_NAMES), 2, figsize=(9, 11.6), sharey=True
    )
    max_mismatches = {"forward": 0, "reverse": 0}
    for i, taxon in enumerate(ORDER_NAMES):
        for j, primer in enumerate(["forward", "reverse"]):
            ax = axes[i, j]
            primer_name = primer_keys[primer]

            # expectation: all 288 primer variants equally frequent
            all_variants = [
                "".join(v) for v in itertools.product(*[iupac[c] for c in primers[primer_name]])
            ]
            site = sites[(taxon, primer_name)]
            pool = pd.Series(
                [sum(a != b for a, b in zip(v, site)) for v in all_variants]
            ).value_counts(normalize=True).sort_index()
            mismatch_values = np.arange(0, pool.index.max() + 1)
            max_mismatches[primer] = max(max_mismatches[primer], pool.index.max())

            for cycles, color in zip(cycle_numbers, cycle_colors):
                subset = histogram[
                    (histogram["taxon"] == taxon)
                    & (histogram["primer"] == primer)
                    & (histogram["cycles"] == cycles)
                ].set_index("mismatches")["fraction"]
                ax.plot(
                    mismatch_values,
                    subset.reindex(mismatch_values, fill_value=0).values,
                    color=color, lw=1.6, marker="o", ms=3, label=f"c = {cycles}",
                )
            ax.plot(
                mismatch_values,
                pool.reindex(mismatch_values, fill_value=0).values,
                color="black", ls="--", lw=1.3, label="all 288 variants equally frequent",
            )
            n_reads = int(
                counts.loc[(counts["taxon"] == taxon) & (counts["primer"] == primer), "reads"].sum()
            )
            ax.set_title(
                f"{taxon}, {primer_name} (n = {n_reads:,} reads)",
                fontsize=9, loc="left", color=ORDER_PALETTE[taxon], fontweight="bold",
            )
            sns.despine(ax=ax)

    # same x-axis within a column: 0 up to the highest possible number of mismatches
    for j, primer in enumerate(["forward", "reverse"]):
        for ax in axes[:, j]:
            ax.set_xticks(range(max_mismatches[primer] + 1))
            ax.set_xlim(-0.5, max_mismatches[primer] + 0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of reads")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.supxlabel(
        "Mismatches of the incorporated variant to the template site", y=0.052, fontsize=10
    )

    plt.tight_layout(rect=(0, 0.075, 1, 1))
    plt.savefig(SI_DIR / "supporting information 31.pdf")
    plt.savefig(SI_DIR / "supporting information 31.png", dpi=200)
    plt.close(fig)


def supporting_information_32(
    supporting_information_11: Path,
    supporting_information_27: Path,
    full_results: pd.DataFrame,
) -> None:
    """Create the GC content table of the amplicons (Supporting Information 32).

    The sequences of Supporting Information 11 are primer-trimmed. For each taxon
    and primer pair (fwh2: all numbers of first-step cycles; BF3: 20 cycles) the
    GC content of a sequence is (G + C) / length. The table gives the number of
    sequences, the length of the most abundant sequence, the mean GC content
    weighted by the read count of each sequence, and the lowest and highest GC
    content. The relative amplification efficiency of the constant bias model
    (Supporting Information 27, Blattodea = 1) is added for comparison. Only taxa
    expected in a community (Supporting Information 14) are used.
    """
    reads = pd.read_parquet(
        supporting_information_11,
        columns=["sample", "primer", "order", "sequence", "read_count"],
    )
    expected = full_results[["sample", "primer", "order"]].drop_duplicates()
    reads = reads.merge(expected, on=["sample", "primer", "order"])

    # reads of a sequence summed over samples, aliquots and cycle numbers
    unique_sequences = reads.groupby(["primer", "order", "sequence"], as_index=False)["read_count"].sum()
    unique_sequences["gc"] = (
        100 * unique_sequences["sequence"].str.count("[GC]") / unique_sequences["sequence"].str.len()
    )

    efficiencies = pd.read_excel(supporting_information_27).set_index("order")
    efficiency_columns = {
        "fwh2": "E_i fwh2 (constant bias model)",
        "bf3": "E_i BF3 (constant bias model)",
    }

    rows = []
    for taxon in ORDER_NAMES:
        for primer, label in [("fwh2", "fwh2"), ("bf3", "BF3")]:
            group = unique_sequences[
                (unique_sequences["primer"] == primer) & (unique_sequences["order"] == taxon)
            ]
            weights = group["read_count"] / group["read_count"].sum()
            most_abundant = group.sort_values("read_count", ascending=False).iloc[0]
            rows.append(
                {
                    "Taxon": taxon,
                    "Primer pair": label,
                    "Sequences (n)": len(group),
                    "Length (nt)": len(most_abundant["sequence"]),
                    "GC content, read-weighted mean (%)": round((group["gc"] * weights).sum(), 1),
                    "GC content, lowest (%)": round(group["gc"].min(), 1),
                    "GC content, highest (%)": round(group["gc"].max(), 1),
                    "Relative E, constant bias model": efficiencies.loc[taxon, efficiency_columns[primer]],
                }
            )
    table = pd.DataFrame(rows)
    table.to_excel(SI_DIR / "supporting information 32.xlsx", index=False)
    print(table.to_string(index=False))


def main():
    """Run the complete analysis from raw data files to all figures and tables.

    This function orchestrates the entire study in nine steps:

    Step 1 — qPCR assay validation (Supporting Information 6):
        Taxon-specific qPCR assays were used to check whether each target insect
        order was detectable in each community sample (both extraction replicates).
        The results are shown as a heatmap of true/false positives and negatives
        (Supporting Information 6). Input: Supporting Information 5.

    Step 2 — Metabarcoding data processing (Supporting Information 12):
        Raw sequencing reads (Supporting Information 11) are summed over sequencing
        lanes per sample, aliquot, PCR cycle count, insect order, and primer pair.
        Reads of non-target taxa and of samples 26, 92 and 93 (not part of the
        controlled experiment) are removed. Reads are converted to proportions
        (relative reads) so that samples with different total read counts are
        comparable. Two aliquots were sequenced per sample as a reproducibility
        check; their proportions are compared with a Wilcoxon signed-rank test
        (Supporting Information 12). Aliquots are then pooled by summing reads,
        which weights the pooled proportion by sequencing depth.

    Step 3 — dPCR data processing (Supporting Information 13):
        Digital PCR (dPCR) gives absolute mtDNA copy numbers per insect taxon
        (Supporting Information 8). The two replicates are checked for consistency
        (Wilcoxon test, Supporting Information 13) and then averaged to obtain one
        copy number per (sample, order).

    Step 4 — Merge all measurements:
        Metabarcoding reads, dPCR copy numbers, and known biomass proportions
        (Supporting Information 1) are joined into one unified table. This
        combined table (Supporting Information 14) is the basis for all downstream
        analyses.

    Step 5 — Fold-change and rank analysis (Figure 2, Table 1, Supporting
    Information 17, 18, 19, and 20):
        For each taxon and sample we compute how far reads deviate from copy numbers
        and from biomass (log2 fold-change). We also check whether the rank order
        of taxa by reads agrees with the rank order by copies (rank concordance).
        Figure 2 shows the fold-change distributions; Table 1 summarises rank
        concordance; Supporting Information 17 counts over- and underrepresentation
        by order. Supporting Information 18 (fwh2) and 19 (BF3) show the per-sample
        community composition at c=20; Supporting Information 20 shows how the
        read proportions change across PCR cycle counts (fwh2).

    Step 6 — Model comparison (Figure 3, Figure 4, Table 2, Supporting
    Information 24, 25, 26, 27, and 28):
        Four approaches to predicting taxon proportions from reads are compared:
          1. No correction: use reads directly as a proxy for copies.
          2. Cycle calibration (Shelton et al. 2023): regress the log-ratio of
             read counts against cycle number within each community, using reads
             measured at multiple cycle counts; the intercept estimates the
             starting copy ratio.
          3. Cycle-dependent bias model: estimate a per-cycle amplification
             efficiency factor E_i from 14 two-species calibration communities;
             apply as reads / E_i^c, then renormalise.
          4. Constant bias model (our approach): estimate a single amplification
             efficiency factor E_i (the odds ratio at c=20) from the same 14
             calibration communities; apply as reads / E_i, then renormalise.
        Approaches 3 and 4 are evaluated only on the 67 communities that were
        not used for calibration (out-of-sample test). Figure 3 compares
        approaches 1 and 2 on all 81 communities; Figure 4 and Supporting
        Information 24 and 26 compare approaches 3 and 4 (fwh2). Table 2 and
        Supporting Information 27 list the E_i values; Supporting Information 25
        extends Supporting Information 21 with E_i, bias model predictions and
        Aitchison distances. Supporting Information 28 repeats the constant bias
        model for BF3 (c=20 only).

    Step 7 — Export of results (Supporting Information 14, 17, 21; Table 1):
        The merged table (Supporting Information 14), rank concordance (Table 1),
        and over-/underrepresentation counts (Supporting Information 17) are
        written to Excel. The cycle calibration predictions (Supporting
        Information 21) are exported as an extended version of Supporting
        Information 14. BF3 rows have no cycle calibration values because that
        method requires data at multiple cycle counts, and BF3 was only sequenced
        at c=20.

    Step 8 — Second-step PCR (Supporting Information 9_1 and 9_2):
        The qPCR raw data and per-well efficiencies of the second-step PCR
        (Supporting Information 10) are summarised in a figure (amplification
        curves and efficiencies per taxon, Supporting Information 9_1) and a table
        (mean, minimum and maximum efficiency per taxon, Supporting Information
        9_2). The statistics quoted in the text (Kruskal-Wallis test, Shapiro-Wilk
        test, fold differences) are printed.

    Step 9 — In silico primer analysis (Supporting Information 29, 30, 31, 32;
    Figure 5):
        The binding sites of the fwh2 primers are read from the Sanger consensus
        sequences (Supporting Information 3). They are shown aligned to the
        degenerate primers (Supporting Information 29), and the mismatches of all
        288 primer variants are tabulated (Supporting Information 30). The counts
        of the incorporated primer variants (Supporting Information 15) are shown
        as histograms of the number of mismatches per number of first-step cycles
        for all taxa (Supporting Information 31) and for Lepidoptera (Figure 5);
        the Spearman correlation of the mean number of mismatches with the cycle
        number per series is printed. The GC content of the amplicons per taxon
        and primer pair is exported together with the relative efficiencies of the
        constant bias model (Supporting Information 32).
    """
    # --- Input files ---
    supporting_information_1 = (
        SI_DIR / "supporting information 1.xlsx"
    )  # community compositions + biomass
    supporting_information_4 = (
        SI_DIR / "supporting information 4.xlsx"
    )  # dPCR primer/probe sequences
    supporting_information_5 = (
        SI_DIR / "supporting information 5.xlsx"
    )  # qPCR validation data
    supporting_information_8 = (
        SI_DIR / "supporting information 8.xlsx"
    )  # dPCR raw copy numbers
    supporting_information_11 = (
        SI_DIR / "supporting information 11.parquet.snappy"
    )  # metabarcoding raw reads
    supporting_information_10 = (
        SI_DIR / "supporting information 10.xlsx"
    )  # qPCR raw data + efficiencies, second-step PCR
    supporting_information_3 = (
        SI_DIR / "supporting information 3.fasta"
    )  # Sanger consensus sequences of the single specimens
    supporting_information_15 = (
        SI_DIR / "supporting information 15.parquet.snappy"
    )  # primer variant counts per taxon, sample, aliquot and cycles

    # --- Step 1: qPCR primer validation heatmap ---
    supporting_information_6(supporting_information_5)

    # --- Step 2: Metabarcoding data processing ---
    aggregated_read_data = aggregate_metabarcoding_data(
        supporting_information_1, supporting_information_11
    )
    relative_read_data = absolute_reads_to_relative_reads(aggregated_read_data)

    # --- Step 3: dPCR data processing ---
    dpcr_data = aggregate_dpcr_data(supporting_information_8)

    # --- Step 4: Merge all data sources ---
    full_results = final_aggregation(
        relative_read_data, dpcr_data, supporting_information_1
    )

    # --- Step 5: Fold-change, rank concordance, and Figure 2 ---
    full_results = compute_fold_changes_and_ranks(full_results)
    plot_fold_change_stripplot(
        full_results, primer="fwh2", cycles=20, filename=str(FIGURES_DIR / "figure 2.pdf")
    )

    # --- Step 6: Model comparison (Figure 3, Figure 4, Table 2, SI 24-28) ---
    predict_copies_model_comparison(full_results)

    # --- Step 7: Export full results and additional figures ---

    # Supporting Information 14: the merged table with all measurements
    full_results.to_excel(SI_DIR / "supporting information 14.xlsx", index=False)

    # Table 1: rank concordance counts (fwh2 and BF3 side by side)
    export_table1_rank_concordance(full_results, filename=str(TABLES_DIR / "table 1.xlsx"))

    # Supporting Information 17: over/concordant/underrepresentation by order
    #   Sheet 1 — Biomass vs Copies  (method-independent; one table)
    #   Sheet 2 — Copies vs Reads    (fwh2 and BF3 side by side)
    #   Sheet 3 — Biomass vs Reads   (fwh2 and BF3 side by side)
    export_si11_over_under(full_results, filename=str(SI_DIR / "supporting information 17.xlsx"))

    # Supporting Information 18 & 19: stacked barplots of community composition
    # at c=20 for fwh2 (SI 18) and BF3 (SI 19)
    plot_stacked_barplots(
        full_results, primer="fwh2", cycles=20, filename=str(SI_DIR / "supporting information 18.pdf")
    )
    plot_stacked_barplots(
        full_results, primer="bf3", cycles=20, filename=str(SI_DIR / "supporting information 19.pdf")
    )

    # Supporting Information 20: how read proportions change across cycle counts
    plot_reads_per_cycle(
        full_results, primer="fwh2", filename=str(SI_DIR / "supporting information 20.pdf")
    )

    # Supporting Information 21: cycle calibration table (fwh2 only).
    # BF3 rows are kept but their cycle-calibration columns are left blank
    # because cycle calibration requires data at multiple cycle counts,
    # and BF3 was only run at c=20.
    cycle_cal_table = compute_cycle_calibration(full_results, primer="fwh2")
    cycle_cal_table["primer"] = "fwh2"
    cycle_cal_export = full_results.merge(
        cycle_cal_table, on=["sample", "order", "primer"], how="left"
    )
    cycle_cal_export.to_excel(SI_DIR / "supporting information 21.xlsx", index=False)

    # --- Step 8: second-step PCR, qPCR efficiencies (SI 9_1, SI 9_2) ---
    supporting_information_9_1(supporting_information_10)
    supporting_information_9_2(supporting_information_10)

    # --- Step 9: in silico primer analysis ---
    # primer binding site alignment (SI 29) and mismatch table (SI 30)
    supporting_information_29(supporting_information_3)
    supporting_information_30(supporting_information_3)

    # histograms of the incorporated primer variants (SI 31)
    supporting_information_31(
        supporting_information_15, supporting_information_3, full_results
    )

    # Figure 5: replacement of the primer binding site (Lepidoptera)
    figure_5(supporting_information_15, supporting_information_3, full_results)

    # GC content of the amplicons (SI 32)
    supporting_information_32(
        supporting_information_11, (SI_DIR / "supporting information 27.xlsx"), full_results
    )

    # Spearman correlation of the mean mismatches with the cycle number per series
    # (the summary quoted in the results is printed)
    primer_variant_trend(
        supporting_information_15, supporting_information_3, full_results
    )


if __name__ == "__main__":
    main()
