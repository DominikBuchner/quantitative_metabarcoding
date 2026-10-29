# Supporting Information — Analysis Code

This repository contains the raw data and analysis code underlying:

> **Characterizing mitochondrial copy number variation and PCR amplification bias as sources of quantitative constraints in DNA metabarcoding**

The script [`supporting information/supporting information 16.py`](supporting%20information/supporting%20information%2016.py) reproduces the entire quantitative analysis of the paper end to end — from the raw qPCR, dPCR and metabarcoding read data to every figure, table, and supporting information file reported in the manuscript.

## Repository structure

```
├── figures/                 Figures 1–5 (.pdf/.png)
├── tables/                  Table 1 and Table 2 (.xlsx)
├── supporting information/  Input data, the analysis script, and all SI outputs
└── pcr_simulator/           Standalone PCR/metabarcoding simulator (HTML)
```

An overview of all supporting information files is given in [`Overview supporting information.xlsx`](supporting%20information/Overview%20supporting%20information.xlsx).

### PCR/metabarcoding simulator

The interactive simulator in [`pcr_simulator/`](pcr_simulator/) is hosted via GitHub Pages and can be used directly in the browser, no download required:

**https://dominikbuchner.github.io/quantitative_metabarcoding/pcr_simulator/pcr_metabarcoding_simulator_en.html**

## Requirements

- Python ≥ 3.11
- Packages listed in [`requirements.txt`](requirements.txt) (exact versions used to generate the results reported in the paper)

### Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the analysis

The script locates its input files and writes its outputs relative to its own location, so it can be run from any working directory:

```bash
python "supporting information/supporting information 16.py"
```

Figures are written to `figures/`, tables to `tables/`, and all other outputs to `supporting information/`, regardless of where the command is run from.

Runtime is a few minutes on a standard laptop. Console output includes read-filtering statistics, Wilcoxon signed-rank test results for the aliquot/replicate reproducibility checks, the statistics of the second-step PCR efficiencies (Kruskal-Wallis test, Shapiro-Wilk test, fold differences), the mismatch table of the primer variants, the GC content table and the Spearman correlations of the incorporated primer variants with the cycle number, all as reported in the text.

## Input data (required, included in this repository)

Located in [`supporting information/`](supporting%20information/):

| File | Content |
|---|---|
| `supporting information 1.xlsx` | Expected community composition and biomass per sample |
| `supporting information 3.fasta` | Sanger consensus sequences of the single specimens (primer binding sites of the fwh2 primers) |
| `supporting information 5.xlsx` | qPCR primer validation results |
| `supporting information 8.xlsx` | dPCR raw mtDNA copy number measurements |
| `supporting information 10.xlsx` | qPCR raw data, per-well efficiencies and melt curves of the second-step PCR |
| `supporting information 11.parquet.snappy` | Raw metabarcoding read counts |
| `supporting information 15.parquet.snappy` | Counts of the incorporated fwh2 primer variants per taxon, sample, aliquot and number of cycles |

`supporting information 4.xlsx` (dPCR primer/probe sequences) is included for reference but is not read by the script.

The primer variant counts (`supporting information 15.parquet.snappy`) were extracted from the merged reads with cutadapt (v. 5.2) on a server; the extraction script is not part of this repository.

## Output

Running the script (re-)generates the following files:

- **Figures** (in [`figures/`](figures/)): `figure 2.pdf`, `figure 3.pdf`/`.png`, `figure 4.pdf`/`.png`, `figure 5.pdf`/`.png`
- **Tables** (in [`tables/`](tables/)): `table 1.xlsx`, `table 2.xlsx`
- **Supporting information** (in [`supporting information/`](supporting%20information/)): `supporting information 6.pdf`, `9_1.pdf`/`.png`, `9_2.xlsx`, `12.pdf`, `13.pdf`, `14.xlsx`, `17.xlsx`, `18.pdf`, `19.pdf`, `20.pdf`, `21.xlsx`, `24.pdf`/`.png`, `25.xlsx`, `26.pdf`/`.png`, `27.xlsx`, `28.pdf`/`.png`, `29.pdf`/`.png`, `30.xlsx`, `31.pdf`/`.png`, `32.xlsx`

All other supporting information files (e.g. SI 2, 7, 9, 22, 23) were prepared independently of this script and are provided as-is.

## Structure of the script

`supporting information 16.py` is organized as a sequence of functions called from `main()`, each documented with a docstring explaining its purpose:

1. **qPCR validation** — heatmap of primer specificity per insect order (SI 6)
2. **Metabarcoding processing** — read aggregation, relative-abundance conversion, aliquot reproducibility (SI 12)
3. **dPCR processing** — relative copy numbers, replicate reproducibility (SI 13)
4. **Data merge** — combines reads, copy numbers, and biomass into one master table (SI 14)
5. **Fold-change / rank concordance** — agreement between the three measurement types (Figure 2, Table 1, SI 17–20)
6. **Bias model comparison** — no correction vs. cycle calibration (Shelton et al. 2023) vs. cycle-dependent vs. constant bias correction (Figures 3–4, Table 2, SI 21, 24–28)
7. **Second-step PCR** — efficiency of the second-step PCR per taxon and melt curves of the qPCR products from the qPCR raw data (SI 9_1, 9_2)
8. **In silico primer analysis** — alignment and mismatch table of the primer variants (SI 29, 30), histograms of the incorporated primer variants (SI 31, Figure 5) and GC content of the amplicons (SI 32)

See the docstrings in the script itself for full methodological detail on each step.

## License

This code is released under the [MIT License](LICENSE).
