# Supporting Information — Analysis Code

This repository contains the raw data and analysis code underlying:

> **Characterizing mitochondrial copy number variation and PCR amplification bias as sources of quantitative constraints in DNA metabarcoding**

The script [`supporting information/supporting information 10.py`](supporting%20information/supporting%20information%2010.py) reproduces the entire quantitative analysis of the paper end to end — from the raw qPCR, ddPCR and metabarcoding read data to every figure, table, and supporting information file reported in the manuscript.

## Repository structure

```
├── figures/                 Figures 2–4 (.pdf/.png)
├── tables/                  Table 1 and Table 2 (.xlsx)
├── supporting information/  Input data, the analysis script, and all SI outputs
└── pcr_simulator/           Standalone PCR/metabarcoding simulator (HTML)
```

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
python "supporting information/supporting information 10.py"
```

Figures are written to `figures/`, tables to `tables/`, and all other outputs to `supporting information/`, regardless of where the command is run from.

Runtime is a few minutes on a standard laptop. Console output includes read-filtering statistics and Wilcoxon signed-rank test results for the aliquot/replicate reproducibility checks reported in the text.

## Input data (required, included in this repository)

Located in [`supporting information/`](supporting%20information/):

| File | Content |
|---|---|
| `supporting information 1.xlsx` | Expected community composition and biomass per sample |
| `supporting information 3.xlsx` | qPCR primer validation results |
| `supporting information 5.xlsx` | ddPCR raw mtDNA copy number measurements |
| `supporting information 6.parquet.snappy` | Raw metabarcoding read counts |

`supporting information 2.xlsx` (ddPCR primer/probe sequences) is included for reference but is not read by the script.

## Output

Running the script (re-)generates the following files:

- **Figures** (in [`figures/`](figures/)): `figure 2.pdf`, `figure 3.pdf`/`.png`, `figure 4.pdf`/`.png`
- **Tables** (in [`tables/`](tables/)): `table 1.xlsx`, `table 2.xlsx`
- **Supporting information** (in [`supporting information/`](supporting%20information/)): `supporting information 4.pdf`, `7.pdf`, `8.pdf`, `9.xlsx`, `11.xlsx`, `12.pdf`, `13.pdf`, `14.pdf`, `15.xlsx`, `18.pdf`/`.png`, `19.xlsx`, `20.pdf`/`.png`, `21.xlsx`, `22.pdf`/`.png`

All other supporting information files (e.g. SI 16, 17) were prepared independently of this script and are provided as-is.

## Structure of the script

`supporting information 10.py` is organized as a sequence of functions called from `main()`, each documented with a docstring explaining its purpose:

1. **qPCR validation** — heatmap of primer specificity per insect order (SI 4)
2. **Metabarcoding processing** — read aggregation, relative-abundance conversion, aliquot reproducibility (SI 7)
3. **ddPCR processing** — relative copy numbers, replicate reproducibility (SI 8)
4. **Data merge** — combines reads, copy numbers, and biomass into one master table (SI 9)
5. **Fold-change / rank concordance** — agreement between the three measurement types (Figure 2, Table 1, SI 11–14)
6. **Bias model comparison** — no correction vs. cycle calibration (Shelton et al. 2023) vs. cycle-dependent vs. constant bias correction (Figures 3–4, Table 2, SI 15, 18–22)

See the docstrings in the script itself for full methodological detail on each step.

## License

This code is released under the [MIT License](LICENSE).
