# Supporting Information — Analysis Code

This repository contains the raw data and analysis code underlying:

> **Characterizing mitochondrial copy number variation and PCR amplification bias as sources of quantitative constraints in DNA metabarcoding**

The script [`supporting information 10.py`](supporting%20information%2010.py) reproduces the entire quantitative analysis of the paper end to end — from the raw qPCR, ddPCR and metabarcoding read data to every figure, table, and supporting information file reported in the manuscript.

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

Run the script from within this folder (it reads and writes files using relative paths):

```bash
python "supporting information 10.py"
```

Runtime is a few minutes on a standard laptop. Console output includes read-filtering statistics and Wilcoxon signed-rank test results for the aliquot/replicate reproducibility checks reported in the text.

## Input data (required, included in this repository)

| File | Content |
|---|---|
| `supporting information 1.xlsx` | Expected community composition and biomass per sample |
| `supporting information 3.xlsx` | qPCR primer validation results |
| `supporting information 5.xlsx` | ddPCR raw mtDNA copy number measurements |
| `supporting information 6.parquet.snappy` | Raw metabarcoding read counts |

`supporting information 2.xlsx` (ddPCR primer/probe sequences) is included for reference but is not read by the script.

## Output

Running the script (re-)generates the following files in this folder:

- **Figures:** `figure 2.pdf`, `figure 3.pdf`/`.png`, `figure 4.pdf`/`.png`
- **Tables:** `table 1.xlsx`, `table 2.xlsx`
- **Supporting information:** `supporting information 4.pdf`, `7.pdf`, `8.pdf`, `9.xlsx`, `11.xlsx`, `12.pdf`, `13.pdf`, `14.pdf`, `15.xlsx`, `18.pdf`/`.png`, `19.xlsx`, `20.pdf`/`.png`, `21.xlsx`, `22.pdf`/`.png`

All other supporting information files in this folder (e.g. SI 16, 17) were prepared independently of this script and are provided as-is.

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
