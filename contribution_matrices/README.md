# Contribution Matrices

A text-mining pipeline that builds and visualizes **term co-occurrence networks** from a corpus of ProQuest dissertation abstracts. Given a set of target keywords, the pipeline computes corpus-level TF-IDF, constructs a co-occurrence contribution matrix, and renders an interactive-quality network graph — revealing which terms cluster together in the research literature.

##  Workflow

```
Excel (ProQuest abstracts)
   │
   ├─ 1. sheet1_col3_top_keywords.py  →  Top-N keyword extraction
   ├─ 2. compute_term_tfidf.py        →  TF-IDF for a curated term list
   ├─ 3. build_contribution_matrices.py →  Co-occurrence matrices
   └─ 4. visualize_contribution_matrix.py →  Network graph (PNG)
```

##  What Each Script Does

| Script | Purpose |
|--------|---------|
| `sheet1_col3_top_keywords.py` | Tokenize abstracts in column 3, remove stopwords, and output the top *N* keywords by frequency. Useful for discovering what terms to include in the curated list. |
| `compute_term_tfidf.py` | Given a hard-coded term list (e.g., `trade`, `tariff`, `tesla`, `byd`), compute **total count**, **document count**, and **TF-IDF** across the full corpus. Uses `inflect` to match both singular and plural forms. |
| `build_contribution_matrices.py` | Read paragraphs from column 3 and a TF-IDF term list. For every paragraph where two terms co-occur, increment their edge weight. Outputs two symmetric matrices: a **binary contribution matrix** (co-occurrence = 1) and a **weighted matrix** (co-occurrence × frequency). |
| `visualize_contribution_matrix.py` | Read a contribution matrix and render it as a **network graph** using NetworkX + Matplotlib. Features spring layout, unique HSV-based node colors, edge filtering by percentile, and peripheral-node pull for cleaner aesthetics. |

##  Installation

```bash
pip install openpyxl inflect numpy matplotlib networkx
```

##  Usage

### Step 1 — Extract top keywords (optional)

```bash
python sheet1_col3_top_keywords.py 50
```

Outputs the 50 most frequent keywords, which you can use to curate your term list.

### Step 2 — Compute TF-IDF

Edit the `TERMS` list in `compute_term_tfidf.py`, then:

```bash
python compute_term_tfidf.py \
  --input "your_proquest_data.xlsm" \
  --output "tf-idf-output.xlsx"
```

### Step 3 — Build contribution matrices

```bash
python build_contribution_matrices.py \
  --input "your_proquest_data.xlsm" \
  --terms "tf-idf-output.xlsx" \
  --output "contribution-matrices.xlsx"
```

Output workbook contains two sheets:
- **contribution_matrix** — binary co‑occurrence
- **weighted_contribution_matrix** — frequency‑weighted co‑occurrence

### Step 4 — Visualize

```bash
python visualize_contribution_matrix.py \
  --input "contribution-matrices.xlsx" \
  --output "network-graph.png" \
  --edge-percentile 88 \
  --top-edges-per-node 4 \
  --layout-k 1.05
```

Key parameters:
| Flag | Default | Effect |
|------|---------|--------|
| `--edge-percentile` | 88 | Only keep edges above this percentile (declutters) |
| `--top-edges-per-node` | 4 | Max edges retained per node |
| `--layout-k` | 1.05 | Spring-layout compactness (lower = tighter) |
| `--peripheral-pull` | 0.68 | Pull isolated nodes toward center |

## Example Output

A network graph where:
- **Node size** = term frequency
- **Node color** = unique hue per term
- **Edge width & opacity** = co‑occurrence strength
- **Layout** = terms that frequently co‑occur are placed closer together

##  Dependencies

- Python ≥ 3.9
- `openpyxl` — Excel I/O
- `inflect` — singular/plural matching
- `numpy` — matrix operations
- `matplotlib` — rendering
- `networkx` — graph data structure & layout

##  Project Structure

```
contribution_matrices/
├── sheet1_col3_top_keywords.py       # Keyword frequency extraction
├── compute_term_tfidf.py             # Corpus-level TF-IDF computation
├── build_contribution_matrices.py    # Co-occurrence matrix builder
├── visualize_contribution_matrix.py  # Network graph renderer
└── README.md
```
