# Deep Research CLI

An interactive command-line tool for comprehensive academic research powered by GenAI and multiple scholarly APIs.

## Features

- **Multi-Source Paper Collection**: Fetches research papers from:

  - Semantic Scholar
  - IEEE Xplore
  - Google Scholar (via SerpAPI)
  - arXiv
  - Crossref

- **Intelligent Paper Ranking**: Uses a weighted combination of:

  - Semantic relevance (using Gemini embeddings)
  - Citation count
  - Publication recency

- **Interactive Refinement**:

  - Generate comprehensive research reports
  - Ask questions about the papers and findings
  - Refine searches based on feedback
  - Continue until satisfied with results

- **Rich Output**:
  - Detailed Markdown reports with structured sections
  - Paper rankings in CSV/JSON formats
  - Interactive terminal UI with progress bars and tables

## Prerequisites

- Python 3.x
- Required API keys:
  - GEMINI_API_KEY (required)
  - SEMANTIC_SCHOLAR_KEY (optional)
  - IEEE_API_KEY (optional)
  - SERPAPI_KEY (optional)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/deep-research.git
cd deep-research
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic command:

```bash
python main.py "your research topic"
```

Advanced options:

```bash
python main.py "Graph Neural Networks for Traffic Forecasting" \
    --top-k 30 \
    --max-papers 80 \
    --weights 0.5 0.3 0.2 \
    --model gemini-2.5-flash \
    --out report.md
```

### Command Line Arguments

- `topic`: Research topic (required)
- `--max-papers`: Maximum papers to fetch (default: 80)
- `--top-k`: Top papers to include in report (default: 25)
- `--weights`: Weights for relevance, citations, recency (default: 0.4 0.25 0.35)
- `--embed-model`: Gemini embedding model (default: text-embedding-004)
- `--model`: Gemini generation model (default: gemini-2.5-flash)
- `--out`: Output Markdown path (default: report.md)
- `--csv`: Output CSV path (default: ranked_papers.csv)
- `--json`: Output JSON path (default: ranked_papers.json)
- `--no-interactive`: Disable interactive refinement/Q&A loop

## Report Structure

Generated reports include:

1. Executive Summary (300-500 words)
2. In-Depth Background & Core Concepts (400-600 words)
3. Comparative Literature Synthesis (600-900 words)
4. Critical Gap Analysis (300-500 words)
5. Future Research Directions (300-500 words)
6. Risks, Ethics, and Limitations (200-400 words)
7. Practical Applications and Tooling Landscape (200-400 words)
8. Conclusion (150-300 words)
9. References

## Interactive Mode

After generating the initial report, you can:

- Ask questions about the research
- Request refinements to focus on specific aspects
- Get clarification on findings
- Accept the report when satisfied

## Dependencies

- google-generativeai >= 0.3.0
- requests >= 2.31.0
- numpy >= 1.24.0
- pandas >= 1.5.0
- python-dateutil >= 2.8.2
- tenacity >= 8.2.2
- tqdm >= 4.65.0
- rich >= 13.5.0
- feedparser

## Notes

- Primary source is Semantic Scholar API
- IEEE Xplore and Google Scholar (via SerpAPI) are optional enrichers
- Ranking uses weighted mixture of semantic relevance, log citations, and recency
- Reports are saved in Markdown format with proper citations
- Paper rankings are exported to both CSV and JSON formats
