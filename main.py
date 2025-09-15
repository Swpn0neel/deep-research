#!/usr/bin/env python3
"""
Deep Research CLI (Terminal-only) — Interactive
-----------------------------------------------

Adds a user-interactive refinement loop and Q&A:
- After generating a report, the app prompts for feedback.
- If feedback requests refinement, it fetches *new* papers based on the feedback, re-ranks, and regenerates the report. Repeats until satisfied.
- If the user asks questions instead, it answers with Gemini using the *current report + top papers* as context (no regeneration). This Q&A is recursive as long as the user wishes.
- Saves JSON/CSV of the ranked paper set each time a report is regenerated.

Usage
-----
python deep_research_cli.py "your research topic here" \
    --top-k 30 \
    --max-papers 80 \
    --weights 0.5 0.3 0.2 \
    --model gemini-2.5-flash \
    --out report.md

Environment
-----------
GEMINI_API_KEY        (required)
SEMANTIC_SCHOLAR_KEY  (optional)
IEEE_API_KEY          (optional)
SERPAPI_KEY           (optional; for Google Scholar via SerpAPI)

Install
-------
pip install google-generativeai requests numpy pandas tenacity python-dateutil tqdm rich

Notes
-----
- Primary source: Semantic Scholar API.
- IEEE Xplore and Google Scholar (via SerpAPI) are optional enrichers.
- Ranking = weighted mixture of (semantic relevance, log citations, recency).
- Interactivity loop supports [refine | ask | accept | exit].
"""

from __future__ import annotations
import os
import sys
import math
import json
import csv
import time
import argparse
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

# ----------- Globals -----------
console = Console()
NOW_YEAR = time.gmtime().tm_year

# ------------- Helpers -------------
class APIError(RuntimeError):
    pass

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(APIError))
def _get_json(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code >= 400:
        raise APIError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except Exception as e:
        raise APIError(f"Invalid JSON from {url}: {e}")

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(APIError))
def _head(url: str) -> bool:
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        return resp.ok
    except requests.RequestException:
        return False


def _log1p_normalized(x: Optional[int]) -> float:
    if not x or x <= 0:
        return 0.0
    # Normalize citation count to ~[0,1] using log scale
    return min(1.0, math.log1p(x) / math.log(1 + 1000))  # 1000 citations ~ 1.0


def _recency_score(year: Optional[int]) -> float:
    if not year:
        return 0.0
    # Map age to [0,1]: current year -> 1, 10 years old -> ~0.2, older -> diminishes
    age = max(0, NOW_YEAR - int(year))
    return max(0.0, 1.0 - (age / 12.0))


@dataclass
class Paper:
    source: str
    paper_id: str
    title: str
    abstract: str
    year: Optional[int]
    authors: List[str]
    venue: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    doi: Optional[str]
    citation_count: Optional[int]
    similarity: float = 0.0
    score: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ------------- Data Sources -------------

class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search(self, query: str, limit: int = 50) -> List[Paper]:
        fields = [
            "title,abstract,year,authors,venue,externalIds,url,openAccessPdf,citationCount"
        ]
        params = {
            "query": query,
            "limit": min(100, max(1, limit)),
            "fields": ",".join(fields),
        }
        res = _get_json(f"{self.BASE}/paper/search", params=params, headers=self.headers)
        data = res.get("data", [])
        papers: List[Paper] = []
        for p in data:
            authors = [a.get("name", "") for a in (p.get("authors") or [])]
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")
            pdf_url = (p.get("openAccessPdf") or {}).get("url")
            if pdf_url and not _head(pdf_url):
                pdf_url = None
            papers.append(Paper(
                source="SemanticScholar",
                paper_id=str(p.get("paperId")),
                title=p.get("title") or "",
                abstract=p.get("abstract") or "",
                year=p.get("year"),
                authors=authors,
                venue=p.get("venue"),
                url=p.get("url"),
                pdf_url=pdf_url,
                doi=doi,
                citation_count=p.get("citationCount"),
            ))
        return papers


class IEEEXploreClient:
    BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def search(self, query: str, limit: int = 30) -> List[Paper]:
        if not self.api_key:
            return []
        params = {
            "apikey": self.api_key,
            "format": "json",
            "max_records": min(200, max(1, limit)),
            "sort_order": "desc",
            "sort_field": "publication_year",
            "querytext": query,
        }
        res = _get_json(self.BASE, params=params)
        articles = (res.get("articles") or [])
        out: List[Paper] = []
        for a in articles:
            year = None
            try:
                year = int(a.get("publication_year")) if a.get("publication_year") else None
            except Exception:
                year = None
            authors = []
            if a.get("authors") and a["authors"].get("authors"):
                authors = [au.get("full_name", "") for au in a["authors"]["authors"]]
            doi = a.get("doi")
            pdf_url = None
            if a.get("pdf_url"):
                pdf_url = a.get("pdf_url")
                if pdf_url and not _head(pdf_url):
                    pdf_url = None
            url = a.get("html_url") or a.get("pdf_url")
            citation_count = None
            try:
                citation_count = int((a.get("citing_paper_count") or 0))
            except Exception:
                citation_count = None
            out.append(Paper(
                source="IEEE",
                paper_id=a.get("article_number") or a.get("doi") or a.get("html_url") or "",
                title=a.get("title") or "",
                abstract=a.get("abstract") or "",
                year=year,
                authors=authors,
                venue=(a.get("publication_title") or a.get("publisher") or None),
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                citation_count=citation_count,
            ))
        return out


class SerpAPIGoogleScholarClient:
    BASE = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not self.api_key:
            return []
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": min(20, max(1, limit)),
            "hl": "en",
        }
        res = _get_json(self.BASE, params=params)
        results = (res.get("organic_results") or [])
        out: List[Paper] = []
        for r in results:
            title = r.get("title") or ""
            url = r.get("link")
            snippet = r.get("snippet") or ""
            # Parse year if present
            year = None
            pub_info = (r.get("publication_info") or {}).get("summary")
            if pub_info:
                for tok in pub_info.split():
                    if tok.isdigit() and len(tok) == 4:
                        try:
                            y = int(tok)
                            if 1900 < y <= NOW_YEAR:
                                year = y
                                break
                        except Exception:
                            pass
            citation_count = None
            if r.get("inline_links") and r["inline_links"].get("cited_by"):
                cited_str = r["inline_links"]["cited_by"].get("total")
                try:
                    citation_count = int(cited_str)
                except Exception:
                    citation_count = None
            out.append(Paper(
                source="GoogleScholar",
                paper_id=url or title,
                title=title,
                abstract=snippet,
                year=year,
                authors=[],
                venue=None,
                url=url,
                pdf_url=None,
                doi=None,
                citation_count=citation_count,
            ))
        return out


# ------------- Gemini (Google Generative AI) -------------

def _setup_gemini(api_key: str, model_name: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    embed = genai.embed_content
    return model, embed


def gemini_embed(embed_fn, text: str, task_type: str = "retrieval_query", model: str = "text-embedding-004") -> np.ndarray:
    try:
        res = embed_fn(model=model, content=text, task_type=task_type)
        vec = np.array(res["embedding"], dtype=np.float32)
        return vec
    except Exception as e:
        raise APIError(f"Gemini embedding failed: {e}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ------------- Core Pipeline -------------

def fetch_papers(query: str, max_papers: int, api_keys: Dict[str, Optional[str]]) -> List[Paper]:
    s2_key = api_keys.get("SEMANTIC_SCHOLAR_KEY")
    ieee_key = api_keys.get("IEEE_API_KEY")
    serp_key = api_keys.get("SERPAPI_KEY")

    s2 = SemanticScholarClient(s2_key) if s2_key else None
    ieee = IEEEXploreClient(ieee_key) if ieee_key else None
    serp = SerpAPIGoogleScholarClient(serp_key) if serp_key else None

    collected: List[Paper] = []

    if s2:
        console.rule("Fetching from Semantic Scholar")
        collected.extend(s2.search(query, limit=max_papers))

    if ieee_key:
        console.rule("Fetching from IEEE Xplore")
        collected.extend(ieee.search(query, limit=min(50, max_papers // 2 or 20)))

    if serp_key:
        console.rule("Fetching from Google Scholar (SerpAPI)")
        collected.extend(serp.search(query, limit=min(20, max_papers // 4 or 10)))

    papers = dedup_papers(collected)
    console.print(f"Collected {len(papers)} unique papers.")
    return papers


def exclude_papers(papers: List[Paper], exclude: List[Paper]) -> List[Paper]:
    """Remove papers from `papers` that match any in `exclude` by DOI or (title+year)."""
    exclude_keys = set(
        (p.doi or f"{p.title.strip().lower()}::{p.year or ''}") for p in exclude
    )
    return [p for p in papers if (p.doi or f"{p.title.strip().lower()}::{p.year or ''}") not in exclude_keys]

def dedup_papers(collected: List[Paper]) -> List[Paper]:
    dedup: Dict[str, Paper] = {}
    for p in collected:
        key = (p.doi or f"{p.title.strip().lower()}::{p.year or ''}")
        if key not in dedup:
            dedup[key] = p
        else:
            q = dedup[key]
            for f in ["abstract", "venue", "url", "pdf_url", "citation_count"]:
                if getattr(q, f) in (None, "") and getattr(p, f) not in (None, ""):
                    setattr(q, f, getattr(p, f))
    return list(dedup.values())


def score_and_rank(papers: List[Paper], topic: str, weights: Tuple[float, float, float], gemini_api_key: str, embed_model: str = "text-embedding-004", model_name: str = "gemini-2.5-flash") -> List[Paper]:
    w_rel, w_cit, w_rec = weights
    _, embed_fn = _setup_gemini(gemini_api_key, model_name=model_name)

    topic_vec = gemini_embed(embed_fn, topic, task_type="retrieval_query", model=embed_model)

    console.rule("Embedding & scoring")
    for p in tqdm(papers, desc="Scoring"):
        text = (p.title or "") + "\n\n" + (p.abstract or "")
        p_vec = gemini_embed(embed_fn, text[:7000], task_type="retrieval_document", model=embed_model)
        sim = cosine(topic_vec, p_vec)
        p.similarity = sim
        c = _log1p_normalized(p.citation_count)
        r = _recency_score(p.year)
        p.score = w_rel * sim + w_cit * c + w_rec * r

    papers.sort(key=lambda x: x.score, reverse=True)
    return papers


def build_context_chunks(papers: List[Paper], top_k: int) -> Tuple[str, str]:
    top = papers[:top_k]
    bibliography = []
    for i, p in enumerate(top, 1):
        authors = ", ".join(p.authors[:6]) + (" et al." if len(p.authors) > 6 else "")
        bib = f"[{i}] {p.title} — {authors} ({p.year or 'n.d.'}). {p.venue or ''}. DOI: {p.doi or 'N/A'}. Link: {p.url or p.pdf_url or 'N/A'}"
        bibliography.append(bib)

    context_chunks = []
    for idx, p in enumerate(top, 1):
        context_chunks.append(textwrap.dedent(f"""
        ### [{idx}] {p.title}
        - Venue: {p.venue or 'Unknown'} | Year: {p.year or 'n.d.'} | Citations: {p.citation_count or 0}
        - URL: {p.url or p.pdf_url or 'N/A'}
        - Abstract: {p.abstract or 'N/A'}
        """))

    return "\n".join(context_chunks), "\n".join(bibliography)


def generate_report(topic: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)

    context_str, bibliography_str = build_context_chunks(papers, top_k)

    sys_prompt = textwrap.dedent(f"""
    You are an expert research analyst. Given a topic and a set of top-ranked papers (with abstracts), write a THOROUGH, DETAILED, and COMPREHENSIVE research report with the following structure:

    1) Executive Summary (300-500 words)
       - Provide a broad, insightful overview of the field, highlighting the most significant findings, trends, and challenges.
       - Summarize the main contributions of the top papers, referencing them with [#] citations.

    2) In-Depth Background & Core Concepts (400-600 words)
       - Explain all relevant background, terminology, and foundational concepts in detail.
       - Include historical context, key definitions, and major theoretical frameworks.
       - Use clear, accessible language for non-experts.

    3) Comparative Literature Synthesis (600-900 words)
       - Analyze and compare the top papers in depth, discussing methodologies, datasets, benchmarks, and results.
       - Identify major research clusters, approaches, and their evolution over time.
       - Highlight consensus, controversies, and open debates, citing papers as [#].

    4) Critical Gap Analysis (300-500 words)
       - Identify and discuss methodological, data, evaluation, reproducibility, and scalability gaps in the literature.
       - Point out under-explored areas, limitations, and weaknesses, with specific paper references.

    5) Future Research Directions (300-500 words)
       - Propose prioritized, concrete, and measurable future research directions.
       - Suggest new methodologies, datasets, or evaluation strategies.
       - Discuss potential for interdisciplinary work and emerging trends.

    6) Risks, Ethics, and Limitations (200-400 words)
       - Analyze ethical, societal, and practical risks associated with the research area.
       - Discuss limitations of current approaches and possible negative impacts.

    7) Practical Applications and Tooling Landscape (200-400 words)
       - Survey real-world applications, tools, and systems based on the reviewed research.
       - Highlight industry adoption, open-source projects, and commercial products.

    8) Conclusion (150-300 words)
       - Synthesize the main insights, reiterate the importance of the topic, and summarize key takeaways.

    Rules:
    - Use clear section headings and subheadings.
    - Use inline numeric citations like [1], [2] that map to the provided bibliography.
    - If evidence is weak or missing, explicitly state so.
    - Be precise, avoid hand-waving, and support all claims with references.
    - Aim for a total length of 3000-5000 words, unless context is sparse.
    - Ensure the report is self-contained and highly informative for both experts and newcomers.
    """
    )

    user_prompt = f"Topic: {topic}\n\nTop Papers Context (ranked):\n" + context_str + "\n\nBibliography (use these citation indices):\n" + bibliography_str

    console.rule("Generating report with Gemini")
    resp = model.generate_content(
        [
            {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]},
        ],
        safety_settings=None,
        generation_config={
            "temperature": 0.6,
            "top_p": 0.9,
        }
    )
    try:
        md = resp.text
    except Exception as e:
        raise APIError(f"Gemini generation failed: {e}")

    # Prepend metadata table
    meta = Table(title="Top Papers Used", show_lines=False, box=box.MINIMAL_DOUBLE_HEAD)
    meta.add_column("#", style="bold")
    meta.add_column("Title")
    meta.add_column("Year")
    meta.add_column("Citations")
    meta.add_column("Score")
    for i, p in enumerate(papers[:top_k], 1):
        meta.add_row(str(i), p.title[:70], str(p.year or ""), str(p.citation_count or 0), f"{p.score:.3f}")
    console.print(meta)

    return md


def answer_question(question: str, report_md: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    """Answer a user question using the current report + top-k papers as context (no regeneration)."""
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)

    context_str, bibliography_str = build_context_chunks(papers, top_k)

    sys_prompt = textwrap.dedent("""
    You are a precise research assistant. Answer the user's question *only* using the provided context: the current report and the top papers. If information is not in context, say you don't have enough evidence rather than guessing. Cite papers with [#] indices where appropriate, and refer to report sections when helpful.
    """)

    user_prompt = (
        "Question: " + question + "\n\n" +
        "Current Report (excerpt allowed):\n" + report_md[:15000] + "\n\n" +
        "Top Papers Context (ranked):\n" + context_str + "\n\nBibliography:\n" + bibliography_str
    )

    resp = model.generate_content(
        [
            {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]},
        ],
        safety_settings=None,
        generation_config={"temperature": 0.4, "top_p": 0.9}
    )
    return getattr(resp, "text", "(No answer produced)")

def analyze_intent(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    sys_prompt = textwrap.dedent("""
    You are an intent classifier for a research assistant tool. The user input could be:
    1) Asking for further refinement of the report.
    2) Asking a specific question about the report.
    3) Expressing satisfaction and wanting to finish.

    Classify the intent into exactly one of these labels: refine, ask, accept.
    Respond only with the label: refine, ask, or accept.
    """)

    user_prompt = f"User input: {user_input}"

    try:
        resp = model.generate_content(
            [
                {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]},
            ],
            generation_config={"temperature": 0.0, "top_p": 0.9}
        )
        intent = resp.text.strip().lower()
        if intent not in ["refine", "ask", "accept"]:
            return "ask"  # fallback
        return intent
    except Exception:
        return "ask"

def generate_query_from_input(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    sys_prompt = textwrap.dedent("""
    Convert the user's feedback into a concise, plain-text search query suitable for finding academic papers.
    The output should be a single line of text with no markdown, explanation, or extra details.
    """)
    user_prompt = f"User feedback: {user_input}"

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}],
        generation_config={"temperature": 0.0, "top_p": 0.9}
    )
    return resp.text.strip()

def generate_question_from_input(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    sys_prompt = textwrap.dedent("""
    Convert the user's statement into a precise research question.
    Output only the question in plain text, without markdown, explanation, or extra context.
    """)
    user_prompt = f"User statement: {user_input}"

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}],
        generation_config={"temperature": 0.0, "top_p": 0.9}
    )
    return resp.text.strip()


# ------------- I/O -------------

def save_outputs(papers: List[Paper], report_md: str, out_path: str, csv_path: str, json_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    console.print(f"[green]Saved report:[/green] {out_path}")

    if papers:
        # CSV
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(papers[0].to_row().keys()))
            writer.writeheader()
            for p in papers:
                writer.writerow(p.to_row())
        console.print(f"[green]Saved rankings CSV:[/green] {csv_path}")

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([p.to_row() for p in papers], f, ensure_ascii=False, indent=2)
        console.print(f"[green]Saved raw JSON:[/green] {json_path}")


# ------------- Interactive Loop -------------

def interactive_loop(topic: str,
                     papers: List[Paper],
                     args: argparse.Namespace,
                     api_keys: Dict[str, Optional[str]],
                     gemini_key: str,
                     report_md: str) -> Tuple[List[Paper], str]:
    """REPL loop that uses intent and input-driven query/question generation."""
    iteration = 1
    current_topic = topic
    current_papers = papers
    current_report = report_md

    console.print(Panel.fit("You can now type your next input. The system will infer whether you want to refine, ask questions, or accept the report.", style="bold cyan"))

    while True:
        user_input = Prompt.ask("[bold yellow]Enter your feedback or question[/bold yellow]")
        if not user_input.strip():
            console.print("[yellow]Please enter some text or 'exit' to quit.[/yellow]")
            continue
        if user_input.strip().lower() == "exit":
            console.print("[yellow]Exiting without further changes.[/yellow]")
            return current_papers, current_report

        intent = analyze_intent(user_input, gemini_key, model_name=args.model)

        if intent == "accept":
            console.print("[bold green]You're satisfied. Finalizing the report.[/bold green]")
            return current_papers, current_report

        elif intent == "ask":
            question = generate_question_from_input(user_input, gemini_key, args.model)
            console.rule("Answering your question")
            answer = answer_question(question, current_report, current_papers, args.top_k, gemini_key, model_name=args.model)
            console.print(Panel.fit(answer, title="Answer", border_style="magenta"))
            continue

        elif intent == "refine":
            iteration += 1
            query = generate_query_from_input(user_input, gemini_key, args.model)
            console.rule(f"Refinement #{iteration}: fetching additional papers for: {query}")

            new_papers = fetch_papers(query, max(10, args.max_papers // 2), api_keys)
            new_papers = exclude_papers(new_papers, current_papers[:args.top_k])
            merged = dedup_papers(current_papers + new_papers)
            reranked = score_and_rank(merged, query, tuple(args.weights), gemini_key, embed_model=args.embed_model, model_name=args.model)

            table = Table(title=f"Top 10 Papers after Refinement #{iteration}", box=box.SIMPLE_HEAVY)
            table.add_column("#")
            table.add_column("Title")
            table.add_column("Year")
            table.add_column("Cites")
            table.add_column("Rel")
            table.add_column("Score")
            for i, p in enumerate(reranked[:10], 1):
                table.add_row(str(i), p.title[:80], str(p.year or ""), str(p.citation_count or 0), f"{p.similarity:.3f}", f"{p.score:.3f}")
            console.print(table)

            current_papers = reranked
            current_report = generate_report(query, current_papers, args.top_k, gemini_key, model_name=args.model)

            # Save only the final version
            save_outputs(current_papers, current_report, args.out, args.csv, args.json)


# ------------- CLI -------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deep Research CLI using Gemini + scholarly APIs (terminal only)")
    ap.add_argument("topic", type=str, help="Research topic, e.g., 'Graph Neural Networks for Traffic Forecasting'")
    ap.add_argument("--max-papers", type=int, default=80, help="Max papers to fetch (after dedup)")
    ap.add_argument("--top-k", type=int, default=25, help="Top K papers to include in report context")
    ap.add_argument("--weights", nargs=3, type=float, default=[0.4, 0.25, 0.35], metavar=("W_REL", "W_CIT", "W_REC"), help="Weights for relevance, citations, recency (sum not required but recommended)")
    ap.add_argument("--embed-model", type=str, default="text-embedding-004", help="Gemini embedding model")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini generation model (e.g., gemini-2.5-flash)")
    ap.add_argument("--out", type=str, default="report.md", help="Output Markdown path")
    ap.add_argument("--csv", type=str, default="ranked_papers.csv", help="Output CSV path")
    ap.add_argument("--json", type=str, default="ranked_papers.json", help="Output JSON path")
    ap.add_argument("--no-interactive", action="store_true", help="Disable interactive refinement/Q&A loop")
    return ap.parse_args()


def main():
    args = parse_args()

    gemini_key = "AIzaSyDICa23cj0UFSvU2ZsTrdGmtRvS3zkGn0g"
    if not gemini_key:
        console.print("[red]GEMINI_API_KEY is required in environment.[/red]")
        sys.exit(1)

    api_keys = {
        "SEMANTIC_SCHOLAR_KEY": os.getenv("SEMANTIC_SCHOLAR_KEY"),
        "IEEE_API_KEY": os.getenv("IEEE_API_KEY"),
        "SERPAPI_KEY": "90d7f2c5ae56195b39b96f126064d594addf1d8f8693de857f69e0c54e2090d6",
    }

    console.rule("Deep Research CLI — Interactive")
    console.print(f"[bold]Topic:[/bold] {args.topic}")

    if not api_keys["SEMANTIC_SCHOLAR_KEY"]:
        console.print("[yellow]Warning: SEMANTIC_SCHOLAR_KEY not provided. Semantic Scholar will be skipped.[/yellow]")
    if not api_keys["IEEE_API_KEY"]:
        console.print("[yellow]Warning: IEEE_API_KEY not provided. IEEE Xplore will be skipped.[/yellow]")
    if not api_keys["SERPAPI_KEY"]:
        console.print("[yellow]Warning: SERPAPI_KEY not provided. Google Scholar will be skipped.[/yellow]")

    papers = fetch_papers(args.topic, args.max_papers, api_keys)
    if not papers:
        console.print("[red]No papers found. Try broadening the query or providing API keys.[/red]")
        sys.exit(2)

    ranked = score_and_rank(papers, args.topic, tuple(args.weights), gemini_key, embed_model=args.embed_model, model_name=args.model)

    # Pretty print top 10 to terminal
    table = Table(title="Top 10 Papers", box=box.SIMPLE_HEAVY)
    table.add_column("#")
    table.add_column("Title")
    table.add_column("Year")
    table.add_column("Cites")
    table.add_column("Rel")
    table.add_column("Score")
    for i, p in enumerate(ranked[:10], 1):
        table.add_row(str(i), p.title[:80], str(p.year or ""), str(p.citation_count or 0), f"{p.similarity:.3f}", f"{p.score:.3f}")
    console.print(table)

    report_md = generate_report(args.topic, ranked, args.top_k, gemini_key, model_name=args.model)
    save_outputs(ranked, report_md, args.out, args.csv, args.json)

    if not args.no_interactive:
        final_papers, final_report = interactive_loop(args.topic, ranked, args, api_keys, gemini_key, report_md)
        save_outputs(final_papers, final_report, args.out, args.csv, args.json)


if __name__ == "__main__":
    main()
