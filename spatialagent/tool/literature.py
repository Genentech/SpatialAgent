"""
Literature Research Tools

Tools for querying scientific literature databases and extracting
content from papers and web pages.

Adapted from Biomni literature tools.
"""

import os
import re
import time
from io import BytesIO
from typing import Annotated
from urllib.parse import urljoin

from langchain_core.tools import tool
from pydantic import Field


# =============================================================================
# Tool 1: Query PubMed
# =============================================================================

def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text (e.g., <sup>, <sub>, <i>)."""
    return re.sub(r'<[^>]+>', '', text)


@tool
def query_pubmed(
    query: Annotated[str, Field(description="Search query string for PubMed")],
    max_papers: Annotated[int, Field(ge=1, le=50, description="Maximum number of papers to retrieve")] = 10,
) -> str:
    """Query PubMed for papers based on the provided search query.

    Returns titles, abstracts, and journal names of matching papers.
    Useful for finding relevant literature on biological topics.

    Examples:
        - query_pubmed({"query": "spatial transcriptomics liver cancer", "max_papers": 5})
        - query_pubmed({"query": "single cell RNA-seq hepatocyte markers"})
    """
    from Bio import Entrez

    Entrez.email = "spatialagent@example.com"
    max_retries = 3

    try:
        # Search PubMed
        current_query = query
        ids = []
        retries = 0

        while not ids and retries <= max_retries:
            if retries > 0:
                # Simplify query by removing the last word
                words = current_query.split()
                if len(words) > 1:
                    current_query = " ".join(words[:-1])
                time.sleep(1)  # Rate limiting

            handle = Entrez.esearch(db="pubmed", term=current_query, retmax=max_papers)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            retries += 1

        if not ids:
            return "No papers found on PubMed after multiple query attempts."

        # Fetch paper details
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        results = []
        for i, article in enumerate(records.get("PubmedArticle", []), 1):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})

            # Extract PubMed ID
            pubmed_id = str(medline.get("PMID", "N/A"))

            # Extract and clean title
            title = str(article_data.get("ArticleTitle", "N/A"))
            title = _strip_html_tags(title)

            # Extract journal name
            journal_info = article_data.get("Journal", {})
            journal = journal_info.get("Title", "N/A")

            # Extract and clean abstract
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            if abstract_parts:
                abstract = " ".join(str(part) for part in abstract_parts)
                abstract = _strip_html_tags(abstract)
            else:
                abstract = "N/A"

            # Truncate long abstracts (1500 chars captures most full abstracts)
            if abstract and len(abstract) > 1500:
                abstract = abstract[:1500] + "..."

            results.append(
                f"Paper {i}:\n  Title: {title}\n  Journal: {journal}\n  PubMed ID: {pubmed_id}\n  Abstract: {abstract}"
            )

        return f"Found {len(results)} papers on PubMed:\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error querying PubMed: {e}"


# =============================================================================
# Tool 2: Query arXiv
# =============================================================================

@tool
def query_arxiv(
    query: Annotated[str, Field(description="Search query string for arXiv")],
    max_papers: Annotated[int, Field(ge=1, le=50, description="Maximum number of papers to retrieve")] = 10,
) -> str:
    """Query arXiv for papers based on the provided search query.

    Returns titles and summaries of matching papers.
    Useful for finding preprints on computational biology, machine learning, etc.

    Examples:
        - query_arxiv({"query": "spatial transcriptomics deep learning", "max_papers": 5})
        - query_arxiv({"query": "single cell foundation model"})
    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for i, paper in enumerate(client.results(search), 1):
            title = paper.title
            summary = paper.summary
            arxiv_id = paper.entry_id.split('/')[-1]
            published = paper.published.strftime('%Y-%m-%d') if paper.published else 'N/A'

            # Truncate long summaries (1500 chars captures most full abstracts)
            if summary and len(summary) > 1500:
                summary = summary[:1500] + "..."

            results.append(f"Paper {i}:\n  Title: {title}\n  arXiv ID: {arxiv_id}\n  Published: {published}\n  Summary: {summary}")

        if results:
            return f"Found {len(results)} papers on arXiv:\n\n" + "\n\n".join(results)
        else:
            return "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


# =============================================================================
# Tool 3: Query Google Scholar (DISABLED - use query_pubmed or query_arxiv)
# =============================================================================
# Commented out due to reliability issues:
# - Google Scholar has aggressive rate limits and may hang indefinitely
# - No timeout mechanism - can block agent execution
# - May require CAPTCHA verification
# Use query_pubmed or query_arxiv instead.
#
# @tool
# def query_scholar(
#     query: Annotated[str, Field(description="Search query string for Google Scholar")],
# ) -> str:
#     """Query Google Scholar for papers based on the provided search query."""
#     from scholarly import scholarly
#
#     try:
#         search_query = scholarly.search_pubs(query)
#         result = next(search_query, None)
#         if result:
#             bib = result.get('bib', {})
#             title = bib.get('title', 'N/A')
#             year = bib.get('pub_year', 'N/A')
#             venue = bib.get('venue', 'N/A')
#             abstract = bib.get('abstract', 'N/A')
#             if abstract and len(abstract) > 500:
#                 abstract = abstract[:500] + "..."
#             return f"Found paper on Google Scholar:\n\nTitle: {title}\nYear: {year}\nVenue: {venue}\nAbstract: {abstract}"
#         else:
#             return "No results found on Google Scholar."
#     except Exception as e:
#         return f"Error querying Google Scholar: {e}"


# =============================================================================
# Tool 4: Semantic Scholar Search
# =============================================================================

@tool
def search_semantic_scholar(
    query: Annotated[str, Field(description="Search query string for academic papers")],
    max_papers: Annotated[int, Field(ge=1, le=20, description="Maximum number of papers to retrieve")] = 10,
) -> str:
    """Search Semantic Scholar for academic papers.

    Returns titles, abstracts, authors, year, and citation counts.
    Useful for finding academic papers with citation information.
    Free API, no key required.

    Examples:
        - search_semantic_scholar({"query": "spatial transcriptomics deep learning", "max_papers": 5})
        - search_semantic_scholar({"query": "single cell RNA-seq clustering"})
    """
    import requests

    try:
        # Semantic Scholar API endpoint
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_papers,
            "fields": "paperId,title,abstract,authors,year,citationCount,journal,url"
        }
        headers = {"User-Agent": "SpatialAgent/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait a moment and try again."

        if response.status_code != 200:
            return f"Error querying Semantic Scholar: HTTP {response.status_code}"

        data = response.json()
        papers = data.get("data", [])

        if not papers:
            return "No papers found on Semantic Scholar."

        results = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "N/A")
            abstract = paper.get("abstract", "N/A") or "N/A"
            year = paper.get("year", "N/A")
            citations = paper.get("citationCount", 0)
            paper_id = paper.get("paperId", "N/A")

            # Get authors (limit to first 3)
            authors_list = paper.get("authors", [])
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list[:3]]
                authors = ", ".join(author_names)
                if len(authors_list) > 3:
                    authors += f" et al. ({len(authors_list)} authors)"
            else:
                authors = "N/A"

            # Get journal
            journal_info = paper.get("journal")
            journal = journal_info.get("name", "N/A") if journal_info else "N/A"

            # Truncate long abstracts
            if abstract and len(abstract) > 1500:
                abstract = abstract[:1500] + "..."

            results.append(
                f"Paper {i}:\n"
                f"  Title: {title}\n"
                f"  Authors: {authors}\n"
                f"  Year: {year} | Citations: {citations}\n"
                f"  Journal: {journal}\n"
                f"  Semantic Scholar ID: {paper_id}\n"
                f"  Abstract: {abstract}"
            )

        return f"Found {len(results)} papers on Semantic Scholar:\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error querying Semantic Scholar: {e}"


# =============================================================================
# Tool 5: Web Search (DuckDuckGo) - DISABLED
# =============================================================================
# Commented out because:
# - DuckDuckGo is blocked on many institutional networks
# - Overlaps with query_pubmed and search_semantic_scholar for academic use
# - Use extract_url_content for specific documentation pages instead
#
# @tool
# def search_duckduckgo(
#     query: Annotated[str, Field(description="Search query string")],
#     num_results: Annotated[int, Field(ge=1, le=10, description="Number of results to return")] = 5,
# ) -> str:
#     """Search the web for information using DuckDuckGo.
#
#     Returns titles, URLs, and descriptions of matching web pages.
#     Useful for finding protocols, documentation, or general information.
#
#     Examples:
#         - search_duckduckgo({"query": "scanpy preprocessing tutorial", "num_results": 5})
#         - search_duckduckgo({"query": "10x Genomics Visium protocol"})
#     """
#     import requests
#     from bs4 import BeautifulSoup
#
#     try:
#         # Use DuckDuckGo lite HTML version (reliable, no API key needed)
#         url = "https://lite.duckduckgo.com/lite/"
#         headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
#
#         resp = requests.post(url, data={"q": query}, headers=headers, timeout=15)
#         soup = BeautifulSoup(resp.text, "html.parser")
#
#         results = []
#         current_result = {}
#
#         # DuckDuckGo lite uses table rows for results
#         for row in soup.find_all("tr"):
#             # Link row
#             link = row.find("a", class_="result-link")
#             if link:
#                 if current_result and "title" in current_result:
#                     results.append(current_result)
#                     if len(results) >= num_results:
#                         break
#                 current_result = {
#                     "title": link.get_text(strip=True),
#                     "url": link.get("href", ""),
#                 }
#             # Description row
#             snippet = row.find("td", class_="result-snippet")
#             if snippet and current_result:
#                 current_result["description"] = snippet.get_text(strip=True)
#
#         # Add last result if not yet added
#         if current_result and "title" in current_result and len(results) < num_results:
#             results.append(current_result)
#
#         if results:
#             formatted = []
#             for r in results:
#                 formatted.append(
#                     f"Title: {r.get('title', 'No title')}\n"
#                     f"URL: {r.get('url', 'No URL')}\n"
#                     f"Description: {r.get('description', 'No description')}"
#                 )
#             return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)
#         else:
#             return "No results found."
#     except Exception as e:
#         return f"Error performing web search: {e}"


# =============================================================================
# Tool 5: Extract URL Content
# =============================================================================
# Comparison with Anthropic's web_fetch server tool (2025-01):
#   | Metric              | extract_url_content | Anthropic web_fetch |
#   |---------------------|---------------------|---------------------|
#   | Wikipedia           | 0.7s, 10K chars     | 8.8s, 38K tokens    |
#   | PubMed (JS-heavy)   | 0.6s, 11K chars ✓   | 11.3s, FAILED       |
#   | Cost                | Free                | Free (token cost)   |
#   | JS rendering        | No (but gets HTML)  | No                  |
#   | PDF support         | No (use extract_pdf)| Yes (native)        |
# Verdict: Our tool is faster and works on more sites (including PubMed).
# =============================================================================

@tool
def extract_url_content(
    url: Annotated[str, Field(description="URL of the webpage to extract content from")],
    max_chars: Annotated[int, Field(ge=1000, le=100000, description="Maximum characters to return")] = 10000,
) -> str:
    """Extract the text content of a webpage.

    Useful for reading documentation, protocols, or online articles.

    Examples:
        - extract_url_content({"url": "https://scanpy.readthedocs.io/en/stable/tutorials.html"})
        - extract_url_content({"url": "https://pubmed.ncbi.nlm.nih.gov/12345/", "max_chars": 20000})
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)

        # Check if the response is in text format
        content_type = response.headers.get("Content-Type", "")
        if "text/plain" in content_type or "application/json" in content_type:
            return response.text.strip()[:max_chars]

        # If it's HTML, use BeautifulSoup to parse
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to find main content first, fallback to body
        content = soup.find("main") or soup.find("article") or soup.body

        if not content:
            return "Could not extract content from the webpage."

        # Remove unwanted elements
        for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            element.decompose()

        # Extract text with better formatting
        paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        cleaned_text = []

        for p in paragraphs:
            text = p.get_text().strip()
            if text:  # Only add non-empty paragraphs
                cleaned_text.append(text)

        result = "\n\n".join(cleaned_text)

        # Limit output size
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n... (content truncated)"

        return result if result else "No text content found on the page."

    except Exception as e:
        return f"Error extracting content from URL: {e}"


# =============================================================================
# Tool 6: Extract PDF Content
# =============================================================================

@tool
def extract_pdf_content(
    url: Annotated[str, Field(description="URL of the PDF file to extract text from")],
) -> str:
    """Extract the text content of a PDF file given its URL.

    Useful for reading papers, supplementary materials, or documentation in PDF format.

    Examples:
        - extract_pdf_content({"url": "https://example.com/paper.pdf"})
    """
    import requests
    import PyPDF2

    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})

        # Check if we actually got a PDF file
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Extract text with PyPDF2
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        # Limit output size
        if len(text) > 8000:
            text = text[:8000] + "\n\n... (content truncated)"

        return text

    except Exception as e:
        return f"Error extracting text from PDF: {e}"


# =============================================================================
# Tool 7: Fetch Supplementary Info from DOI
# =============================================================================

@tool
def fetch_supplementary_from_doi(
    doi: Annotated[str, Field(description="The paper DOI (e.g., '10.1038/s41586-021-03775-x')")],
    output_dir: Annotated[str, Field(description="Directory to save supplementary files")] = "supplementary_info",
) -> str:
    """Fetch paper content and supplementary information given a DOI.

    Extracts: title, abstract, full text (if available), and supplementary material links.
    This is the primary tool for accessing paper content from a DOI.

    Examples:
        - fetch_supplementary_from_doi({"doi": "10.1038/s41586-021-03775-x"})
        - fetch_supplementary_from_doi({"doi": "10.1016/j.cell.2021.04.048"})
    """
    import requests
    import time
    from bs4 import BeautifulSoup

    research_log = []
    research_log.append(f"Fetching paper content for DOI: {doi}")

    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        # Retry logic for transient errors
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.get(crossref_url, headers=headers, timeout=30, allow_redirects=True)
            if response.status_code < 500:
                break
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        if response.status_code != 200:
            return f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"

        publisher_url = response.url
        research_log.append(f"Publisher URL: {publisher_url}")

        # Parse page content
        soup = BeautifulSoup(response.content, "html.parser")

        # === Extract Paper Title ===
        title = None
        title_selectors = [
            "h1.article-title", "h1.c-article-title", "h1.citation__title",
            "h1.core-title", "h1#title", "meta[name='citation_title']",
            "meta[name='dc.title']", "title"
        ]
        for selector in title_selectors:
            if selector.startswith("meta"):
                elem = soup.select_one(selector)
                if elem:
                    title = elem.get("content", "")
                    break
            else:
                elem = soup.select_one(selector)
                if elem:
                    title = elem.get_text(strip=True)
                    break
        if title:
            research_log.append(f"\n=== TITLE ===\n{title}")

        # === Extract Abstract ===
        abstract = None
        abstract_selectors = [
            "section#abstract", "div#abstract", "div.abstract", "section.abstract",
            "div[class*='abstract']", "p.abstract", "div.c-article-section__content",
            "meta[name='citation_abstract']", "meta[name='dc.description']"
        ]
        for selector in abstract_selectors:
            if selector.startswith("meta"):
                elem = soup.select_one(selector)
                if elem:
                    abstract = elem.get("content", "")
                    break
            else:
                elem = soup.select_one(selector)
                if elem:
                    abstract = elem.get_text(separator=" ", strip=True)
                    break
        if abstract:
            research_log.append(f"\n=== ABSTRACT ===\n{abstract[:3000]}")

        # === Extract Full Text Sections (Results, Methods, etc.) ===
        full_text_sections = []
        section_selectors = [
            "section.c-article-section", "div.article-section", "section[id*='sec']",
            "div.section", "div.NLM_sec", "div[class*='section']"
        ]
        for selector in section_selectors:
            sections = soup.select(selector)
            for sec in sections[:10]:  # Limit to first 10 sections
                heading = sec.find(["h2", "h3", "h4"])
                heading_text = heading.get_text(strip=True) if heading else "Section"
                content = sec.get_text(separator=" ", strip=True)[:2000]
                if len(content) > 100:  # Only include substantial sections
                    full_text_sections.append(f"\n--- {heading_text} ---\n{content}")
            if full_text_sections:
                break

        if full_text_sections:
            research_log.append(f"\n=== FULL TEXT EXCERPTS ===")
            research_log.extend(full_text_sections[:5])  # Limit output

        # === Check for PMC Full Text ===
        pmc_link = None
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "ncbi.nlm.nih.gov/pmc" in href or "/pmc/articles/" in href:
                pmc_link = href if href.startswith("http") else urljoin(publisher_url, href)
                break
        if pmc_link:
            research_log.append(f"\n=== PMC FULL TEXT AVAILABLE ===\n{pmc_link}")

        # === Find Supplementary Material Links ===
        supplementary_links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            text = link.get_text().lower()
            if any(kw in text for kw in ["supplementary", "supplemental", "appendix", "supporting"]):
                full_url = urljoin(publisher_url, href)
                if full_url not in supplementary_links:
                    supplementary_links.append(full_url)

        if supplementary_links:
            research_log.append(f"\n=== SUPPLEMENTARY MATERIALS ===")
            for link in supplementary_links[:5]:
                research_log.append(f"- {link}")

        # === Summary ===
        if not abstract and not full_text_sections:
            research_log.append(f"\n[WARNING] Could not extract paper content. Try web_search with the DOI for more details.")

        return "\n".join(research_log)

    except Exception as e:
        return f"Error fetching paper content: {e}\n\n[TIP] Try using web_search with query: '{doi} [specific keywords]'"


# =============================================================================
# Tool 8: Provider Web Search (Anthropic, OpenAI, Google)
# =============================================================================

@tool
def web_search(
    query: str,
    model: str = None,
    max_results: int = 5,
    allowed_domains: list = None,
) -> dict:
    """Search the web using server-side web search from Anthropic, OpenAI, or Google.

    The provider is automatically detected from the model name:
        - "claude-*" → Anthropic
        - "gpt-*", "o3-*", "o4-*" → OpenAI
        - "gemini-*" → Google

    Args:
        query: The search query string.
        model: Model name (e.g., "gemini-3-flash-preview"). Provider is inferred from this.
            If not set, falls back to DEFAULT_WEB_SEARCH_PROVIDER env var or available API keys.
        max_results: Maximum number of search results (where supported).
        allowed_domains: Optional list of domains to restrict search to.

    Returns:
        dict with keys:
            - provider: The provider used
            - model: The model used
            - response: The text response with search results
            - citations: List of cited URLs (where available)
            - error: Error message if failed, None otherwise

    Examples:
        >>> result = web_search("latest scanpy version")  # Uses configured model
        >>> result = web_search("COVID-19 updates", model="gpt-5")
        >>> result = web_search("spatial methods", model="claude-sonnet-4-5-20250929")

    Note:
        For academic paper search, use search_semantic_scholar() or query_pubmed() instead.
    """
    import os

    # Detect provider from model name
    provider = None
    if model:
        model_lower = model.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            provider = "anthropic"
        elif any(x in model_lower for x in ["gpt-", "gpt5", "o3", "o4"]):
            provider = "openai"
        elif "gemini" in model_lower:
            provider = "google"

    # Fallback: Check explicit default env var
    if not provider:
        default_provider = os.environ.get("DEFAULT_WEB_SEARCH_PROVIDER", "").lower()
        if default_provider in ["anthropic", "openai", "google"]:
            provider = default_provider

    # Fallback: Auto-detect based on available API keys
    if not provider:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
            model = None  # Don't pass local LLM model name to Google
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            model = None
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model = None
        else:
            return {
                "provider": None,
                "model": model,
                "response": None,
                "citations": [],
                "error": "Cannot determine web search provider. Either: "
                         "(1) Pass a model name like 'claude-*', 'gpt-*', or 'gemini-*', "
                         "(2) Set DEFAULT_WEB_SEARCH_PROVIDER env var, or "
                         "(3) Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            }

    result = {
        "provider": provider,
        "model": model,
        "response": None,
        "citations": [],
        "error": None
    }

    try:
        if provider == "anthropic":
            result = _anthropic_web_search(query, model, max_results, allowed_domains)
        elif provider == "openai":
            result = _openai_web_search(query, model, allowed_domains)
        elif provider == "google":
            result = _google_web_search(query, model)
        else:
            result["error"] = f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'google'."
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _anthropic_web_search(
    query: str,
    model: str = None,
    max_results: int = 5,
    allowed_domains: list = None,
) -> dict:
    """Execute web search using Anthropic's server-side tool."""
    import anthropic

    client = anthropic.Anthropic()
    model = model or "claude-sonnet-4-5-20250929"

    # Build tool config
    tool_config = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_results
    }
    if allowed_domains:
        tool_config["allowed_domains"] = allowed_domains

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        tools=[tool_config],
        messages=[{"role": "user", "content": query}]
    )

    # Extract response text and citations
    text_parts = []
    citations = []

    for block in response.content:
        if hasattr(block, 'text'):
            text_parts.append(block.text)
            # Extract citations if present
            if hasattr(block, 'citations') and block.citations:
                for cite in block.citations:
                    if hasattr(cite, 'url'):
                        citations.append({
                            "url": cite.url,
                            "title": getattr(cite, 'title', ''),
                            "text": getattr(cite, 'cited_text', '')[:200] if hasattr(cite, 'cited_text') else ''
                        })

    # Get search count from usage
    search_count = 0
    if hasattr(response, 'usage') and hasattr(response.usage, 'server_tool_use'):
        search_count = getattr(response.usage.server_tool_use, 'web_search_requests', 0)

    return {
        "provider": "anthropic",
        "model": model,
        "response": "\n".join(text_parts),
        "citations": citations,
        "searches_used": search_count,
        "error": None
    }


def _openai_web_search(
    query: str,
    model: str = None,
    allowed_domains: list = None,
) -> dict:
    """Execute web search using OpenAI's Responses API."""
    from openai import OpenAI

    client = OpenAI()
    model = model or "gpt-4o"

    # Build tool config
    tool_config = {"type": "web_search"}
    if allowed_domains:
        tool_config["filters"] = {"allowed_domains": allowed_domains}

    response = client.responses.create(
        model=model,
        tools=[tool_config],
        input=query
    )

    # Extract citations from annotations
    citations = []
    if hasattr(response, 'output') and response.output:
        for item in response.output:
            if hasattr(item, 'content') and item.content:
                for content in item.content:
                    if hasattr(content, 'annotations') and content.annotations:
                        for ann in content.annotations:
                            if hasattr(ann, 'url'):
                                citations.append({
                                    "url": ann.url,
                                    "title": getattr(ann, 'title', ''),
                                    "text": ""
                                })

    return {
        "provider": "openai",
        "model": model,
        "response": response.output_text if hasattr(response, 'output_text') else str(response),
        "citations": citations,
        "error": None
    }


def _google_web_search(
    query: str,
    model: str = None,
) -> dict:
    """Execute web search using Google's Gemini with Google Search."""
    import os
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model = model or "gemini-3-flash-preview"

    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

    # Extract grounding metadata for citations
    citations = []
    search_queries = []

    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            gm = candidate.grounding_metadata
            # Get search queries used
            if hasattr(gm, 'web_search_queries'):
                search_queries = list(gm.web_search_queries) if gm.web_search_queries else []
            # Get grounding chunks (sources)
            if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        citations.append({
                            "url": getattr(chunk.web, 'uri', ''),
                            "title": getattr(chunk.web, 'title', ''),
                            "text": ""
                        })

    return {
        "provider": "google",
        "model": model,
        "response": response.text if hasattr(response, 'text') else str(response),
        "citations": citations,
        "search_queries": search_queries,
        "error": None
    }
