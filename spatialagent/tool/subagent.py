"""
Subagent Tools for SpatialAgent.

These tools act as autonomous subagents that perform multi-step analysis tasks,
including reading files, interpreting images, and synthesizing comprehensive reports.
"""

import os
import json
import base64
import pandas as pd
from typing import Annotated
from os.path import exists
from langchain_core.tools import tool
from pydantic import Field

# Default model for subagent LLM calls (fallback if agent model not set)
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _get_subagent_model() -> str:
    """Get the model to use for subagent calls.

    Uses the main agent's model if available, otherwise falls back to DEFAULT_MODEL.
    Handles cases where model name resolution failed (returns "unknown").
    """
    try:
        from ..agent import get_agent_model
        model = get_agent_model()
        # Fallback if model is empty or resolution failed ("unknown")
        if not model or model == "unknown":
            return DEFAULT_MODEL
        return model
    except ImportError:
        return DEFAULT_MODEL


def _resize_image_if_needed(image_path: str, max_size_bytes: int = 4_000_000) -> bytes:
    """Resize image if file size exceeds API limit.

    Claude API has a 5MB per-image limit. We resize to stay under that.
    File size scales roughly with pixel count, so we reduce dimensions proportionally.

    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum file size in bytes (default 4MB for safety margin)

    Returns:
        Image bytes (resized if necessary, original if not)
    """
    from PIL import Image
    import io
    import math

    with open(image_path, "rb") as f:
        original_bytes = f.read()

    if len(original_bytes) <= max_size_bytes:
        return original_bytes

    # Need to resize - scale down proportionally
    with Image.open(image_path) as img:
        scale = math.sqrt(max_size_bytes / len(original_bytes))
        new_size = (max(int(img.width * scale), 100), max(int(img.height * scale), 100))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format=img.format or 'PNG')
        print(f"Resized image: {img.width}x{img.height} -> {new_size[0]}x{new_size[1]}")
        return buffer.getvalue()


# =============================================================================
# Deep Research Report Subagent
# =============================================================================

@tool
def report_subagent(
    user_query: Annotated[str, Field(description="The research question being investigated (e.g., 'How does cellular composition change during disease progression?')")],
    data_info: Annotated[str, Field(description="Brief dataset description including species, tissue, technology, and conditions (e.g., 'MERFISH spatial transcriptomics of mouse colon, 4 disease stages')")],
    save_path: Annotated[str, Field(description="Directory path containing analysis outputs (figures, CSVs, observation_log.jsonl)")] = "./experiments",
) -> str:
    """Generate a publication-quality research report by analyzing all saved artifacts.

    IMPORTANT: Call this tool with exactly 3 parameters as keyword arguments.

    Example usage:
        report_subagent(
            user_query="How does the cellular microenvironment change during colitis?",
            data_info="MERFISH mouse colon, 50k cells, 4 disease stages (Healthy, DSS3, DSS9, DSS21)",
            save_path="./experiments/my_analysis/"
        )

    What this tool does automatically:
    1. Reads observation_log.jsonl from save_path for analysis history
    2. Discovers and interprets ALL figures (PNG/JPG) using vision LLM
    3. Reads and summarizes CSV data tables for statistics
    4. Extracts insights from JSON summary files
    5. Generates a comprehensive markdown report with:
       - Executive summary
       - Introduction and background
       - Methods documentation
       - Results with figure references
       - Discussion and interpretation
       - Conclusions and future directions

    Output: Saves report to {save_path}/deep_research_report.md

    Note: Do NOT pass dictionaries or extra parameters like 'key_findings',
    'figure_descriptions', or 'methods_summary' - these are auto-discovered.

    Returns:
        str: Summary of report generation with file path
    """
    from ..agent import make_llm
    import glob

    print("🔬 Starting Deep Research Report Subagent...")

    model = _get_subagent_model()
    print(f"   Using model: {model}")
    llm = make_llm(model, stop_sequences=[], max_tokens=16384)  # Higher limit for long reports

    output_path = f"{save_path}/deep_research_report.md"

    # =========================================================================
    # PASS 1: Inventory all artifacts
    # =========================================================================
    print("📋 Pass 1: Inventorying all artifacts...")

    # Read accumulated observations
    observation_log_path = f"{save_path}/observation_log.jsonl"
    observations = []
    if exists(observation_log_path):
        with open(observation_log_path, 'r') as f:
            for line in f:
                try:
                    observations.append(json.loads(line.strip()))
                except:
                    pass
    print(f"   Found {len(observations)} logged observations")

    # Discover all files
    figures = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg']:
        figures.extend(glob.glob(f"{save_path}/**/{ext}", recursive=True))
    figures = sorted(set(figures))
    print(f"   Found {len(figures)} figures")

    csv_files = sorted(glob.glob(f"{save_path}/**/*.csv", recursive=True))
    print(f"   Found {len(csv_files)} CSV files")

    json_files = [f for f in sorted(glob.glob(f"{save_path}/**/*.json", recursive=True))
                  if 'observation_log' not in f]
    print(f"   Found {len(json_files)} JSON files")

    # =========================================================================
    # PASS 2: Interpret ALL figures with vision LLM
    # =========================================================================
    print(f"🖼️ Pass 2: Interpreting {min(len(figures), 15)} figures with vision LLM...")

    figure_analyses = {}
    for i, fig_path in enumerate(figures[:15]):  # Limit to 15 most important figures
        fig_name = os.path.basename(fig_path)
        print(f"   Analyzing {fig_name}...")

        try:
            # Resize if needed
            image_bytes = _resize_image_if_needed(fig_path)
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            ext = fig_path.lower().split('.')[-1]
            mime_type = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext, 'image/png')

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"""Analyze this scientific figure from a spatial transcriptomics analysis.
Figure filename: {fig_name}
Research context: {user_query[:500]}
Dataset: {data_info}

Provide a concise analysis (200-300 words) covering:
1. What the figure shows (plot type, axes, coloring)
2. Key patterns or findings visible
3. Biological interpretation
4. Main takeaway for the research question"""},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
                ]}
            ]

            response = llm.invoke(messages)
            figure_analyses[fig_name] = response.content

        except Exception as e:
            figure_analyses[fig_name] = f"Could not analyze: {str(e)}"

    # =========================================================================
    # PASS 3: Analyze CSV data tables
    # =========================================================================
    print(f"📊 Pass 3: Analyzing {min(len(csv_files), 10)} CSV data tables...")

    csv_summaries = {}
    for csv_path in csv_files[:10]:  # Limit to 10 most important
        csv_name = os.path.basename(csv_path)
        print(f"   Reading {csv_name}...")

        try:
            df = pd.read_csv(csv_path)
            summary = f"**{csv_name}**\n"
            summary += f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"- Columns: {', '.join(df.columns[:10])}"
            if len(df.columns) > 10:
                summary += f" ... (+{len(df.columns)-10} more)"
            summary += "\n"

            # Add numeric summary for key columns
            numeric_cols = df.select_dtypes(include=['number']).columns[:5]
            if len(numeric_cols) > 0:
                summary += f"- Key statistics:\n"
                for col in numeric_cols:
                    summary += f"  - {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}\n"

            # Sample first few rows
            summary += f"- Sample data:\n```\n{df.head(3).to_string()}\n```\n"

            csv_summaries[csv_name] = summary

        except Exception as e:
            csv_summaries[csv_name] = f"Could not read: {str(e)}"

    # =========================================================================
    # PASS 4: Read JSON summaries
    # =========================================================================
    print(f"📄 Pass 4: Reading JSON summaries...")

    json_summaries = {}
    for json_file in json_files[:5]:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                content_str = json.dumps(content, indent=2)
                if len(content_str) > 3000:
                    content_str = content_str[:3000] + "\n... (truncated)"
                json_summaries[os.path.basename(json_file)] = content_str
        except:
            pass

    # =========================================================================
    # PASS 5: Format observation history
    # =========================================================================
    print(f"📝 Pass 5: Formatting observation history...")

    observation_narrative = []
    for obs in observations:
        step_text = f"**Step {obs.get('step', '?')}:**\n"
        step_text += f"Code: `{obs.get('code_snippet', 'N/A')[:200]}...`\n"
        result = obs.get('result_summary', '')
        if result:
            step_text += f"Result: {result[:500]}...\n" if len(result) > 500 else f"Result: {result}\n"
        fig_interp = obs.get('figure_interpretations', '')
        if fig_interp:
            step_text += f"Figure insights: {fig_interp[:300]}...\n" if len(fig_interp) > 300 else f"Figure insights: {fig_interp}\n"
        observation_narrative.append(step_text)

    # =========================================================================
    # PASS 6: Generate comprehensive report
    # =========================================================================
    print(f"📝 Pass 6: Generating comprehensive report...")

    # Build comprehensive context
    figure_section = "\n\n".join([f"### {name}\n{analysis}" for name, analysis in figure_analyses.items()])
    csv_section = "\n\n".join(csv_summaries.values())
    json_section = "\n\n".join([f"**{k}:**\n```json\n{v}\n```" for k, v in json_summaries.items()])
    obs_section = "\n".join(observation_narrative[:20])  # Limit to 20 steps

    report_prompt = f"""You are a senior computational biologist writing a comprehensive research report.

## RESEARCH QUESTION
{user_query}

## DATASET
{data_info}

## ANALYSIS HISTORY ({len(observations)} steps performed)
{obs_section}

## FIGURE ANALYSES (Vision LLM interpretations)
{figure_section[:8000]}

## DATA TABLE SUMMARIES
{csv_section[:3000]}

## JSON SUMMARIES
{json_section[:2000]}

---

Based on ALL the above information, write a comprehensive, publication-quality research report (5000-7000 words).

Structure your report as follows:

# Deep Research Report: [Create an informative title based on the research question]

## Executive Summary
(3-5 bullet points of the most important findings)

## 1. Introduction
### 1.1 Background
(Relevant biological context - what is known about this system/disease/tissue)
### 1.2 Research Question
(Clearly state the research objectives)
### 1.3 Significance
(Why this research matters)

## 2. Data Overview
### 2.1 Dataset Description
(Technology, species, tissue, sample sizes, conditions)
### 2.2 Data Quality
(Any quality considerations noted during analysis)

## 3. Methods Summary
### 3.1 Preprocessing
### 3.2 Analysis Pipeline
(List the key analytical approaches used)

## 4. Results
### 4.1 [First Major Finding Category]
(Describe findings, reference specific figures by filename)
### 4.2 [Second Major Finding Category]
(Continue for each major result area)
### 4.3 [Continue as needed]

## 5. Discussion
### 5.1 Key Insights
(What do these results tell us biologically?)
### 5.2 Comparison to Literature
(How do these findings relate to known biology?)
### 5.3 Mechanistic Interpretation
(What biological mechanisms might explain the observations?)

## 6. Conclusions
(Summarize the main conclusions that address the original research question)

## 7. Limitations
(What are the caveats and limitations of this analysis?)

## 8. Future Directions
(What follow-up experiments or analyses would be valuable?)

## 9. Methods Details
(Technical details for reproducibility)

---

Write the complete report now. Be specific, cite the actual figures generated, and provide biological interpretation."""

    messages = [
        {"role": "system", "content": "You are an expert computational biologist writing a comprehensive research report. Write in clear, scientific prose. Be specific and detailed."},
        {"role": "user", "content": report_prompt}
    ]

    response = llm.invoke(messages)
    report_content = response.content

    # === Add figure gallery at the end ===
    figure_gallery = "\n\n---\n\n## Appendix: Figure Gallery\n\n"
    for fig in figures[:30]:  # Limit to 30 figures
        fig_name = os.path.basename(fig)
        figure_gallery += f"### {fig_name}\n![{fig_name}]({fig})\n\n"

    full_report = report_content + figure_gallery

    # === Save report ===
    with open(output_path, 'w') as f:
        f.write(full_report)

    # Also save as HTML for better viewing
    html_path = None
    try:
        import markdown
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Deep Research Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #3498db; margin: 0; padding-left: 15px; color: #666; }}
        ul, ol {{ margin-left: 20px; }}
        .executive-summary {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
{markdown.markdown(full_report, extensions=['tables', 'fenced_code'])}
</body>
</html>"""
        html_path = f"{save_path}/deep_research_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
    except ImportError:
        pass

    print("✅ Deep Research Report generation complete!")

    result = f"""
✅ Successfully generated deep research report!

📊 **Subagent Analysis Summary:**
- Observations analyzed: {len(observations)} steps
- Figures interpreted: {len(figure_analyses)} images (vision LLM)
- CSV tables summarized: {len(csv_summaries)} files
- JSON summaries extracted: {len(json_summaries)} files

📄 **Output Files:**
- Markdown: {output_path}
"""
    if html_path:
        result += f"- HTML (styled): {html_path}\n"

    result += f"""
📈 **Report Contents:**
- Executive Summary with key findings
- Full methodology documentation
- Results with figure references
- Discussion and biological interpretation
- Limitations and future directions
- Figure gallery appendix ({len(figures)} figures)
"""

    return result


# =============================================================================
# Conclusion Verification Subagent
# =============================================================================

@tool
def verification_subagent(
    user_query: Annotated[str, Field(description="The research question being investigated (e.g., 'How does cellular composition change during disease?')")],
    conclusions: Annotated[str, Field(description="The conclusions to verify, as a string (e.g., '1) Immune cells increase 10-fold. 2) Epithelial cells decrease 50%.')")],
    data_info: Annotated[str, Field(description="Brief dataset description (e.g., 'MERFISH mouse colon, 50k cells, 4 disease stages')")],
    save_path: Annotated[str, Field(description="Directory path containing analysis outputs (figures, CSVs, observation_log.jsonl)")] = "./experiments",
) -> str:
    """Verify analysis conclusions against saved evidence (figures, data, observations).

    IMPORTANT: Call this tool with exactly 4 parameters as keyword arguments.

    Example usage:
        verification_subagent(
            user_query="How does the cellular microenvironment change during colitis?",
            conclusions="1) DSS9 shows 78% reduction in epithelial cells. 2) Immune cells increase 9-fold. 3) Spatial organization is disrupted.",
            data_info="MERFISH mouse colon, 50k cells, 4 disease stages",
            save_path="./experiments/my_analysis/"
        )

    What this tool does automatically:
    1. Reads observation_log.jsonl for analysis history
    2. Examines figures with vision LLM to verify visual claims
    3. Reads CSV data to verify quantitative claims
    4. Cross-checks each conclusion against actual evidence
    5. Generates a verification report with:
       - Score for each conclusion (supported/not supported)
       - Evidence citations
       - Suggested improvements
       - Recommended additional analyses

    Output: Saves report to {save_path}/conclusion_verification.md

    Note: Pass conclusions as a simple string, not a dictionary.

    Returns:
        str: Detailed verification report with scores and recommendations
    """
    from ..agent import make_llm
    import glob

    print("🔍 Starting Conclusion Verification Subagent...")

    model = _get_subagent_model()
    print(f"   Using model: {model}")
    llm = make_llm(model, stop_sequences=[])

    # =========================================================================
    # PASS 1: Inventory all artifacts
    # =========================================================================
    print("📋 Pass 1: Inventorying all artifacts...")

    # Read accumulated observations
    observation_log_path = f"{save_path}/observation_log.jsonl"
    observations = []
    if exists(observation_log_path):
        with open(observation_log_path, 'r') as f:
            for line in f:
                try:
                    observations.append(json.loads(line.strip()))
                except:
                    pass
    print(f"   Found {len(observations)} logged observations")

    # Discover all files
    figures = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg']:
        figures.extend(glob.glob(f"{save_path}/**/{ext}", recursive=True))
    figures = sorted(set(figures))
    print(f"   Found {len(figures)} figures")

    csv_files = sorted(glob.glob(f"{save_path}/**/*.csv", recursive=True))
    print(f"   Found {len(csv_files)} CSV files")

    # Extract analyses performed from observations
    analyses_performed = []
    for obs in observations:
        code = obs.get('code_snippet', '')
        if code:
            # Extract key analysis types from code
            if 'sc.pl.' in code or 'plt.' in code:
                analyses_performed.append(f"Visualization: {code[:100]}...")
            elif 'sc.tl.' in code:
                analyses_performed.append(f"Analysis: {code[:100]}...")
            elif any(x in code for x in ['groupby', 'value_counts', 'crosstab']):
                analyses_performed.append(f"Statistical summary: {code[:100]}...")

    # =========================================================================
    # PASS 2: Sample key figures to verify visual claims
    # =========================================================================
    print(f"🖼️ Pass 2: Sampling {min(len(figures), 5)} key figures for verification...")

    figure_verifications = {}
    # Select diverse figures (first, middle, last, and key ones)
    sample_indices = [0, len(figures)//3, len(figures)//2, 2*len(figures)//3, len(figures)-1]
    sample_indices = sorted(set(i for i in sample_indices if 0 <= i < len(figures)))[:5]

    for idx in sample_indices:
        fig_path = figures[idx]
        fig_name = os.path.basename(fig_path)
        print(f"   Verifying {fig_name}...")

        try:
            image_bytes = _resize_image_if_needed(fig_path)
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            ext = fig_path.lower().split('.')[-1]
            mime_type = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext, 'image/png')

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"""You are verifying scientific conclusions. Analyze this figure critically.

Figure: {fig_name}
Research question: {user_query[:300]}
Claimed conclusions: {conclusions[:500]}

Answer these questions:
1. What does this figure actually show?
2. Does the figure support, contradict, or is neutral to the claimed conclusions?
3. Are there any patterns in the figure NOT mentioned in the conclusions?
4. What is the quality of this visualization?

Be objective and critical. If conclusions are not supported, say so clearly."""},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
                ]}
            ]

            response = llm.invoke(messages)
            figure_verifications[fig_name] = response.content

        except Exception as e:
            figure_verifications[fig_name] = f"Could not verify: {str(e)}"

    # =========================================================================
    # PASS 3: Read CSV data to verify quantitative claims
    # =========================================================================
    print(f"📊 Pass 3: Analyzing {min(len(csv_files), 5)} CSV files for data verification...")

    data_evidence = {}
    for csv_path in csv_files[:5]:
        csv_name = os.path.basename(csv_path)
        print(f"   Reading {csv_name}...")

        try:
            df = pd.read_csv(csv_path)
            summary = f"**{csv_name}**\n"
            summary += f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"- Columns: {', '.join(df.columns[:8])}\n"

            # Extract key statistics
            numeric_cols = df.select_dtypes(include=['number']).columns[:3]
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    summary += f"- {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}\n"

            # Look for categorical distributions
            cat_cols = df.select_dtypes(include=['object']).columns[:2]
            for col in cat_cols:
                if df[col].nunique() < 20:
                    summary += f"- {col} distribution: {dict(df[col].value_counts().head(5))}\n"

            data_evidence[csv_name] = summary

        except Exception as e:
            data_evidence[csv_name] = f"Could not read: {str(e)}"

    # =========================================================================
    # PASS 4: Format observation history
    # =========================================================================
    print(f"📝 Pass 4: Extracting key observations...")

    observation_summary = []
    for obs in observations[-10:]:  # Last 10 observations
        step = obs.get('step', '?')
        code = obs.get('code_snippet', '')[:150]
        result = obs.get('result_summary', '')[:200]
        if code or result:
            observation_summary.append(f"Step {step}: {code}... → {result}")

    # =========================================================================
    # PASS 5: Generate comprehensive verification report
    # =========================================================================
    print(f"✅ Pass 5: Generating verification report...")

    # Build verification context
    figure_section = "\n\n".join([f"### {name}\n{analysis}" for name, analysis in figure_verifications.items()])
    data_section = "\n\n".join(data_evidence.values())
    obs_section = "\n".join(observation_summary)
    analyses_section = "\n".join(f"- {a}" for a in analyses_performed[:15])

    verification_prompt = f"""You are a senior computational biologist performing quality control on analysis conclusions.

## ORIGINAL RESEARCH QUESTION
{user_query}

## DATASET
{data_info}

## CLAIMED CONCLUSIONS
{conclusions}

## ANALYSES PERFORMED
{analyses_section if analyses_section else "Not specified"}

## FIGURE VERIFICATION (Vision LLM Analysis)
{figure_section[:4000]}

## DATA EVIDENCE FROM CSV FILES
{data_section[:3000]}

## OBSERVATION HISTORY (Last 10 steps)
{obs_section}

---

Based on the above evidence, provide a thorough verification report:

## 1. Query Alignment Assessment
**Score: X/10**
- How well do the conclusions address the original research question?
- What aspects are well-addressed?
- What aspects are missing or inadequately addressed?

## 2. Evidence Support Assessment
**Score: X/10**
- Are the conclusions supported by the figures and data?
- List specific claims and whether they are: ✅ Supported, ⚠️ Partially supported, ❌ Not supported, ❓ Cannot verify
- Are there findings in the data NOT mentioned in conclusions?

## 3. Completeness Evaluation
**Score: X/10**
- Were sufficient analyses performed to draw these conclusions?
- What additional analyses would strengthen the conclusions?

## 4. Scientific Rigor Check
- Are there alternative interpretations not considered?
- Are there potential confounders or limitations?
- Is the analytical approach appropriate?

## 5. Specific Issues Found
List any specific problems with the conclusions:
1. [Issue and why it's a problem]
2. [Issue and why it's a problem]
...

## 6. Recommended Improvements

### High Priority (significantly strengthen conclusions)
- [Specific improvement]: [Why needed]

### Medium Priority (add useful context)
- [Specific improvement]: [Why useful]

### Optional (nice to have)
- [Specific improvement]: [Benefit]

## 7. Suggested Additional Analyses
What analyses should be performed to better address the research question?
1. [Analysis]: [What it would reveal]
2. [Analysis]: [What it would reveal]

## 8. Overall Verdict
**Overall Score: X/10**

Summary: [2-3 sentence summary of verification results]

Recommendation: [ACCEPT / REVISE / MAJOR REVISION NEEDED]
"""

    messages = [
        {"role": "system", "content": "You are an expert reviewer performing rigorous quality control. Be objective, critical, and constructive. Do not hesitate to point out issues."},
        {"role": "user", "content": verification_prompt}
    ]

    response = llm.invoke(messages)
    verification_report = response.content

    # Save verification report
    output_path = f"{save_path}/conclusion_verification.md"
    with open(output_path, 'w') as f:
        f.write(f"# Conclusion Verification Report\n\n")
        f.write(f"**Generated by:** Verification Subagent\n")
        f.write(f"**Artifacts analyzed:** {len(observations)} observations, {len(figures)} figures, {len(csv_files)} CSV files\n\n")
        f.write("---\n\n")
        f.write(verification_report)

    print("✅ Conclusion Verification complete!")

    result = f"""
🔍 **Conclusion Verification Complete!**

📊 **Subagent Analysis Summary:**
- Observations reviewed: {len(observations)} steps
- Figures verified: {len(figure_verifications)} images (vision LLM)
- CSV data checked: {len(data_evidence)} files
- Analyses identified: {len(analyses_performed)}

📄 **Output:** {output_path}

---

{verification_report}
"""

    return result
