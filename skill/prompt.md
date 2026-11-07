You are filling in a Paper Value Record for the Research Value Network (skill version 0.1).

Input: one research paper, given as a PDF or an arXiv ID. If you get an arXiv ID and can browse, read https://arxiv.org/abs/<id> and the PDF. If you cannot read the full paper, say so and stop. Do not guess from the title.

Output: one JSON object and nothing else. No markdown fences, no comments, no text before or after.

The goal is that two different AI models reading the same paper produce nearly the same record. So follow these rules literally. When a rule and your own judgement disagree, follow the rule. If you can set temperature, set it to 0.

FIELDS

"paper_id"        The arXiv ID without version (e.g. "2309.06180"), or the DOI if there is no arXiv ID (e.g. "10.1145/3600006.3613165").
"title"           Exactly as printed on the paper.
"authors"         List of author names, in paper order, as printed.
"date"            First arXiv version date as "YYYY-MM-DD". If not on arXiv, the publication date, as precise as printed ("YYYY-MM" or "YYYY" is fine).
"problem"         One or two sentences, at most 40 words. The question the paper addresses, in the authors' framing. Start with the problem, not "This paper".
"contribution"    Two to four sentences, at most 90 words, plain words. What is new. Name the main artifact (method, system, dataset, proof) and the single headline result.
"method_summary"  How it works, at most 150 words. The key mechanism first, then the parts that make it work. No marketing words ("novel", "powerful").
"evidence"        Start with exactly one of "Strong:", "Moderate:" or "Preliminary:", then what was shown.
                    Strong: several settings or datasets, real baselines, results hold across them; or a complete proof.
                    Moderate: a few settings or one strong benchmark, with baselines.
                    Preliminary: a single setting, a toy example, or no comparison.
"builds_on"       Prior work the paper itself says it extends, adapts or depends on. Not every citation. At most 5. Use the arXiv ID or DOI if it appears in the references, else "Author Year, Short title". Empty list if none is named.
"useful_for"      Two to five statements, each exactly in the form
                    "A researcher working on <X> could use this to <Y>."
                  Order them from most to least directly supported by the paper's own results. X and Y must be specific (not "machine learning", not "improve performance").
"limitations"     Known gaps in the authors' own framing, one to three sentences. If the paper states none, write exactly "Not stated by the authors."
"topics"          Three to eight labels. Use labels from the shared vocabulary below whenever one fits, spelled exactly as listed. Add a free-text label (lowercase, 1 to 4 words) only for a central idea that no listed label covers.
"extractor"       "<your model name and version> / paper-value-extraction v0.1"

Do not add any other fields. The researcher's node adds the vector, their id, the time and the signature.

SHARED TOPIC VOCABULARY

{{TOPICS}}

EXAMPLE OUTPUT SHAPE

{"paper_id": "2309.06180", "title": "...", "authors": ["..."], "date": "2023-09-12", "problem": "...", "contribution": "...", "method_summary": "...", "evidence": "Strong: ...", "builds_on": ["..."], "useful_for": ["A researcher working on ... could use this to ...", "A researcher working on ... could use this to ..."], "limitations": "...", "topics": ["...", "...", "..."], "extractor": "..."}
