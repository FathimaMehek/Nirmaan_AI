# Nirmaan AI – Self Introduction Scoring Tool

## Project Overview
Built an AI-powered tool that evaluates self-introduction transcripts using NLP, semantic similarity and custom scoring rules. The system helps learners improve communication quality by generating structured feedback across grammar, flow, sentiment, vocabulary richness, filler-word usage and more.

## Live App:
**[Launch Nirmaan AI 🚀](https://fathimamehek-nirmaan-ai.streamlit.app/)**

## The engine analyzes transcripts using:
- Sentence embeddings (semantic understanding)
- Rule-based scoring (rubric-driven evaluation)
- Grammar heuristics
- Sentiment analysis
- Speech rate analysis
- Keyword presence detection
- Vocabulary richness (TTR)

A clean, user-friendly interface was built using Streamlit where users can paste text or upload a file and instantly receive a complete communication score report.

## Code and Resources Used
- Python Version: 3.12.7
- Packages: sentence-transformers, numpy, pandas, scikit-learn, nltk, streamlit, json, re, pathlib
- Web App: Streamlit
- Rubric Source: Extracted from XLSX → converted into JSON (rubric.json)

## Rubric Extraction (XLSX → JSON)
The scoring rubric was originally provided in an Excel (.xlsx) sheet.
To integrate it into the NLP engine:
- The structured rubric was extracted programmatically from the XLSX file.
- All rubric fields (criteria, metric weights, keyword lists, scoring rules) were cleaned + standardized.
- The final cleaned structure was exported into a JSON file (rubric.json).
- This JSON file is now read directly by the scoring engine to ensure consistency and flexibility.

This allows:
- Easy updates to scoring rules
- JSON-driven evaluation
- A clean separation between data and logic

## NLP & Scoring Engine – How It Works
The scoring engine (scoring_engine.py) performs multiple layers of analysis:

1. Tokenization & Preprocessing
- Extracts words
- Counts total tokens
- Computes vocabulary richness (TTR)
- Calculates filler-word frequency

2. Semantic Embedding
- Uses all-MiniLM-L6-v2 from Sentence Transformers
- Computes similarity between transcript and metric descriptions
- Smooths raw rule-based scores

3. Rule-Based Scoring (from rubric.json)

    Each metric in the rubric contains:
- Weight
- Numeric ranges
- Keyword lists
- Rule thresholds
- Expected order (flow)
- Filler word dictionary
- Sentiment thresholds
  
   Example metrics include:
- Salutation
- Keyword Presence
- Flow
- Speech Rate
- Grammar Quality
- Vocabulary Richness
- Filler Word Rate
- Positivity Score

4. Grammar Analysis
   
    Lightweight grammar heuristics detect:
- Double spaces
- Missing capital letters
- Very short/fragmented sentences
- Common misspellings
- Maps to rubric quality bands.

5. Sentiment Analysis
   
    Uses NLTK VADER to compute a positivity score.

7. Final Score Calculation
For each metric:
- Rule score
- Semantic similarity score
- Weighted blended final score
  
    Outputs structured results:
- Criterion-level scores
- Metric-level scores
- Overall score

## Web Application (Streamlit)
The UI (app.py) provides:
- Text input
- File upload
- Speaking duration input
- Single-click scoring
- Beautiful summary dashboard
- Per-metric breakdown with semantic match
- Motivational footer for learners
UI Highlights:
- Gradient overall score card
- Boxes for each criterion
- Detailed metric cards (rule score, final score, semantic similarity, feedback)

## Model & Techniques Used
- Sentence Transformers (MiniLM-L6-v2)
- Cosine Similarity
- VADER Sentiment Analyzer
- Rule-based evaluation from JSON
- Grammar heuristics
- Vocabulary TTR metric
- Speech rate estimator

## Example Output
The tool outputs JSON containing:
- Overall Score (0–100)
- Total Words
- Per-criterion scores
  
    Per-metric details:
- Final Score
- Rule Score
- Semantic Similarity
- Feedback

## Deployment
The application is deployed via Streamlit Cloud

## Future Improvements
- Add voice input (speech-to-text)
- Advanced grammar model (Transformer based)
- Multi-language scoring
- Leaderboard system
- Teacher dashboards for batch evaluation
