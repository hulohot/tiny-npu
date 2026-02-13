# Memory - Ethan Brugger

---

# TODO: Dissertation Feedback Tool

**Status**: Not started  
**Priority**: High (for dissertation work)  
**Added**: 2026-02-12

## Goal
Build a tool that provides instant feedback on dissertation structure and content using multiple LLM evaluations.

## Features Needed
1. **Input**: Accept Word documents (.docx)
2. **Multi-LLM Evaluation**: Different models evaluate different criteria:
   - **Soundness/Logic**: Check argument structure, reasoning flow
   - **Grammar/Style**: Writing quality, clarity, tone
   - **Citations**: Verify citation format, check for missing refs
   - **Technical Accuracy**: Verify claims against sources
   - **Structure**: Chapter organization, transitions, coherence
3. **Output**: Structured feedback report with specific suggestions

## Technical Approach
- Python script using `python-docx` to read Word files
- Call multiple LLM APIs with specific prompts per criteria
- Aggregate results into markdown/PDF report

## Commands (planned)
```bash
# Analyze dissertation chapter
python3 dissertation_review.py review --chapter chapter1.docx --criteria all

# Check specific aspect  
python3 dissertation_review.py review --chapter chapter2.docx --criteria citations
```
