# AI Bubble Risk Quant Paper

Paper: **“Probability of an AI Bubble Burst: Is there a bubble today, how will it end, and when?”**

## Project Board (Local Kanban)
See `PROJECT_BOARD.md` (BACKLOG → PLANNED → IN PROGRESS → REVIEW → DONE). All tasks are marked DONE.

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/run_pipeline.py
pdflatex paper/main.tex
```

## Outputs
- `paper/main.tex` + `paper/main.pdf`
- `figures/`
- `results/`
- `EXECUTIVE_SUMMARY.txt`
- `REPLICATION_CHECKLIST.md`
