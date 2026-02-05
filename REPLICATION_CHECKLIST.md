1. Create env: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Fetch data: `python scripts/fetch_data.py`
4. Run pipeline: `python scripts/run_pipeline.py`
5. Build PDF: `pdflatex paper/main.tex`
6. Check results in `results/` and figures in `figures/`
7. Review `RESULTS_MANIFEST.md`
