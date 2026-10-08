# Parallax

Text prompt → image (SDXL-Turbo) → monocular depth (Depth Anything) → 2.5D parallax video, in a Gradio app.

## Pipeline
prompt → SDXL-Turbo (1-4 steps) → Depth Anything → depth-weighted pixel remap per frame → MP4

## Run
    pip install -r requirements.txt
    python app.py            # demo UI
    python eval_clip.py      # CLIP-score vs. steps vs. latency; writes eval_results.csv + eval_chart.png

## Notes / limitations
- Parallax uses backward warping, so large strengths stretch edges; keep strength moderate.
- Depth is relative (not metric) and comes from a single image.
- Eval: CLIP similarity measures prompt alignment, not aesthetic quality.
