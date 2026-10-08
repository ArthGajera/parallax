import tempfile, os
import gradio as gr
from core import generate_image, estimate_depth, render_parallax, depth_to_image

MOTIONS = ["Orbit", "Pan (left-right)", "Dolly zoom"]


def run(prompt, seed, steps, motion, strength, seconds):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    img, gen_time = generate_image(prompt, seed, steps)
    depth = estimate_depth(img)
    out = os.path.join(tempfile.mkdtemp(), "parallax.mp4")
    render_parallax(img, depth, motion, strength, seconds, out_path=out)
    return img, depth_to_image(depth), out, f"Image generated in {gen_time:.2f}s ({int(steps)} step(s))"


with gr.Blocks(title="Parallax Studio") as demo:
    gr.Markdown("# Parallax Studio\nText → image (SDXL-Turbo) → depth (Depth Anything) → 2.5D camera-move video")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="a cozy cabin in a misty pine forest at sunrise, cinematic")
            seed = gr.Number(label="Seed", value=42, precision=0)
            steps = gr.Slider(1, 4, value=2, step=1, label="Diffusion steps")
            motion = gr.Dropdown(MOTIONS, value="Orbit", label="Camera motion")
            strength = gr.Slider(5, 60, value=28, step=1, label="Parallax strength (px)")
            seconds = gr.Slider(2, 6, value=3, step=1, label="Duration (s)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Generated image")
            depth_out = gr.Image(label="Estimated depth (white = near)")
            vid_out = gr.Video(label="Parallax video")
            info = gr.Markdown()
    btn.click(run, [prompt, seed, steps, motion, strength, seconds], [img_out, depth_out, vid_out, info])

if __name__ == "__main__":
    demo.launch()
