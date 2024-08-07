import gradio as gr
import numpy as np
import random
import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
import devicetorch
dtype = torch.bfloat16
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None
css="""
nav {
  text-align: center;
}
#logo{
  width: 50px;
  display: inline;
}
"""
def infer(prompt, checkpoint="black-forest-labs/FLUX.1-schnell", seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    global pipe
    global selected
    # if the new checkpoint is different from the selected one, re-instantiate the pipe
    if selected != checkpoint:
        if checkpoint == "sayakpaul/FLUX.1-merged":
            transformer = FluxTransformer2DModel.from_pretrained("sayakpaul/FLUX.1-merged", torch_dtype=dtype)
            pipe = FluxPipeline.from_pretrained("cocktailpeanut/xulf-d", transformer=transformer, torch_dtype=dtype)
        else:
            #transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8.safetensors")
            #pipe = FluxPipeline.from_pretrained(checkpoint, transformer=transformer, torch_dtype=torch.bfloat16)
            pipe = FluxPipeline.from_pretrained(checkpoint, torch_dtype=dtype)

        pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        if device == "cuda":
            #pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
          
        selected = checkpoint
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
            prompt = prompt, 
            width = width,
            height = height,
            num_inference_steps = num_inference_steps, 
            generator = generator,
            guidance_scale=0.0
    ).images[0] 
    devicetorch.empty_cache(torch)
    return image, seed
def update_slider(checkpoint, num_inference_steps):
    if checkpoint == "sayakpaul/FLUX.1-merged":
        return 8
    else:
        return 4
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("<nav><img id='logo' src='file/icon.png'/></nav>")
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
        with gr.Accordion("Advanced Settings"):
            checkpoint = gr.Dropdown(
              value= "black-forest-labs/FLUX.1-schnell",
              choices=[
                "black-forest-labs/FLUX.1-schnell",
                "sayakpaul/FLUX.1-merged"
              ]
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=576,
                )
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )
            checkpoint.change(fn=update_slider, inputs=[checkpoint], outputs=[num_inference_steps])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, checkpoint, seed, randomize_seed, width, height, num_inference_steps],
        outputs = [result, seed]
    )
demo.launch()
