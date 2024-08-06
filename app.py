import gradio as gr
import torch
from diffusers import DiffusionPipeline
import devicetorch
device = devicetorch.get(torch)
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to(device)
def generate_image(prompt):
    return pipe(
      prompt,
      num_inference_steps=2,
      strength=0.5,
      guidance_scale=0.0
    ).images[0]
app = gr.Interface(fn=generate_image, inputs="text", outputs="image")
app.launch()
