import gradio as gr
import numpy as np
import random
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel, freeze, qfloat8, quantize
from datetime import datetime
from PIL import Image
import json
import devicetorch
import os
import argparse
import time

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel
dtype = torch.bfloat16
#dtype = torch.float32
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./outputs", type=str, help="The path to the output folder")
    parser.add_argument("--port", default=1024, type=int, help="The port to run the Gradio App on.")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="The host to run the Gradio App on.")
    parser.add_argument("--share", action="store_true", help="Whether to share this gradio demo.")
    parser.add_argument("--save_prompt", action="store_true", help="Whether to save the output prompt and other configurations.")
    # parser.add_argument("--model_path", default="./models", type=str, help="The path to the models folder")
    # parser.add_argument(
    #     "--low_gpu_memory_mode",
    #     action="store_true",
    #     help="Whether to enable low GPU memory mode",
    # )
    return parser.parse_args()

css="""
nav {
  text-align: center;
}
.shield {
    margin-right:5px;
}
.shields {
    align:center;
    margin:auto;
    display:flex;
}
"""

# #logo{
#   width: 50px;
#   display: inline;
#   align: center;
# }

args=parse_args()

#save all generated images into an output folder with unique name
def save_images(images,timestamp):  
    output_folder = args.output
    onetime_output_folder = os.path.join(output_folder,timestamp)
    os.makedirs(onetime_output_folder, exist_ok=True)
    saved_paths = []
    
    for i, img in enumerate(images):
        filename = f"output_{i}.png"
        filepath = os.path.join(onetime_output_folder, filename)
        img.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths

def save_images_with_prompt(
    prompt=" ",
    checkpoint="",
    seed=42,
    guidance_scale=0.0,
    width=1024, height=1024,
    num_inference_steps=4,
    max_memory_usage=0.0,
    generation_time=0.0,
    images=None,
    timestamp=None,
    # filename='diffusion_params.json'
):
    """
    Saves the parameters used in generating an image with a diffuser model to a JSON file.

    Args:
        prompt (str): The text prompt used for generation.
        checkpoint (str): The checkpoint name or path of the model.
        seed (int): The random seed for reproducibility.
        guidance_scale (float): The scale for classifier-free guidance.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        num_inference_steps (int): The number of inference steps.
        max_memory_usage (float): Maximum GPU memory usage during generation (in MB).
        generation_time (float): Time taken to generate the image (in seconds).
        filename (str): The name of the output JSON file.

    Returns:
        None
    """
    
    params = {
        'prompt': prompt,
        'checkpoint': checkpoint,
        'seed': seed,
        'guidance_scale': guidance_scale,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'max_memory_usage': f"{max_memory_usage} MB",
        'generation_time': f"{generation_time} s",
    }
    
    output_folder = args.output
    onetime_output_folder=os.path.join(output_folder,str(timestamp))
    os.makedirs(onetime_output_folder, exist_ok=True)
    
    filename = f"prompt_{timestamp}.json"
    filepath=os.path.join(onetime_output_folder,filename)

    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    return save_images(images,timestamp)

def infer_casdao(
    prompt, 
    checkpoint="../fshare/models/sayakpaul/FLUX.1-merged", 
    seed=42, 
    guidance_scale=0.0, 
    num_images_per_prompt=1, 
    randomize_seed=False, 
    width=1024, height=1024, 
    num_inference_steps=4, 
    whether_save_prompt=False,
    progress=gr.Progress(track_tqdm=True),
    ):
    
    global pipe
    global selected
    # if the new checkpoint is different from the selected one, re-instantiate the pipe
    
    torch.cuda.reset_peak_memory_stats()
    
    if selected!=checkpoint:
        torch.cuda.empty_cache()
            
        if checkpoint == "../fshare/models/sayakpaul/FLUX.1-merged":
            bfl_repo = "../fshare/models/cocktailpeanut/xulf-d"
            # print("initializing quantized transformer...")
            print("初始化量化的 transformer 中...")
            transformer = QuantizedFluxTransformer2DModel.from_pretrained("../fshare/models/cocktailpeanut/flux1-merged-q8")
            # print("initialized!")
            print("初始化完成！")
            
            # print(f"moving device to {device}")
            print(f"将 Transformer 模型移到 {device} 设备中。")
            transformer.to(device=device, dtype=dtype)
        
            # print(f"initializing pipeline...")
            print(f"初始化 Flux 管线中...")
            pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=dtype)
            # print("initialized!")
            print("初始化完成")
            pipe.transformer = transformer
        
            if device == "cuda":
                # print(f"enable model cpu offload...")
                print(f"启用CPU加载模型降低负载...")
                #pipe.enable_model_cpu_offload()
                pipe.enable_sequential_cpu_offload()
                # print(f"done!")
                print(f"启用完成！")
            
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()  
            pipe.to(torch.bfloat16)
        
        else:
            if checkpoint == "../fshare/models/black-forest-labs/FLUX.1-schnell":
                bfl_repo = checkpoint
                # print("initializing quantized transformer...")
                print("初始化量化的 transformer 中...")
                transformer = FluxTransformer2DModel.from_pretrained("../fshare/models/cocktailpeanut/xulf-s/transformer",torch_dtype=dtype)    
                quantize(transformer, weights=qfloat8)
                freeze(transformer)
                # print("initialized!")
                print("初始化完成！")
            elif checkpoint == "../fshare/models/black-forest-labs/FLUX.1-dev":
                bfl_repo = checkpoint
                # print("initializing quantized transformer...")
                print("初始化量化的 transformer 中...")
                transformer = FluxTransformer2DModel.from_pretrained("../fshare/models/cocktailpeanut/xulf-d/transformer",torch_dtype=dtype)    
                quantize(transformer, weights=qfloat8)
                freeze(transformer)
                # print("initialized!")
                print("初始化完成！")
            
            text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)
            
            pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
            
            pipe.enable_model_cpu_offload()
        
        selected = checkpoint
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
        
    # print(f"Started the inference. Wait...")
    print(f"已开始推理。请稍等...")
    # 记录推理前的时间
    start_time = time.time()
    images = pipe(
            prompt = prompt,
            width = width,
            height = height,
            num_inference_steps = num_inference_steps,
            generator = generator,
            num_images_per_prompt = num_images_per_prompt,
            guidance_scale=guidance_scale
    ).images
        
    # 计算用时和峰值显存占用
    elapsed_time = time.time() - start_time
    max_vram_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    timestamp_after_generation = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
    # print(f"Inference finished!")
    print(f"推理已完成！")
    print("生成所耗费的时间：", elapsed_time, "s")
    print("峰值显存占用：", max_vram_usage, "MB")
        
    devicetorch.empty_cache(torch)
    # print(f"emptied cache.")
    print(f"已清理缓存。")
        
    saved_paths=None
    if whether_save_prompt:
        saved_paths=save_images_with_prompt(
            prompt=prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            checkpoint=selected,
            width = width, height = height,
            num_inference_steps = num_inference_steps,
            max_memory_usage=max_vram_usage,
            generation_time=elapsed_time,
            images=images,
            timestamp=timestamp_after_generation
        )
    else:
        saved_paths=save_images(images,timestamp_after_generation)
        
    return images, seed, saved_paths
    
def update_slider(checkpoint, num_inference_steps):
    if checkpoint == "../fshare/models/sayakpaul/FLUX.1-merged":
        return 8
    else:
        return 4
    
with gr.Blocks(css=css, title="Flux-WebUI") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(
            """
            <div align="shields">
                <!--img src="./icon.png"/-->
                <h1>Flux-WebUI：体验Flux的简易WebUI</h1>
            </div>
            <center>
                <div class="shields">
                    <div class="shield">
                        <a href="https://ai.casdao.com/">
                            <img src="https://img.shields.io/badge/Casdao-%E6%99%BA%E7%AE%97%E7%A9%BA%E9%97%B4-blue" alt="算力互联-智算空间" height="50">
                        </a>
                    </div>
                    <div class="shield">
                        <a href="https://github.com/pinokiofactory/flux-webui">
                            <img src="https://img.shields.io/github/stars/pinokiofactory/flux-webui?style=social" alt="GitHub标星" height="50">
                        </a>
                    </div>
                    <div class="shield">
                        <a href="https://blackforestlabs.ai/">
                            <img alt="Black Forest Labs" src="https://img.shields.io/badge/Black_Forest_Labs-gray" height="50">
                        </a>
                    </div>
                </div>
            </center>
            """
        )
        with gr.Row():
            checkpoint = gr.Dropdown(
                label="模型（Model）",
                value= "../fshare/models/black-forest-labs/FLUX.1-dev",
                choices=[
                    "../fshare/models/black-forest-labs/FLUX.1-dev"
                    "../fshare/models/black-forest-labs/FLUX.1-schnell",
                    "../fshare/models/sayakpaul/FLUX.1-merged",
                ],
                scale=8
            )
            run_button = gr.Button("运行（Run）", scale=2)
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    prompt = gr.Text(
                        label="提示词（Prompt）",
                        lines=5,
                        max_lines=10,
                        placeholder="使用英文输入提示词（Enter the prompt）",
                        container=True,
                    )
                    store_prompt = gr.Checkbox(label="保存生成设置（Save generation configs）", value=True, scale=1)
                with gr.Column():
                    seed = gr.Slider(
                        label="种子（Seed）",
                        info="当固定种子后，生成的结果更容易复现。",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="随机种子（Randomize Seed）", value=True)
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="推理步数（Number of inference steps）",
                        info="对于flux.1-schnell，4步即可生成效果很好的图片。",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=4,
                    )
                    guidance_scale = gr.Number(
                        label="引导系数（Guidance Scale）",
                        info="对于schnell模型，只需要设置为0就可以生成很好的模型。",
                        minimum=0,
                        maximum=50,
                        value=0.0,
                    )
                with gr.Row():
                    width = gr.Slider(
                        label="宽度（Width）",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="高度（Height）",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
            result = gr.Gallery(label="生成结果（Result）", show_label=False, object_fit="contain", format="png",height=500)
        with gr.Row():
            num_images_per_prompt = gr.Slider(
                label="生成图片数量（Number of images）",
                minimum=1,
                maximum=50,
                step=1,
                value=1,
            )
        checkpoint.change(fn=update_slider, inputs=[checkpoint, num_inference_steps], outputs=[num_inference_steps])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer_casdao,
        inputs = [prompt, checkpoint, seed, guidance_scale, num_images_per_prompt, randomize_seed, width, height, num_inference_steps,store_prompt],
        outputs = [result, seed]
    )
demo.launch(
    share=args.share,
    server_name=args.host,
    server_port=args.port,
)