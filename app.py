import gradio as gr
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler
from custom_pipeline import CosStableDiffusionXLInstructPix2PixPipeline
from huggingface_hub import hf_hub_download
import numpy as np
import math
#import spaces 
import torch 
from PIL import Image
import gc

if torch.backends.mps.is_available():
    DEVICE = "mps"
    torch.mps.empty_cache()
    gc.collect()
elif torch.cuda.is_available():
    DEVICE = "cuda"
    torch.cuda.empty_cache()
    gc.collect()
else:
    DEVICE = "cpu"

print(f"DEVICE={DEVICE}")

#edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")
#normal_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl.safetensors")
edit_file = hf_hub_download(repo_id="cocktailpeanut/c", filename="cosxl_edit.safetensors")
normal_file = hf_hub_download(repo_id="cocktailpeanut/c", filename="cosxl.safetensors")

def set_timesteps_patched(self, num_inference_steps: int, device = None):
    self.num_inference_steps = num_inference_steps
    
    ramp = np.linspace(0, 1, self.num_inference_steps)
    sigmas = torch.linspace(math.log(self.config.sigma_min), math.log(self.config.sigma_max), len(ramp)).exp().flip(0)
    
    sigmas = (sigmas).to(dtype=torch.float32, device=device)
    self.timesteps = self.precondition_noise(sigmas)
    
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

EDMEulerScheduler.set_timesteps = set_timesteps_patched

pipe_edit = CosStableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8
)
pipe_edit.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
pipe_edit.to(DEVICE)

pipe_normal = StableDiffusionXLPipeline.from_single_file(normal_file, torch_dtype=torch.float16)
pipe_normal.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
pipe_normal.to(DEVICE)

#@spaces.GPU
def run_normal(prompt, negative_prompt="", guidance_scale=7, progress=gr.Progress(track_tqdm=True)):
    return pipe_normal(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=20).images[0]

#@spaces.GPU
def run_edit(image, prompt, resolution, negative_prompt="", guidance_scale=7, progress=gr.Progress(track_tqdm=True)):
    #resolution = 1024
    print(f"width={image.width}, height={image.height}")
    image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
    #image.resize((resolution, resolution))
    #return pipe_edit(prompt=prompt,image=image,height=resolution,width=resolution,negative_prompt=negative_prompt, guidance_scale=guidance_scale,num_inference_steps=20).images[0]
    print(f"width={image.width}, height={image.height}")
    img = pipe_edit(prompt=prompt,image=image,height=image.height,width=image.width,negative_prompt=negative_prompt, guidance_scale=guidance_scale,num_inference_steps=20).images[0]
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
        gc.collect()
    return img
css = '''
.gradio-container{
max-width: 768px !important;
margin: 0 auto;
}
'''
normal_examples = ["portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography", "backlit photography of a dog", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece"]
edit_examples = [["mountain.png", "make it a cloudy day"], ["painting.png", "make the earring fancier"]]
with gr.Blocks(css=css) as demo:
    gr.Markdown('''# CosXL demo
    Unofficial demo for CosXL, a SDXL model tuned to produce full color range images. CosXL Edit allows you to perform edits on images. Both have a [non-commercial community license](https://huggingface.co/stabilityai/cosxl/blob/main/LICENSE)
    ''')
    with gr.Tab("CosXL Edit"):
      with gr.Group():
          image_edit = gr.Image(label="Image you would like to edit", type="pil")
          prompt_edit = gr.Textbox(label="Prompt", scale=4, placeholder="Edit instructions, e.g.: Make the day cloudy")
          size_edit = gr.Number(label="Size", value=1024, maximum=1024, minimum=512, precision=0)
          button_edit = gr.Button("Generate", min_width=120)
          output_edit = gr.Image(label="Your result image", interactive=False)
          with gr.Accordion("Advanced Settings", open=False):
            negative_prompt_edit = gr.Textbox(label="Negative Prompt")
            guidance_scale_edit = gr.Number(label="Guidance Scale", value=7)
      gr.Examples(examples=edit_examples, fn=run_edit, inputs=[image_edit, prompt_edit, size_edit], outputs=[output_edit], cache_examples=False)
    with gr.Tab("CosXL"):
      with gr.Group():
          with gr.Row():
            prompt_normal = gr.Textbox(show_label=False, scale=4, placeholder="Your prompt, e.g.: backlit photography of a dog")
            button_normal = gr.Button("Generate", min_width=120)
          output_normal = gr.Image(label="Your result image", interactive=False)
          with gr.Accordion("Advanced Settings", open=False):
            negative_prompt_normal = gr.Textbox(label="Negative Prompt")
            guidance_scale_normal = gr.Number(label="Guidance Scale", value=7)
      gr.Examples(examples=normal_examples, fn=run_normal, inputs=[prompt_normal], outputs=[output_normal], cache_examples=False) 
    button_edit.click(
        
    )
    gr.on(
        triggers=[
            button_normal.click,
            prompt_normal.submit
        ],
        fn=run_normal,
        inputs=[prompt_normal, negative_prompt_normal, guidance_scale_normal],
        outputs=[output_normal],
    )
    gr.on(
        triggers=[
            button_edit.click,
            prompt_edit.submit
        ],
        fn=run_edit,
        inputs=[image_edit, prompt_edit, size_edit, negative_prompt_edit, guidance_scale_edit],
        outputs=[output_edit]
    )
if __name__ == "__main__":
    #demo.launch(share=True)
    demo.launch()
