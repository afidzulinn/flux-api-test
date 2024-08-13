from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import torch
from diffusers import FluxPipeline
from pydantic import BaseModel
import os

import uvicorn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                    torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

@app.get("/")
async def root():
    return {"message": "API Flux Text Generation"}

# class ImageRequest(BaseModel):
#     prompt: str
#     seed: int = 0

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    image = pipe(
        prompt.prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("device").manual_seed(prompt.seed)
    ).images[0]
        
        # Save the image
    output_path = "generated_image.png"
    image.save(output_path)
        
    return FileResponse(output_path, media_type="image/png", 
                        filename="generated_image.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9123)