import torch
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.nn import functional as f
import clip
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import (
    PIL_INTERPOLATION,
)
from torchvision.utils import make_grid, save_image
import torchvision
import os

dirname = "./to_classify/"

# CLIP classification util
def clip_classify(images, prompts, model):
    
    text_embeddings = model.encode_text(prompts)
    images_embeddings = images

    dots = (images_embeddings @ text_embeddings.T)
    return f.softmax(dots, dim=1)

# preprocessing util
def preprocess1(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        print(w, h)

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


# load stable diffusion pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16, safety_checker=None
).to(device)
# uncomment this if you have troubles with available memory
# pipe.enable_attention_slicing()
# load CLIP mopdel
model, preprocess = clip.load("ViT-L/14", device=device)


# add path to your image
for filename in os.listdir(dirname):
    init_image = Image.open(dirname + filename).resize((448,448), PIL.Image.Resampling.LANCZOS)
    init_image_tensor = preprocess1(init_image)
    break

classes = [
]



N = int(input("Enter the number of classes: "))
strength = float(input("Enter strength param (between 0.4 and 0.99. More strength, more creativity): "))
print(f"Enter {N} prompts (classes): ")
for _ in range(N):
    classes.append(input())

prompts = [
    f'A photo of a {representative}' for representative in classes
]

images = pipe(prompt=prompts, image=init_image, strength=strength, guidance_scale=7.5, num_inference_steps=50, num_images_per_prompt=1).images
images_tensor_list = [torchvision.transforms.ToTensor()(image) for image in images ]

images_tensor_list_stacked = torch.stack(images_tensor_list, dim=0)
results = torch.linalg.norm(images_tensor_list_stacked.flatten(start_dim=1, end_dim=-1)-init_image_tensor.flatten(start_dim=1, end_dim=-1), dim=1)

init_clip = model.encode_image(preprocess(init_image).unsqueeze(0).to(device))
clip_results = clip_classify(init_clip, prompts=clip.tokenize(prompts).to(device), model=model)

recons_clip=model.encode_image( torch.stack([preprocess(image) for image in images], dim=0).to(device))
results = torch.linalg.norm(recons_clip-init_clip, dim=1)
print(f'clip decision: {classes[torch.argmax(clip_results).item()]}')
print(f'diffclip decision: {classes[torch.argmin(results).item()]}')

Grid = make_grid(images_tensor_list, nrow=len(classes))
save_image(Grid, "./sides/sides.jpg")
