from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

MODEL_PATH = "./models/dreambooth_model"

def inpaint_product(background, mask, prompt):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")

    background_img = Image.open(background).convert("RGB")
    mask_img = Image.open(mask).convert("L")

    result = pipe(prompt, image=background_img, mask_image=mask_img).images[0]
    return result

if __name__ == "__main__":
    inpainted_image = inpaint_product("./data/background.jpg", "./data/mask.png", "A photorealistic sks-brand soda can")
    inpainted_image.save("./solution/provisional/inpainted_image.jpg")
