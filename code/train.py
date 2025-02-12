from diffusers import StableDiffusionPipeline, DreamBoothPipeline
import torch

MODEL_PATH = "./models/dreambooth_model"

def fine_tune_dreambooth(product_images, prompt):
    pipeline = DreamBoothPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipeline.to("cuda")

    pipeline.train_model(
        instance_data_dir=product_images,  # Folder with product images
        instance_prompt=prompt,
        output_dir=MODEL_PATH
    )
    print("Model training complete. Saved at:", MODEL_PATH)

if __name__ == "__main__":
    fine_tune_dreambooth("./data/product_images", "A photorealistic sks-brand soda can")
