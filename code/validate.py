import torch
import clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def compute_clip_score(image_path, text_prompt):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    text = clip.tokenize([text_prompt]).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.cosine_similarity(image_features, text_features).item()
    
    return similarity

if __name__ == "__main__":
    score = compute_clip_score("./solution/provisional/inpainted_image.jpg", "A photorealistic sks-brand soda can")
    print("CLIP Similarity Score:", score)
