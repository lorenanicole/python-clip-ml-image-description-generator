from PIL import Image
from io import BytesIO
from fastapi import File, FastAPI, UploadFile
import torch
from torchvision import transforms as T
import clip
import hashlib

from cache import Cache, my_lru_cache

app = FastAPI()
cache = Cache(max_size=128)

# Indicates if using Nvidia graphics card or not - cuda = yes otherwise cpu
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
model = "ViT-B/32"
model, preprocess = clip.load(model, device=device, jit=False)

@app.get("/test")
async def root(name: str):
    if name:
        return {"message": f"Hello {name}"}
    return {"message": f"Hello World"}

@app.post("/description")
async def generate_description(image: UploadFile = File(...)):
    image = await image.read()
    bytes_img = BytesIO(image)
    image_obj = Image.open(bytes_img)

    readable_hash = hashlib.sha256(image).hexdigest()
    text_labels_in_cache = cache.get(readable_hash)
    if text_labels_in_cache and text_labels_in_cache != -1:
        return text_labels_in_cache

    image_input = preprocess(image_obj).unsqueeze(0).to(device)
    # Can replace with https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html
    text_snippets = ["dog", "cat", "tiger", "man", "woman", "child", "grandma"] 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_snippets])

    # Calculate features for our "zero-shot learning." CLIP is a model not trained
    # on specific tasks but is when a model attempts to predict a class it saw 
    # zero times in the training data.
    # 
    # We explictly tell torch no to keep track of the gradients when updating weights
    # as this will impact back propagation and then we calcualte similarity scorings.
    with torch.no_grad():
        image_features = model.encode_image(image_input) 
        text_features = model.encode_text(text_inputs)  
    
    # Get the top 5 most similary labels!
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)# .cpu().numpy() 
    values, indices = similarity[0].topk(5)

    # Marry the value and indices back to the text label
    text_labels = {}
    for indx, val in enumerate(indices):
        text_labels[text_snippets[val]] = values[indx].item()
    
    cache.put(readable_hash, text_labels)
    return text_labels

@app.get('/cache-stats')
def get_cache_stats():
    return cache