import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# load sample image
path_image = "/home/randellini/LAVIS_image_text/images/sicurezza_lavoro.jpg"
raw_image = Image.open(path_image).convert("RGB")


# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 1) ask the model to generate the response
model.generate({"image": image, "prompt": "Is something dangerous happening?"})

# 2) generate a short description.
model.generate({"image": image, "prompt": "Write a short description for the image."})

# 3) generate a detailed description
model.generate({"image": image, "prompt": "Write a detailed description."})