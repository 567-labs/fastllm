# sample inference code for how to use the model with a checkpoint

from model import SimilarityModel
import torch

# make sure you run main.py first to create checkpoint
model = SimilarityModel.load_from_checkpoint("checkpoints_stratified/checkpoint-0.ckpt")
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
model.to(device)
text = "hello world"
with torch.no_grad():
    finetune_embedding = model(["hello world", "i like taco"])
    print(finetune_embedding)
