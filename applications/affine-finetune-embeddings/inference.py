# test inference code using a checkpoint

from model import SimilarityModel
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import sys


def inference(
    t1: List[str], checkpoint_path: str, model_id: str = "BAAI/bge-small-en-v1.5"
) -> torch.Tensor:
    model = SimilarityModel.load_from_checkpoint(checkpoint_path)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_embedding_model = AutoModel.from_pretrained(model_id)

    encoded_input = tokenizer(t1, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = base_embedding_model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        embedding = model_output[0][:, 0].to(device)

        model.eval()
        finetune_embedding = model.encode(embedding)

    return finetune_embedding


if __name__ == "__main__":
    if len(sys.argv) != 2:
        checkpoint_num = 0
    else:
        checkpoint_num = sys.argv[1]
    checkpoint_dirpath = "checkpoints_stratified"
    res = inference(
        ["hi world"], f"{checkpoint_dirpath}/checkpoint-{checkpoint_num}.ckpt"
    )
    print(res)
    print("embedding shape", res.shape)
