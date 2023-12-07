# test inference code using a checkpoint

from model import SimilarityModel
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List


def inference(t1: List[str], t2: List[str], checkpoint_path=str):
    model_id = "BAAI/bge-small-en-v1.5"

    # model = SimilarityModel.load_from_checkpoint(
    #     "checkpoints_stratified/similarity-model-220-128-epoch=02-val_loss=3.24-val_recall=1.00-val_f1=1.00.ckpt"
    # )
    model = SimilarityModel.load_from_checkpoint(
        checkpoint_path
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_embedding_model = AutoModel.from_pretrained(model_id)
    encoded_input_1 = tokenizer(t1, padding=True, truncation=True, return_tensors="pt")
    encoded_input_2 = tokenizer(t2, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output_1 = base_embedding_model(**encoded_input_1)
        model_output_2 = base_embedding_model(**encoded_input_2)
        # Perform pooling. In this case, cls pooling.
        embedding_1 = model_output_1[0][:, 0]
        embedding_2 = model_output_2[0][:, 0]

    embedding_1 = embedding_1.to(device)
    embedding_2 = embedding_2.to(device)

    model.eval()
    with torch.no_grad():
        res = model(embedding_1, embedding_2)

    return res


if __name__ == "__main__":
    res = inference(["hi world"], ["hello cats"])
