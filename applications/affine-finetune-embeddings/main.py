from model import SimilarityModel
import torch

dropout_fraction = 0.5
n_dims = 256
lr = 1e-4
use_relu = True
model_id = "BAAI/bge-small-en-v1.5"

model = SimilarityModel(
    n_dims=n_dims,
    dropout_fraction=dropout_fraction,
    lr=lr,
    use_relu=use_relu,
    model_id=model_id,
)

e1 = torch.rand(512).unsqueeze(0)
e2 = torch.rand(512).unsqueeze(0)

t1 = "hello world"
t2 = "hello planet"
res = model(t1, t2)
print(res)
