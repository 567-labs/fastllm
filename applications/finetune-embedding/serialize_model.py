model = "/path/to/model"

from model import SimilarityModel
import numpy as np

model = SimilarityModel.load_from_checkpoint(model)
parameters = model.matrix.detach().numpy()

print("Sodel shape is:", parameters.shape)
print("Saving to `embedding_transform.npy`")
np.save("embedding_transform.npy", parameters)

assert np.load("embedding_transform.npy").shape == parameters.shape
