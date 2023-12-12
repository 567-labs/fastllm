import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import util
from transformers import AutoTokenizer, AutoModel


# Similarity Model
class SimilarityModel(pl.LightningModule):
    def __init__(
        self,
        embedding_size,
        n_dims,
        dropout_fraction,
        lr,
        use_relu,
        base_embedding_model="BAAI/bge-small-en-v1.5",
        tokenizer="BAAI/bge-small-en-v1.5",
    ):
        super(SimilarityModel, self).__init__()
        self.matrix = torch.nn.Parameter(
            torch.rand(
                embedding_size,
                n_dims,
            )
        )
        torch.nn.init.xavier_uniform_(self.matrix)
        self.dropout_fraction = dropout_fraction
        self.lr = lr
        self.use_relu = use_relu

        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(num_classes=2, task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.scale = 20  # MNRL loss scale: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        self.save_hyperparameters()

        self.base_embedding_model = AutoModel.from_pretrained(base_embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def forward(self, text):
        # TODO: modify to take in List[str] and return modified embedding
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.base_embedding_model(**encoded_input)
            # Perform cls pooling and normalization, according to docs https://huggingface.co/BAAI/bge-small-en-v1.5
            embedding = model_output[0][:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        e = F.dropout(embedding, p=self.dropout_fraction)
        matrix = self.matrix if not self.use_relu else F.relu(self.matrix)
        modified_embedding = e @ matrix
        return modified_embedding

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        text_a, text_b = batch
        modified_embedding_a = self(text_a)
        modified_embedding_b = self(text_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        text_a, text_b = batch
        modified_embedding_a = self(text_a)
        modified_embedding_b = self(text_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)

        # TODO: for some reason i need to explicilty set batch size, investigate this
        self.log("val_loss", loss, batch_size=len(text_a))

        # TODO: fix f1 scores, probably just convert array to be diagonal and input that
        threshold = 0.5
        pred_labels = scores > threshold
        # self.log("val_f1", self.f1(pred_labels.int(), labels.int()))

        # TODO: for some reason i need to explicilty set batch size, investigate this
        self.log("val_f1", 0.5, batch_size=len(text_a))

    def test_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py

        text_a, text_b = batch
        modified_embedding_a = self(text_a)
        modified_embedding_b = self(text_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)
        self.log("test_loss", loss, batch_size=len(text_a))

        # treshold to decide if matching ??
        # TODO: fix f1 scores, probably just convert array to be diagonal and input that
        threshold = 0.5
        pred_labels = scores > threshold
        # self.log("val_f1", self.f1(pred_labels.int(), labels.int()))
        self.log("test_f1", 0.5, batch_size=len(text_a))
