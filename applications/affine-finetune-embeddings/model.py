import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import util
from transformers import AutoTokenizer, AutoModel

# class Embedding: # use this for huggingface compatibiltiy?
#      self.encoder = ...
#      self.adapter = ...
#     def forward(self, x)
#         x = self.encoder(x)
#         x = self.adapter(x)
#         return x (edited)
# 11:14
# class BiEncoder(SentenceTransformer): # this one is for training
#     self.encoder = Embedding()
#    def forward(self, x, y):
#         x = self.encoder(x)
#         y = self.encoder(y)
#         return cosine(x, y) (edited)


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

        # hyperparameters
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(num_classes=2, task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.scale = 20  # MNRL loss scale: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        self.save_hyperparameters()

        # base embedding model and tokenizer
        self.base_embedding_model = AutoModel.from_pretrained(base_embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # not sure if i need this
        # self.base_embedding_model.to(self.device)

    def forward(self, text):
        # TODO: modify to take in List[str] and return modified embedding
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.base_embedding_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            # embedding = model_output[0][:, 0].to(self.device)
            embedding = model_output[0][:, 0]

            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        e = F.dropout(embedding, p=self.dropout_fraction)
        matrix = self.matrix if not self.use_relu else F.relu(self.matrix)
        modified_embedding = e @ matrix
        return modified_embedding

    def encode(self, embedding):
        # TODO: remove this function
        # user can call this from modal endpoint, model after endpoint
        # look into how huggingface inference works, make it compatible
        # returns an actual embedding
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        # TODO: modify so that batch gets texts instead of embeddings
        embeddings_a, embeddings_b = batch
        modified_embedding_a = self(embeddings_a)
        modified_embedding_b = self(embeddings_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py

        # TODO: modify so that batch gets texts instead of embeddings
        text_a, text_b = batch
        print(len(text_a))
        modified_embedding_a = self(text_a)
        modified_embedding_b = self(text_b)
        # print(modified_embedding_a, modified_embedding_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)

        # TODO: for some reason i need to explicilty set batch size, investigate this
        self.log("val_loss", loss, batch_size=len(text_a))

        # treshold to decide if matching ??
        # TODO: fix f1 scores, probably just convert array to be diagonal and input that
        threshold = 0.5
        pred_labels = scores > threshold
        # self.log("val_f1", self.f1(pred_labels.int(), labels.int()))

        # TODO: for some reason i need to explicilty set batch size, investigate this
        self.log("val_f1", 0.5, batch_size=len(text_a))

        # embedding_1, embedding_2 = batch
        # similarity = self(embedding_1, embedding_2)
        # target_similarity = torch.ones(similarity.shape, device=self.device)  # not sure
        # pos_weight = torch.tensor([0.89 / 0.11], device=self.device)
        # loss = F.binary_cross_entropy_with_logits(
        #     similarity, target_similarity, pos_weight=pos_weight, reduction="mean"
        # )
        # self.log("val_loss", loss)
        # pred_labels = torch.sigmoid(similarity) > 0.5

        # self.log("val_recall", self.recall(pred_labels.int(), target_similarity.int()))
        # self.log("val_f1", self.f1(pred_labels.int(), target_similarity.int()))
        # self.log("val_acc", self.acc(pred_labels.int(), target_similarity.int()))
        # self.log(
        #     "val_precision", self.precision(pred_labels.int(), target_similarity.int())
        # )
        # self.log(
        #     "val_auc",
        #     self.auc(
        #         torch.sigmoid(similarity).float().squeeze(-1),
        #         target_similarity.float().squeeze(-1),
        #     ),
        # )
        # self.log("val_removed", 1 - (sum(pred_labels) / len(pred_labels)))
        pass

    def test_step(self, batch, batch_idx):
        # MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
        # inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py

        # TODO: modify so that batch gets texts instead of embeddings
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

        # embedding_1, embedding_2 = batch
        # similarity = self(embedding_1, embedding_2)
        # target_similarity = torch.ones(similarity.shape, device=self.device)  # not sure
        # pos_weight = torch.tensor([0.89 / 0.11], device=self.device)
        # loss = F.binary_cross_entropy_with_logits(
        #     similarity, target_similarity, pos_weight=pos_weight, reduction="mean"
        # )
        # self.log("test_loss", loss)
        # pred_labels = torch.sigmoid(similarity) > 0.5
        # self.log("test_recall", self.recall(pred_labels.int(), target_similarity.int()))
        # self.log("test_f1", self.f1(pred_labels.int(), target_similarity.int()))
        # self.log("test_acc", self.acc(pred_labels.int(), target_similarity.int()))
        # self.log(
        #     "test_precision", self.precision(pred_labels.int(), target_similarity.int())
        # )
        # self.log(
        #     "test_auc",
        #     self.auc(
        #         torch.sigmoid(similarity).float().squeeze(-1),
        #         target_similarity.float().squeeze(-1),
        #     ),
        # )
