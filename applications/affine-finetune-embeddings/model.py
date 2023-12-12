import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import util

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
    def __init__(self, embedding_size, n_dims, dropout_fraction, lr, use_relu):
        super(SimilarityModel, self).__init__()
        # TODO: add base embedding and tokenization model here
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
        self.save_hyperparameters()
        self.scale = 20  # MNRL loss scale: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss

    def forward(self, embedding_1):
        # TODO: modify to take in text and return modified embedding
        e = F.dropout(embedding_1, p=self.dropout_fraction)
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
        embeddings_a, embeddings_b = batch
        modified_embedding_a = self(embeddings_a)
        modified_embedding_b = self(embeddings_b)
        # print(modified_embedding_a, modified_embedding_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)
        self.log("val_loss", loss)

        # treshold to decide if matching ??
        # TODO: fix f1 scores, probably just convert array to be diagonal and input that
        threshold = 0.5
        pred_labels = scores > threshold
        # self.log("val_f1", self.f1(pred_labels.int(), labels.int()))

        self.log("val_f1", 0.5)

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
        embeddings_a, embeddings_b = batch
        modified_embedding_a = self(embeddings_a)
        modified_embedding_b = self(embeddings_b)
        scores = util.cos_sim(modified_embedding_a, modified_embedding_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = F.cross_entropy(scores, labels)
        self.log("test_loss", loss)

        # treshold to decide if matching ??
        # TODO: fix f1 scores, probably just convert array to be diagonal and input that
        threshold = 0.5
        pred_labels = scores > threshold
        # self.log("val_f1", self.f1(pred_labels.int(), labels.int()))
        self.log("test_f1", 0.5)

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
