import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from ogb.utils import smiles2graph
from torch_geometric.datasets import QM9
from model.gin_model import GNN, GNN_MoleculeSTM
from model.tokenGT import TokenGT
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


sg_ckpt_path = "/data/all_checkpoints/MT_mistral7b_string+graph_12ep_0304/epoch=11-step=61512.ckpt"
ms_ckpt_path = "/data/all_checkpoints/MoleculeSTM/molecule_model.pth"

gin_weight_list = [
    "atom_encoder.atom_embedding_list.0.weight",
    "atom_encoder.atom_embedding_list.1.weight",
    "atom_encoder.atom_embedding_list.2.weight",
    "atom_encoder.atom_embedding_list.3.weight",
    "atom_encoder.atom_embedding_list.4.weight",
    "atom_encoder.atom_embedding_list.5.weight",
    "atom_encoder.atom_embedding_list.6.weight",
    "atom_encoder.atom_embedding_list.7.weight",
    "atom_encoder.atom_embedding_list.8.weight",
    "gnns.0.eps",
    "gnns.0.mlp.0.weight",
    "gnns.0.mlp.0.bias",
    "gnns.0.mlp.1.weight",
    "gnns.0.mlp.1.bias",
    "gnns.0.mlp.3.weight",
    "gnns.0.mlp.3.bias",
    "gnns.0.bond_encoder.bond_embedding_list.0.weight",
    "gnns.0.bond_encoder.bond_embedding_list.1.weight",
    "gnns.0.bond_encoder.bond_embedding_list.2.weight",
    "gnns.1.eps",
    "gnns.1.mlp.0.weight",
    "gnns.1.mlp.0.bias",
    "gnns.1.mlp.1.weight",
    "gnns.1.mlp.1.bias",
    "gnns.1.mlp.3.weight",
    "gnns.1.mlp.3.bias",
    "gnns.1.bond_encoder.bond_embedding_list.0.weight",
    "gnns.1.bond_encoder.bond_embedding_list.1.weight",
    "gnns.1.bond_encoder.bond_embedding_list.2.weight",
    "gnns.2.eps",
    "gnns.2.mlp.0.weight",
    "gnns.2.mlp.0.bias",
    "gnns.2.mlp.1.weight",
    "gnns.2.mlp.1.bias",
    "gnns.2.mlp.3.weight",
    "gnns.2.mlp.3.bias",
    "gnns.2.bond_encoder.bond_embedding_list.0.weight",
    "gnns.2.bond_encoder.bond_embedding_list.1.weight",
    "gnns.2.bond_encoder.bond_embedding_list.2.weight",
    "gnns.3.eps",
    "gnns.3.mlp.0.weight",
    "gnns.3.mlp.0.bias",
    "gnns.3.mlp.1.weight",
    "gnns.3.mlp.1.bias",
    "gnns.3.mlp.3.weight",
    "gnns.3.mlp.3.bias",
    "gnns.3.bond_encoder.bond_embedding_list.0.weight",
    "gnns.3.bond_encoder.bond_embedding_list.1.weight",
    "gnns.3.bond_encoder.bond_embedding_list.2.weight",
    "gnns.4.eps",
    "gnns.4.mlp.0.weight",
    "gnns.4.mlp.0.bias",
    "gnns.4.mlp.1.weight",
    "gnns.4.mlp.1.bias",
    "gnns.4.mlp.3.weight",
    "gnns.4.mlp.3.bias",
    "gnns.4.bond_encoder.bond_embedding_list.0.weight",
    "gnns.4.bond_encoder.bond_embedding_list.1.weight",
    "gnns.4.bond_encoder.bond_embedding_list.2.weight",
    "batch_norms.0.weight",
    "batch_norms.0.bias",
    "batch_norms.1.weight",
    "batch_norms.1.bias",
    "batch_norms.2.weight",
    "batch_norms.2.bias",
    "batch_norms.3.weight",
    "batch_norms.3.bias",
    "batch_norms.4.weight",
    "batch_norms.4.bias",
    # "blip2model.ln_graph.weight",
    # "blip2model.ln_graph.bias",
]

qformer_weight_list = [
    "bert.embeddings.LayerNorm.weight",
    "bert.embeddings.LayerNorm.bias",
    "bert.encoder.layer.0.attention.self.query.weight",
    "bert.encoder.layer.0.attention.self.query.bias",
    "bert.encoder.layer.0.attention.self.key.weight",
    "bert.encoder.layer.0.attention.self.key.bias",
    "bert.encoder.layer.0.attention.self.value.weight",
    "bert.encoder.layer.0.attention.self.value.bias",
    "bert.encoder.layer.0.attention.output.dense.weight",
    "bert.encoder.layer.0.attention.output.dense.bias",
    "bert.encoder.layer.0.attention.output.LayerNorm.weight",
    "bert.encoder.layer.0.attention.output.LayerNorm.bias",
    "bert.encoder.layer.0.crossattention.self.query.weight",
    "bert.encoder.layer.0.crossattention.self.query.bias",
    "bert.encoder.layer.0.crossattention.self.key.weight",
    "bert.encoder.layer.0.crossattention.self.key.bias",
    "bert.encoder.layer.0.crossattention.self.value.weight",
    "bert.encoder.layer.0.crossattention.self.value.bias",
    "bert.encoder.layer.0.crossattention.output.dense.weight",
    "bert.encoder.layer.0.crossattention.output.dense.bias",
    "bert.encoder.layer.0.crossattention.output.LayerNorm.weight",
    "bert.encoder.layer.0.crossattention.output.LayerNorm.bias",
    "bert.encoder.layer.0.intermediate_query.dense.weight",
    "bert.encoder.layer.0.intermediate_query.dense.bias",
    "bert.encoder.layer.0.output_query.dense.weight",
    "bert.encoder.layer.0.output_query.dense.bias",
    "bert.encoder.layer.0.output_query.LayerNorm.weight",
    "bert.encoder.layer.0.output_query.LayerNorm.bias",
    "bert.encoder.layer.1.attention.self.query.weight",
    "bert.encoder.layer.1.attention.self.query.bias",
    "bert.encoder.layer.1.attention.self.key.weight",
    "bert.encoder.layer.1.attention.self.key.bias",
    "bert.encoder.layer.1.attention.self.value.weight",
    "bert.encoder.layer.1.attention.self.value.bias",
    "bert.encoder.layer.1.attention.output.dense.weight",
    "bert.encoder.layer.1.attention.output.dense.bias",
    "bert.encoder.layer.1.attention.output.LayerNorm.weight",
    "bert.encoder.layer.1.attention.output.LayerNorm.bias",
    "bert.encoder.layer.1.intermediate_query.dense.weight",
    "bert.encoder.layer.1.intermediate_query.dense.bias",
    "bert.encoder.layer.1.output_query.dense.weight",
    "bert.encoder.layer.1.output_query.dense.bias",
    "bert.encoder.layer.1.output_query.LayerNorm.weight",
    "bert.encoder.layer.1.output_query.LayerNorm.bias",
    "bert.encoder.layer.2.attention.self.query.weight",
    "bert.encoder.layer.2.attention.self.query.bias",
    "bert.encoder.layer.2.attention.self.key.weight",
    "bert.encoder.layer.2.attention.self.key.bias",
    "bert.encoder.layer.2.attention.self.value.weight",
    "bert.encoder.layer.2.attention.self.value.bias",
    "bert.encoder.layer.2.attention.output.dense.weight",
    "bert.encoder.layer.2.attention.output.dense.bias",
    "bert.encoder.layer.2.attention.output.LayerNorm.weight",
    "bert.encoder.layer.2.attention.output.LayerNorm.bias",
    "bert.encoder.layer.2.crossattention.self.query.weight",
    "bert.encoder.layer.2.crossattention.self.query.bias",
    "bert.encoder.layer.2.crossattention.self.key.weight",
    "bert.encoder.layer.2.crossattention.self.key.bias",
    "bert.encoder.layer.2.crossattention.self.value.weight",
    "bert.encoder.layer.2.crossattention.self.value.bias",
    "bert.encoder.layer.2.crossattention.output.dense.weight",
    "bert.encoder.layer.2.crossattention.output.dense.bias",
    "bert.encoder.layer.2.crossattention.output.LayerNorm.weight",
    "bert.encoder.layer.2.crossattention.output.LayerNorm.bias",
    "bert.encoder.layer.2.intermediate_query.dense.weight",
    "bert.encoder.layer.2.intermediate_query.dense.bias",
    "bert.encoder.layer.2.output_query.dense.weight",
    "bert.encoder.layer.2.output_query.dense.bias",
    "bert.encoder.layer.2.output_query.LayerNorm.weight",
    "bert.encoder.layer.2.output_query.LayerNorm.bias",
    "bert.encoder.layer.3.attention.self.query.weight",
    "bert.encoder.layer.3.attention.self.query.bias",
    "bert.encoder.layer.3.attention.self.key.weight",
    "bert.encoder.layer.3.attention.self.key.bias",
    "bert.encoder.layer.3.attention.self.value.weight",
    "bert.encoder.layer.3.attention.self.value.bias",
    "bert.encoder.layer.3.attention.output.dense.weight",
    "bert.encoder.layer.3.attention.output.dense.bias",
    "bert.encoder.layer.3.attention.output.LayerNorm.weight",
    "bert.encoder.layer.3.attention.output.LayerNorm.bias",
    "bert.encoder.layer.3.intermediate_query.dense.weight",
    "bert.encoder.layer.3.intermediate_query.dense.bias",
    "bert.encoder.layer.3.output_query.dense.weight",
    "bert.encoder.layer.3.output_query.dense.bias",
    "bert.encoder.layer.3.output_query.LayerNorm.weight",
    "bert.encoder.layer.3.output_query.LayerNorm.bias",
    "bert.encoder.layer.4.attention.self.query.weight",
    "bert.encoder.layer.4.attention.self.query.bias",
    "bert.encoder.layer.4.attention.self.key.weight",
    "bert.encoder.layer.4.attention.self.key.bias",
    "bert.encoder.layer.4.attention.self.value.weight",
    "bert.encoder.layer.4.attention.self.value.bias",
    "bert.encoder.layer.4.attention.output.dense.weight",
    "bert.encoder.layer.4.attention.output.dense.bias",
    "bert.encoder.layer.4.attention.output.LayerNorm.weight",
    "bert.encoder.layer.4.attention.output.LayerNorm.bias",
    "bert.encoder.layer.4.crossattention.self.query.weight",
    "bert.encoder.layer.4.crossattention.self.query.bias",
    "bert.encoder.layer.4.crossattention.self.key.weight",
    "bert.encoder.layer.4.crossattention.self.key.bias",
    "bert.encoder.layer.4.crossattention.self.value.weight",
    "bert.encoder.layer.4.crossattention.self.value.bias",
    "bert.encoder.layer.4.crossattention.output.dense.weight",
    "bert.encoder.layer.4.crossattention.output.dense.bias",
    "bert.encoder.layer.4.crossattention.output.LayerNorm.weight",
    "bert.encoder.layer.4.crossattention.output.LayerNorm.bias",
    "bert.encoder.layer.4.intermediate_query.dense.weight",
    "bert.encoder.layer.4.intermediate_query.dense.bias",
    "bert.encoder.layer.4.output_query.dense.weight",
    "bert.encoder.layer.4.output_query.dense.bias",
    "bert.encoder.layer.4.output_query.LayerNorm.weight",
    "bert.encoder.layer.4.output_query.LayerNorm.bias",
    # "blip2model.opt_proj.weight",
    # "blip2model.opt_proj.bias",
    # "blip2model.query_tokens",
]

qm9_name2idx = {
    "dipole": 0,
    "isotropic": 1,
    # "homo": 2,
    # "lumo": 3,
    # "gap": 4,
    "electronic": 5,
    "vibrational": 6,
    "internalEnergy0K": 7,
    "internalEnergy298K": 8,
    "enthalpy": 9,
    "freeEnergy298K": 10,
    "capavity": 11,
    "atomizationEnergy0K": 12,
    "atomizationEnergy298K": 13,
    "atomizationEnthalpy298K": 14,
    "atomizationFreeEnergy298K": 15,
    "rotationA": 16,
    "rotationB": 17,
    "rotationC": 18,
}

qm9_idx2name = {
    0: "dipole",
    1: "isotropic",
    # 2: "homo",
    # 3: "lumo",
    # 4: "gap",
    2: "electronic",
    3: "vibrational",
    4: "internalEnergy0K",
    5: "internalEnergy298K",
    6: "enthalpy",
    7: "freeEnergy298K",
    8: "capavity",
    9: "atomizationEnergy0K",
    10: "atomizationEnergy298K",
    11: "atomizationEnthalpy298K",
    12: "atomizationFreeEnergy298K",
    13: "rotationA",
    14: "rotationB",
    15: "rotationC"
}

model_settings = {
    "GM_GM_-_-": {
        "backbone": "GIN->MLP",
        "activate_GIN": True,
        "activate_QFormer": False,
        "activate_MLP": True,
        "initial_GIN": None,
        "initial_QFormer": None,
        "lr": 1e-4,
    },
    # "GM_M_-_-": {
    #     "backbone": "GIN->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": None,
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GM_M_MoleculeSTM_-": {
    #     "backbone": "GIN->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": "MoleculeSTM",
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GM_M_StringGraph_-": {
    #     "backbone": "GIN->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": "StringGraph",
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GM_M_MolPO_-": {
    #     "backbone": "GIN->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": "MolPO",
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GQM_QM_StringGraph_-": {
    #     "backbone": "GIN->QFormer->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": True,
    #     "activate_MLP": True,
    #     "initial_GIN": "StringGraph",
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GQM_QM_MolPO_-": {
    #     "backbone": "GIN->QFormer->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": True,
    #     "activate_MLP": True,
    #     "initial_GIN": "MolPO",
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
    # "GQM_M_StringGraph_StringGraph": {
    #     "backbone": "GIN->QFormer->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": "StringGraph",
    #     "initial_QFormer": "StringGraph",
    #     "lr": 1e-4,
    # },
    # "GQM_M_MolPO_MolPO": {
    #     "backbone": "GIN->QFormer->MLP",
    #     "activate_GIN": False,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": "MolPO",
    #     "initial_QFormer": "MolPO",
    #     "lr": 1e-4,
    # },
    }


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QM9DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        QM9(root='./data/qm9')

    def setup(self, stage=None):
        dataset = QM9(root='./data/qm9')
        smiles_list = dataset.smiles
        graphs = []
        targets = dataset.data.y # (133885, 19)
        # Remove indices 2, 3, 4 at dim=1 (homo, lumo, gap) from target
        # Remove indices
        targets = torch.cat( (dataset.data.y[:, :2], dataset.data.y[:, 5:16]), dim=1)
        # Normalization
        self.targets_min = targets.min(dim=0).values
        self.targets_max = targets.max(dim=0).values
        targets_scaled = (targets - self.targets_min) / (self.targets_max - self.targets_min)


        for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
            try:
                graph_dict = smiles2graph(smiles)
            except:
                # print(f"Error in smiles: {smiles}")
                continue
            graph_data = Data(
                x=torch.tensor(graph_dict['node_feat'], dtype=torch.long),
                edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph_dict['edge_feat'], dtype=torch.long),
                # y=targets[i]
                y=targets_scaled[i]
            )
            graphs.append(graph_data)
            # if i == 10000:
            #     break

        val_size = 3000
        test_size = 3000

        self.train_set = graphs[:-val_size-test_size]
        self.val_set = graphs[-val_size-test_size:-test_size]
        self.test_set = graphs[-test_size:]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class QM9Trainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = kwargs['backbone']
        self.activate_GIN = kwargs['activate_GIN']
        self.activate_QFormer = kwargs['activate_QFormer']
        self.activate_MLP = kwargs['activate_MLP']
        self.initial_GIN = kwargs['initial_GIN']
        self.initial_QFormer = kwargs['initial_QFormer']
        self.lr = kwargs['lr']

        self.gnn = TokenGT(input_feat_dim=9, hidden_dim=1024,
            num_layers=5, num_heads=8, method="laplacian", d_p=64, d_e=64,
            use_graph_token=True)
        # self.gnn = TokenGT(input_feat_dim=9, hidden_dim=1024,
        #     num_layers=5, num_heads=8,
        #     num_classes=3, method="orf", d_p=9, d_e=4,
        #     use_graph_token=True)

        self.ln_graph = LayerNorm(1024)

        self.mlp = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 13)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        output, _ = self.gnn(x, edge_index, edge_attr, batch)
        graph_embedding = output[:, :1, :]
        # node_embedding = output[:, 1:, :]
        graph_embedding = self.ln_graph(graph_embedding)
        pred = self.mlp(graph_embedding.squeeze(1))

        return pred

    def training_step(self, batch, batch_idx):
        targets_min = self.trainer.datamodule.targets_min.to(self.device)
        targets_max = self.trainer.datamodule.targets_max.to(self.device)

        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.reshape(-1, pred.size(1))
        pred_unscaled = pred * (targets_max - targets_min) + targets_min
        y_unscaled = y * (targets_max - targets_min) + targets_min
        total_loss = 0.
        for i in range(pred.size(1)):
            loss = F.mse_loss(pred[:, i], y[:, i])
            mse = F.mse_loss(pred_unscaled[:, i], y_unscaled[:, i])
            mae = F.l1_loss(pred_unscaled[:, i], y_unscaled[:, i])
            total_loss += loss
            self.log(f'train/loss_{qm9_idx2name[i]}', loss, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'train/MSE_{qm9_idx2name[i]}', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'train/MAE_{qm9_idx2name[i]}', mae, sync_dist=True, batch_size=batch.y.size(0))
        self.log('train/total_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        targets_min = self.trainer.datamodule.targets_min.to(self.device)
        targets_max = self.trainer.datamodule.targets_max.to(self.device)

        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.reshape(-1, pred.size(1))
        pred_unscaled = pred * (targets_max - targets_min) + targets_min
        y_unscaled = y * (targets_max - targets_min) + targets_min
        total_loss = 0.
        for i in range(pred.size(1)):
            loss = F.mse_loss(pred[:, i], y[:, i])
            mse = F.mse_loss(pred_unscaled[:, i], y_unscaled[:, i])
            mae = F.l1_loss(pred_unscaled[:, i], y_unscaled[:, i])
            total_loss += loss
            self.log(f'validation/loss_{qm9_idx2name[i]}', loss, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'validation/MSE_{qm9_idx2name[i]}', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'validation/MAE_{qm9_idx2name[i]}', mae, sync_dist=True, batch_size=batch.y.size(0))
        self.log('validation/total_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))

    def test_step(self, batch, batch_idx):
        targets_min = self.trainer.datamodule.targets_min.to(self.device)
        targets_max = self.trainer.datamodule.targets_max.to(self.device)

        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.reshape(-1, pred.size(1))
        pred_unscaled = pred * (targets_max - targets_min) + targets_min
        y_unscaled = y * (targets_max - targets_min) + targets_min
        total_loss = 0.
        for i in range(pred.size(1)):
            loss = F.mse_loss(pred[:, i], y[:, i])
            mse = F.mse_loss(pred_unscaled[:, i], y_unscaled[:, i])
            mae = F.l1_loss(pred_unscaled[:, i], y_unscaled[:, i])
            total_loss += loss
            self.log(f'test/loss_{qm9_idx2name[i]}', loss, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'test/MSE_{qm9_idx2name[i]}', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log(f'test/MAE_{qm9_idx2name[i]}', mae, sync_dist=True, batch_size=batch.y.size(0))
        self.log('test/total_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        # log gin parameter weights mean
        for name, param in self.gnn.state_dict().items():
            try:
                self.logger.log_metrics({f"GNN_parameters/{name}_mean": param.mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"GNN_parameters/{name}": param}, step=self.current_epoch) # for scalar values such as running_var of BatchNorm
        # log MLP parameter weights mean
        for name, param in self.mlp.state_dict().items():
            try:
                self.logger.log_metrics({f"MLP_parameters/{name}_mean": param.float().mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"MLP_parameters/{name}": param}, step=self.current_epoch)
        # Log learning rate
        self.logger.log_metrics({'hyperparameters/lr': self.lr}, step=self.current_epoch)


# 학습 실행 코드
if __name__ == '__main__':

    # tasks = list(qm9_name2idx.keys())
    # model_types = list(model_settings.keys())
    model_type = "GM_GM_-_-"

    data_module = QM9DataModule(batch_size=128)
    pl_model = QM9Trainer(**model_settings[model_type])

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/total_loss',
        mode='min',
        save_top_k=1,
        filename='best-model'
    )

    trainer = pl.Trainer(
        max_epochs=1500,
        accelerator='auto',
        devices=[4,5,6,7],
        # devices=[4],
        default_root_dir=f'./gnn_ablation/all_except_lumo_homo_gap_scaled_tokengt/{model_type}',
        # default_root_dir=f'./gnn_ablation/debug/{model_type}',
        callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=0.5
    )
    trainer.fit(pl_model, datamodule=data_module)
    best_model_path = checkpoint_callback.best_model_path
    trainer.test(ckpt_path=best_model_path, datamodule=data_module)
