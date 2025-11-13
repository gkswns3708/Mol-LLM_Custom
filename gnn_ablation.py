import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from rdkit import Chem
from rdkit.Chem import Fragments
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from ogb.utils import smiles2graph
from torch_geometric.datasets import QM9
from model.gin_model import GNN, GNN_MoleculeSTM
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from tqdm import tqdm
from rdkit import RDLogger
from sklearn.metrics import accuracy_score, roc_auc_score
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


sg_ckpt_path = "/data/all_checkpoints/MT_mistral7b_string+graph_12ep_0304/epoch=11-step=61512.ckpt"
ms_ckpt_path = "/data/all_checkpoints/MoleculeSTM/molecule_model.pth"

RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

func_idx2name = {idx: name for idx, name in enumerate(RDKIT_PROPS)}

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
    "homo": 2,
    "lumo": 3,
    "gap": 4,
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

model_settings = {
    # "GM_GM_-_-": {
    #     "backbone": "GIN->MLP",
    #     "activate_GIN": True,
    #     "activate_QFormer": False,
    #     "activate_MLP": True,
    #     "initial_GIN": None,
    #     "initial_QFormer": None,
    #     "lr": 1e-4,
    # },
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
    "GQM_QM_MolPO_-": {
        "backbone": "GIN->QFormer->MLP",
        "activate_GIN": False,
        "activate_QFormer": True,
        "activate_MLP": True,
        "initial_GIN": "MolPO",
        "initial_QFormer": None,
        "lr": 1e-4,
    },
    "GQM_M_StringGraph_StringGraph": {
        "backbone": "GIN->QFormer->MLP",
        "activate_GIN": False,
        "activate_QFormer": False,
        "activate_MLP": True,
        "initial_GIN": "StringGraph",
        "initial_QFormer": "StringGraph",
        "lr": 1e-4,
    },
    "GQM_M_MolPO_MolPO": {
        "backbone": "GIN->QFormer->MLP",
        "activate_GIN": False,
        "activate_QFormer": False,
        "activate_MLP": True,
        "initial_GIN": "MolPO",
        "initial_QFormer": "MolPO",
        "lr": 1e-4,
    },
    }


def check_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    fg_presence = {}
    for func_name in RDKIT_PROPS:
        fg_presence[func_name] = -1
        try:
            func = getattr(Fragments, func_name)
            count = func(mol)
            fg_presence[func_name] = count
        except Exception:
            fg_presence[func_name] = -1
            
    return fg_presence


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QM9DataModule(pl.LightningDataModule):
    def __init__(self, task, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.task = task

    def prepare_data(self):
        QM9(root='./data/qm9')

    def setup(self, stage=None):
        dataset = QM9(root='./data/qm9')
        smiles_list = dataset.smiles
        graphs = []
        task_idx = qm9_name2idx[self.task]
        targets = dataset.data.y[:, task_idx:task_idx+1]

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
                y=targets[i]
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


class FunctionalGroupDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
    
    def prepare_data(self):
        QM9(root='./data/qm9')
    
    def setup(self, stage=None):
        dataset = QM9(root='./data/qm9')
        smiles_list = dataset.smiles
        graphs = []

        for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
            try:
                graph_dict = smiles2graph(smiles)
                func_groups = check_functional_groups(smiles)
                # Convert the functional groups to a boolean tensor
                func_groups = torch.tensor(list(func_groups.values()), dtype=torch.float32) > 0
            except:
                # print(f"Error in smiles: {smiles}")
                continue
            graph_data = Data(
                x=torch.tensor(graph_dict['node_feat'], dtype=torch.long),
                edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph_dict['edge_feat'], dtype=torch.long),
                y=func_groups
            )
            graphs.append(graph_data)
            # if i == 50000:
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


class TaskModel(pl.LightningModule):
    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.backbone = kwargs['backbone']
        self.activate_GIN = kwargs['activate_GIN']
        self.activate_QFormer = kwargs['activate_QFormer']
        self.activate_MLP = kwargs['activate_MLP']
        self.initial_GIN = kwargs['initial_GIN']
        self.initial_QFormer = kwargs['initial_QFormer']
        self.lr = kwargs['lr']

        sg_ckpt_path = "/data/all_checkpoints/MT_mistral7b_string+graph_12ep_0304/epoch=11-step=61512.ckpt"
        mp_ckpt_path = "/data/all_checkpoints/MT_mistral7b_string+graph2string+graph_mdpo_5ep_lr_0124/epoch=04-step=12154.ckpt"
        ms_ckpt_path = "/data/all_checkpoints/MoleculeSTM/molecule_model.pth"

        sg_ckpt = torch.load(sg_ckpt_path, map_location='cpu')
        mp_ckpt = torch.load(mp_ckpt_path, map_location='cpu')
        ms_ckpt = torch.load(ms_ckpt_path, map_location='cpu')

        ################## Initialize GNN ##################
        self.gin = GNN_MoleculeSTM(
            num_layer=5,
            emb_dim=300,
            gnn_type="gin",
            drop_ratio=0.0,
            JK="last",
            # args=args,
        )
        self.ln_graph = LayerNorm(300)
        if task == "funcgroup":
            self.last_dim = 85
        elif task == "qm9":
            self.last_dim = 1
        else:
            raise ValueError(f"Invalid task: {task}")


        ################## Initialize MLP ##################
        if self.backbone == "GIN->MLP":
            self.mlp = nn.Sequential(
                nn.Linear(300, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, self.last_dim)
            )
        elif self.backbone == "GIN->QFormer->MLP":
            self.query_encoder = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Linear(1024, 50),
                nn.ReLU(),
            )
            self.mlp = nn.Sequential(
                nn.Linear(1600, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, self.last_dim)
            )
        else:
            raise ValueError(f"Invalid backbone type: {self.backbone}")

        ################## Initialize Qformer ##################
        bert_name = "allenai/scibert_scivocab_uncased"
        n_query_token = 32
        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = 300
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = n_query_token
        encoder_config.num_hidden_layers = 5
        self.Qformer = BertLMHeadModel.from_pretrained(bert_name, config=encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, n_query_token, encoder_config.hidden_size)
        )
        # query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        ## remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096
        )

        ################## Load pre-trained weights - GIN ##################
        if self.initial_GIN == "StringGraph":
            for name in gin_weight_list:
                self.gin.state_dict()[name].copy_(sg_ckpt['state_dict']['blip2model.graph_encoder.'+name])
            self.ln_graph.load_state_dict({
                "weight": sg_ckpt['state_dict']['blip2model.ln_graph.weight'],
                "bias": sg_ckpt['state_dict']['blip2model.ln_graph.bias']
            }, strict=True)
        elif self.initial_GIN == "MolPO":
            for name in gin_weight_list:
                self.gin.state_dict()[name].copy_(mp_ckpt['state_dict']['blip2model.graph_encoder.'+name])
            self.ln_graph.load_state_dict({
                "weight": mp_ckpt['state_dict']['blip2model.ln_graph.weight'],
                "bias": mp_ckpt['state_dict']['blip2model.ln_graph.bias']
            }, strict=True)
        elif self.initial_GIN == "MoleculeSTM":
            for name in gin_weight_list:
                self.gin.state_dict()[name].copy_(ms_ckpt['molecule_node_model.'+name])
        else:
            pass

        ################## Load pre-trained weights - Q-Former ##################
        if self.initial_QFormer == "StringGraph":
            for name in qformer_weight_list:
                self.Qformer.state_dict()[name].copy_(sg_ckpt['state_dict']['blip2model.Qformer.'+name])
            self.query_tokens.data.copy_(sg_ckpt['state_dict']['blip2model.query_tokens'])
            # self.Qformer.bert.embeddings.LayerNorm.load_state_dict({
            #     "weight": sg_ckpt['state_dict']['blip2model.Qformer.bert.embeddings.LayerNorm.weight'],
            #     "bias": sg_ckpt['state_dict']['blip2model.Qformer.bert.embeddings.LayerNorm.bias']
            # }, strict=True)
            self.opt_proj.load_state_dict({
                "weight": sg_ckpt['state_dict']['blip2model.opt_proj.weight'],
                "bias": sg_ckpt['state_dict']['blip2model.opt_proj.bias']
            }, strict=True)
        elif self.initial_QFormer == "MolPO":
            for name in qformer_weight_list:
                self.Qformer.state_dict()[name].copy_(mp_ckpt['state_dict']['blip2model.Qformer.'+name])
            self.query_tokens.data.copy_(mp_ckpt['state_dict']['blip2model.query_tokens'])
            # self.Qformer.bert.embeddings.LayerNorm.load_state_dict({
            #     "weight": sg_ckpt['state_dict']['blip2model.Qformer.bert.embeddings.LayerNorm.weight'],
            #     "bias": sg_ckpt['state_dict']['blip2model.Qformer.bert.embeddings.LayerNorm.bias']
            # }, strict=True)
            self.opt_proj.load_state_dict({
                "weight": mp_ckpt['state_dict']['blip2model.opt_proj.weight'],
                "bias": mp_ckpt['state_dict']['blip2model.opt_proj.bias']
            }, strict=True)
        
        ################## Freeze parameters - GIN ##################
        if not self.activate_GIN:
            for param in self.gin.parameters():
                param.requires_grad = False
        
        ################## Freeze parameters - Q-Former ##################
        if not self.activate_QFormer:
            for param in self.Qformer.parameters():
                param.requires_grad = False

        ################## Freeze parameters - MLP ##################
        if not self.activate_MLP:
            for param in self.mlp.parameters():
                param.requires_grad = False

    def forward(self, x, edge_index, edge_attr, batch):
        if self.task == "funcgroup":
            if self.backbone == "GIN->MLP":
                output, _ = self.gin(x, edge_index, edge_attr, batch)
                graph_embedding = output[:, :1, :]
                # node_embedding = output[:, 1:, :]
                pred = self.mlp(graph_embedding.squeeze(1))
            elif self.backbone == "GIN->QFormer->MLP":
                mol_embeds, mol_masks = self.gin(
                    x, edge_index, edge_attr, batch
                )
                mol_embeds = self.ln_graph(mol_embeds, mol_masks)
                query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=mol_embeds,
                    encoder_attention_mask=mol_masks,
                    return_dict=True,
                )
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
                mol_tokens = self.query_encoder(mol_tokens)
                mol_tokens = mol_tokens.view(mol_tokens.size(0), -1)
                pred = self.mlp(mol_tokens)
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone}")
        elif self.task == "qm9":
            if self.backbone == "GIN->MLP":
                output, _ = self.gin(x, edge_index, edge_attr, batch)
                graph_embedding = output[:, :1, :]
                # node_embedding = output[:, 1:, :]
                pred = self.mlp(graph_embedding.squeeze(1)).squeeze(1)
            elif self.backbone == "GIN->QFormer->MLP":
                mol_embeds, mol_masks = self.gin(
                    x, edge_index, edge_attr, batch
                )
                mol_embeds = self.ln_graph(mol_embeds, mol_masks)
                query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=mol_embeds,
                    encoder_attention_mask=mol_masks,
                    return_dict=True,
                )
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
                mol_tokens = self.query_encoder(mol_tokens)
                mol_tokens = mol_tokens.view(mol_tokens.size(0), -1)
                pred = self.mlp(mol_tokens).squeeze(1)
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone}")
        else:
            raise ValueError(f"Invalid task: {self.task}")

        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.task == "funcgroup":
            pred = torch.sigmoid(pred)
            task_size = pred.size(1)
            y = batch.y.view(-1, task_size).float()
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
            self.log('train/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
        else:
            loss = F.mse_loss(pred, batch.y)
            mse = loss
            mae = F.l1_loss(pred, batch.y)
            self.log('train/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
            self.log('train/MSE', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log('train/MAE', mae, sync_dist=True, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.task == "funcgroup":
            pred = torch.sigmoid(pred)
            task_size = pred.size(1)
            y = batch.y.view(-1, task_size).float()
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
            self.log('validation/online_loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
            return_dict = {
                "pred": pred,
                "y": y,
                "loss": loss,
            }
        else:
            loss = F.mse_loss(pred, batch.y)
            mse = loss
            mae = F.l1_loss(pred, batch.y)
            self.log('validation/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
            self.log('validation/MSE', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log('validation/MAE', mae, sync_dist=True, batch_size=batch.y.size(0))
            return_dict = {
                "pred": pred,
                "y": batch.y,
                "loss": loss,
            }
        return return_dict

    def test_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.task == "funcgroup":
            pred = torch.sigmoid(pred)
            task_size = pred.size(1)
            y = batch.y.view(-1, task_size).float()
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
            self.log('test/online_loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
            return_dict = {
                "pred": pred,
                "y": y,
                "loss": loss,
            }
        else:
            pred = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(pred, batch.y)
            mse = loss
            mae = F.l1_loss(pred, batch.y)
            self.log('test/loss', loss, prog_bar=True, sync_dist=True, batch_size=batch.y.size(0))
            self.log('test/MSE', mse, sync_dist=True, batch_size=batch.y.size(0))
            self.log('test/MAE', mae, sync_dist=True, batch_size=batch.y.size(0))
        return return_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        # log gin parameter weights mean
        for name, param in self.gin.state_dict().items():
            try:
                self.logger.log_metrics({f"GIN_parameters/{name}_mean": param.mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"GIN_parameters/{name}": param}, step=self.current_epoch) # for scalar values such as running_var of BatchNorm
        # log Qformer parameter weights mean
        for name, param in self.Qformer.state_dict().items():
            try:
                self.logger.log_metrics({f"Qformer_parameters/{name}_mean": param.float().mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"Qformer_parameters/{name}": param}, step=self.current_epoch)
        # log MLP parameter weights mean
        for name, param in self.mlp.state_dict().items():
            try:
                self.logger.log_metrics({f"MLP_parameters/{name}_mean": param.float().mean()}, step=self.current_epoch)
            except:
                self.logger.log_metrics({f"MLP_parameters/{name}": param}, step=self.current_epoch)
        # Log learning rate
        self.logger.log_metrics({'hyperparameters/lr': self.lr}, step=self.current_epoch)



class ValidationCallback(Callback):
    def __init__(self):
        self.val_outs = []
        self.val_labels = []
        self.val_loss = []
        self.test_outs = []
        self.test_labels = []
        self.test_loss = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_outs.append(outputs['pred'])
        self.val_labels.append(outputs['y'])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_outs.append(outputs['pred'])
        self.test_labels.append(outputs['y'])
    
    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.val_outs, dim=0)
        ys = torch.cat(self.val_labels, dim=0)
        preds = pl_module.all_gather(preds)
        ys = pl_module.all_gather(ys)
        preds = preds.view(-1, preds.size(-1))
        ys = ys.view(-1, ys.size(-1))
        if trainer.is_global_zero:
            if pl_module.task == "funcgroup":
                for i in range(preds.size(1)):
                    pred = preds[:, i]
                    y = ys[:, i]
                    acc = (pred > 0.5).float() == y
                    acc = acc.float().mean().item()
                    auroc = roc_auc_score(y.cpu().numpy(), pred.cpu().numpy())
                    trainer.logger.log_metrics({f'validation_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                    trainer.logger.log_metrics({f'validation_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                    trainer.logger.log_metrics({f'validation_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
                loss = F.binary_cross_entropy_with_logits(preds, ys.float())
                trainer.logger.log_metrics({'validation/total_loss': loss}, step=trainer.current_epoch)
                trainer.logger.log_metrics({'validation/instance_size': preds.size(0)}, step=trainer.current_epoch)
            else:
                mse = F.mse_loss(preds, ys)
                mae = F.l1_loss(preds, ys)
                trainer.logger.log_metrics({'validation/MSE': mse}, step=trainer.current_epoch)
                trainer.logger.log_metrics({'validation/MAE': mae}, step=trainer.current_epoch)
        self.val_outs = []
        self.val_labels = []
    
    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.test_outs, dim=0)
        ys = torch.cat(self.test_labels, dim=0)
        preds = pl_module.all_gather(preds)
        ys = pl_module.all_gather(ys)
        preds = preds.view(-1, preds.size(-1))
        ys = ys.view(-1, ys.size(-1))
        if trainer.is_global_zero:
            if pl_module.task == "funcgroup":
                for i in range(preds.size(1)):
                    pred = preds[:, i]
                    y = ys[:, i]
                    acc = (pred > 0.5).float() == y
                    acc = acc.float().mean().item()
                    auroc = roc_auc_score(y.cpu().numpy(), pred.cpu().numpy())
                    trainer.logger.log_metrics({f'test_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                    trainer.logger.log_metrics({f'test_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                    trainer.logger.log_metrics({f'test_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
                loss = F.binary_cross_entropy_with_logits(preds, ys.float())
                trainer.logger.log_metrics({'test/total_loss': loss}, step=trainer.current_epoch)
                trainer.logger.log_metrics({'test/instance_size': preds.size(0)}, step=trainer.current_epoch)
            else:
                mse = F.mse_loss(preds, ys)
                mae = F.l1_loss(preds, ys)
                trainer.logger.log_metrics({'test/MSE': mse}, step=trainer.current_epoch)
                trainer.logger.log_metrics({'test/MAE': mae}, step=trainer.current_epoch)
        self.test_outs = []
        self.test_labels = []


# 학습 실행 코드
if __name__ == '__main__':

    qm9_tasks = list(qm9_name2idx.keys())
    model_types = list(model_settings.keys())

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/online_loss',
        mode='min',
        save_top_k=1,
        filename='best-model'
    )
    validation_callback = ValidationCallback()



    task = "funcgroup" # "funcgroup" or "qm9"
    if task == "funcgroup":
        for model_type in model_types:
            data_module = FunctionalGroupDataModule(batch_size=128)
            pl_model = TaskModel(task=task, **model_settings[model_type])

            trainer = pl.Trainer(
                max_epochs=300,
                accelerator='auto',
                devices=[4,5,6,7],
                # devices=[4],
                default_root_dir=f'./gnn_ablation/funcgroup/{model_type}',
                callbacks=[checkpoint_callback, validation_callback],
                strategy='ddp_find_unused_parameters_true'
            )
            trainer.fit(pl_model, datamodule=data_module)
            best_model_path = checkpoint_callback.best_model_path
            trainer.test(ckpt_path=best_model_path, datamodule=data_module)
            if trainer.global_rank == 0:
                if os.path.exists(checkpoint_callback.best_model_path):
                    os.remove(checkpoint_callback.best_model_path)
                    print(f"Deleted checkpoint file: {checkpoint_callback.best_model_path}")
        
    else:
        for qm9_task in qm9_tasks:
            for model_type in model_types:
                data_module = QM9DataModule(task=qm9_task, batch_size=128)
                pl_model = TaskModel(**model_settings[model_type])

                trainer = pl.Trainer(
                    max_epochs=50,
                    accelerator='auto',
                    # devices=[4,5,6,7],
                    devices=[4],
                    default_root_dir=f'./gnn_ablation/{qm9_task}/{model_type}',
                    callbacks=[checkpoint_callback, validation_callback],
                    strategy='ddp_find_unused_parameters_true'
                )
                trainer.fit(pl_model, datamodule=data_module)
                best_model_path = checkpoint_callback.best_model_path
                trainer.test(ckpt_path=best_model_path, datamodule=data_module)
                if trainer.global_rank == 0:
                    if os.path.exists(checkpoint_callback.best_model_path):
                        os.remove(checkpoint_callback.best_model_path)
                        print(f"Deleted checkpoint file: {checkpoint_callback.best_model_path}")
