import os
import json
import torch
import pickle
import selfies as sf
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import warnings

from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Config
from model.tokenGT import TokenGT, BERTTokenGT
from sklearn.exceptions import UndefinedMetricWarning
from rdkit import Chem
from rdkit.Chem import Fragments
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch_geometric.data import DataLoader, Data
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
RDKIT_PROPS = RDKIT_PROPS[12:-1]

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
    # "MoleculeSTM": {
    #     "backbone": "MoleculeSTM->MLP",
    #     "activate_GNN": True,
    #     "activate_MLP": True,
    #     "lr": 1e-4,
    #     "max_seq_len": 512,
    #     "gnn_output_dim": 1024,
    # },
    "TokenGT": {
        "backbone": "TokenGT->MLP",
        "activate_GNN": True,
        "activate_MLP": True,
        "lr": 1e-4,
        "max_seq_len": 512,
        "gnn_output_dim": 1024,
    }
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class SelfiesTokenizer:
    def __init__(self):
        if not os.path.exists("Analysis/dump_data/pubchem/selfies_vocab.json"):
            df = pd.read_csv("Analysis/dump_data/pubchem/sparse_based_sampled.csv")
            selfies_tokens = set()
            for i, row in tqdm(enumerate(df.iterrows()), total=len(df), desc="Extracting SELFIES tokens"):
                smiles = row[1]['SMILES']
                try:
                    selfies = sf.encoder(smiles)
                except:
                    # print(f"Error in smiles: {smiles}")
                    continue
                selfies_tokens.update(sf.get_alphabet_from_selfies(list(sf.split_selfies(selfies))))
            self.vocab = sorted(list(selfies_tokens))
            selfies_vocab = {token: idx for idx, token in enumerate(self.vocab)}
            with open("Analysis/dump_data/pubchem/selfies_vocab.json", "w") as f:
                json.dump(selfies_vocab, f, indent=4)
            self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
            self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        else:
            with open("Analysis/dump_data/pubchem/selfies_vocab.json", "r") as f:
                selfies_vocab = json.load(f)
            self.vocab = sorted(list(selfies_vocab.keys()))
            self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
            self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        # Add "." token in the front
        self.vocab = ["."] + self.vocab
        # Add special tokens in the front
        self.vocab = ["<pad>", "<unk>", "<eos>"] + self.vocab
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def tokenize(self, selfies_str):
        return sf.split_selfies(selfies_str)

    def encode(self, selfies_str):
        return [self.token_to_id[tok] for tok in self.tokenize(selfies_str)] + [self.token_to_id["<eos>"]]

    def decode(self, token_ids):
        return sf.decoder("".join([self.id_to_token[i] for i in token_ids]))

    def vocab_size(self):
        return len(self.vocab)


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=128, max_seq_len=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def prepare_data(self):
        pass

    def tokenize_selfies(self, selfies_vocab, selfies):
        selfies = sf.split_selfies(selfies)
        selfies = [selfies_vocab[token] for token in selfies]
        tokens = torch.tensor(selfies, dtype=torch.long)

        return tokens

    def setup(self, stage=None):
        # Load Analysis/dump_data/pubchem/sparse_based_sampled.csv
        df = pd.read_csv("Analysis/dump_data/pubchem/sparse_based_sampled.csv")

        # Get column names which start with "fr_"
        fg_columns = [col for col in df.columns if col.startswith("fr_")]
        total_skipped = 0

        graphs_saved = torch.load("Analysis/dump_data/pubchem/graphs_5M.pt")
        graphs = []
        for graph in tqdm(graphs_saved, total=len(graphs_saved), desc="Processing graphs"):
            seq_len = graph['x'].shape[0] + graph['edge_index'].shape[1] + 1
            if seq_len > self.max_seq_len:
                # print(f"Skipping {smiles} due to length {seq_len} > {self.max_seq_len}")
                total_skipped += 1
                continue
            graph['selfies_tokens'] = graph['selfies_tokens'].tolist()[:self.max_seq_len]
            graphs.append(graph)
        del graphs_saved
        # graphs = []
        # for i, row in tqdm(enumerate(df.iterrows()), total=len(df)):
        #     smiles = row[1]['SMILES']
        #     try:
        #         graph_dict = smiles2graph(smiles)
        #         seq_len = graph_dict['node_feat'].shape[0] + graph_dict['edge_index'].shape[1] + 1
        #         if seq_len > self.max_seq_len:
        #             # print(f"Skipping {smiles} due to length {seq_len} > {self.max_seq_len}")
        #             total_skipped += 1
        #             continue
        #         selfies = sf.encoder(smiles)
        #     except:
        #         # print(f"Error in smiles: {smiles}")
        #         continue
        #     fg_vector = torch.tensor(row[1][fg_columns].values.astype(float), dtype=torch.float32)
        #     selfies_tokens = self.tokenizer.encode(selfies)[:self.max_seq_len]
        #     # selfies_tokens = [100, 200, 2]
        #     graph_data = Data(
        #         x=torch.tensor(graph_dict['node_feat'], dtype=torch.long),
        #         edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
        #         edge_attr=torch.tensor(graph_dict['edge_feat'], dtype=torch.long),
        #         fg_vector=fg_vector,
        #         selfies_tokens=selfies_tokens,
        #     )
        #     graphs.append(graph_data)
        #     if i == 100000:
        #         break
        print(f"Total skipped: {total_skipped}")
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
    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = kwargs['max_seq_len']
        self.gnn_output_dim = kwargs['gnn_output_dim']
        self.backbone = kwargs['backbone']
        self.activate_GNN = kwargs['activate_GNN']
        self.activate_MLP = kwargs['activate_MLP']
        self.lr = kwargs['lr']

        self.n_funcgroup = len(RDKIT_PROPS)

        ################## Initialize GNN ##################
        if self.backbone == "MoleculeSTM->MLP":
            self.gnn = GNN_MoleculeSTM(
                num_layer=5,
                emb_dim=self.gnn_output_dim,
                gnn_type="gin",
                drop_ratio=0.0,
                JK="last",
                # args=args,
            )
        elif self.backbone == "TokenGT->MLP":
            # self.gnn = TokenGT(
            #     input_feat_dim=9,
            #     hidden_dim=self.gnn_output_dim,
            #     num_layers=5,
            #     num_heads=8,
            #     method="laplacian",
            #     d_p=64,
            #     d_e=64,
            #     use_graph_token=True
            # )
            self.gnn = BERTTokenGT(
                input_feat_dim=9,
                hidden_dim=self.gnn_output_dim,
                num_layers=5,
                num_heads=8,
                method="laplacian",
                d_p=64,
                d_e=64,
                use_graph_token=True,
                max_position_embeddings=1024
            )
        else:
            raise ValueError(f"Invalid backbone type: {self.backbone}")
        
        self.ln_graph = LayerNorm(self.gnn_output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.gnn_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_funcgroup)
        )

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size(),
            n_positions=self.max_seq_len,
            n_embd=self.gnn_output_dim,
            n_layer=6,
            n_head=8,
            eos_token_id=0,
            pad_token_id=1,
        )
        self.selfies_decoder = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer

    def forward(self, x, edge_index, edge_attr, batch, selfies_tokens):
        output, _ = self.gnn(x, edge_index, edge_attr, batch)
        graph_embedding = output[:, :1, :]
        # node_embedding = output[:, 1:, :]
        graph_embedding = self.ln_graph(graph_embedding)
        pred_funcgroup = self.mlp(graph_embedding.squeeze(1))

        # Token padding
        pad_id = 1  # Ensure it's defined
        input_ids_tensor = [torch.tensor(ids, dtype=torch.long) for ids in selfies_tokens]
        padded_input_ids = pad_sequence(input_ids_tensor, batch_first=True, padding_value=pad_id).to(graph_embedding.device)
        
        # Shift for decoder input and labels
        decoder_input_ids = padded_input_ids[:, :-1]  # input to decoder
        labels = padded_input_ids.clone()
        attention_mask = (decoder_input_ids != pad_id).long()

        # Embed tokens
        token_embeds = self.selfies_decoder.transformer.wte(decoder_input_ids)
        inputs_embeds = torch.cat([graph_embedding, token_embeds], dim=1)

        # Adjust attention mask
        extended_attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),  # for graph_embedding
            attention_mask
        ], dim=1)

        labels[labels == pad_id] = -100  # Ignore padding tokens in loss calculation

        # cut off by max_seq_len
        inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
        extended_attention_mask = extended_attention_mask[:, :self.max_seq_len]
        labels = labels[:, :self.max_seq_len]
        # Decode
        pred_selfies = self.selfies_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels
        )

        return pred_funcgroup, pred_selfies

    def training_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('train/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('train/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        total_loss = loss_funcgroup + pred_selfies.loss
        self.log('train/total_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('validation/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('validation/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('validation/total_loss', loss_funcgroup + pred_selfies.loss, prog_bar=True, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return_dict = {
            "pred_funcgroup": pred_funcgroup.detach(),
            "y_funcgroup": y.detach(),
            "loss_funcgroup": loss_funcgroup.detach(),
            "loss_selfies": pred_selfies.loss.detach(),
        }
        return return_dict

    def test_step(self, batch, batch_idx):
        pred_funcgroup, pred_selfies = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.selfies_tokens)
        pred_funcgroup = torch.sigmoid(pred_funcgroup)
        task_size = pred_funcgroup.size(1)
        y = batch.fg_vector.view(-1, task_size).float()
        loss_funcgroup = F.binary_cross_entropy_with_logits(pred_funcgroup, y.float())
        self.log('test/loss_funcgroup', loss_funcgroup, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))
        self.log('test/loss_selfies', pred_selfies.loss, prog_bar=False, sync_dist=True, batch_size=batch.fg_vector.size(0))

        return_dict = {
            "pred_funcgroup": pred_funcgroup.detach(),
            "y_funcgroup": y.detach(),
            "loss_funcgroup": loss_funcgroup.detach(),
            "loss_selfies": pred_selfies.loss.detach(),
        }
        return return_dict

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


class ValidationCallback(Callback):
    def __init__(self):
        self.val_outs = []
        self.val_labels = []
        self.val_loss = []
        self.test_outs = []
        self.test_labels = []
        self.test_loss = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_outs.append(outputs['pred_funcgroup'].detach().cpu())
        self.val_labels.append(outputs['y_funcgroup'].detach().cpu())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_outs.append(outputs['pred_funcgroup'].detach().cpu())
        self.test_labels.append(outputs['y_funcgroup'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.val_outs, dim=0)
        ys = torch.cat(self.val_labels, dim=0)
        preds = pl_module.all_gather(preds)
        ys = pl_module.all_gather(ys)
        preds = preds.view(-1, preds.size(-1))
        ys = ys.view(-1, ys.size(-1))
        if trainer.is_global_zero:
            for i in range(preds.size(1)):
                pred = preds[:, i]
                y = ys[:, i]
                acc = (pred > 0.5).float() == y
                acc = acc.float().mean().item()
                auroc = roc_auc_score(y.cpu().float().numpy(), pred.cpu().float().numpy())
                trainer.logger.log_metrics({f'validation_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'validation_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'validation_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
            # loss = F.binary_cross_entropy_with_logits(preds, ys.float())
            # trainer.logger.log_metrics({'validation/total_loss': loss}, step=trainer.current_epoch)
            trainer.logger.log_metrics({'validation/instance_size': preds.size(0)}, step=trainer.current_epoch)
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
            for i in range(preds.size(1)):
                pred = preds[:, i]
                y = ys[:, i]
                acc = (pred > 0.5).float() == y
                acc = acc.float().mean().item()
                auroc = roc_auc_score(y.cpu().float().numpy(), pred.cpu().float().numpy())
                trainer.logger.log_metrics({f'test_acc/{func_idx2name[i]}': acc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'test_auroc/{func_idx2name[i]}': auroc}, step=trainer.current_epoch)
                trainer.logger.log_metrics({f'test_num/{func_idx2name[i]}': y.sum().item()}, step=trainer.current_epoch)
            # loss = F.binary_cross_entropy_with_logits(preds, ys.float())
            # trainer.logger.log_metrics({'test/total_loss': loss}, step=trainer.current_epoch)
            trainer.logger.log_metrics({'test/instance_size': preds.size(0)}, step=trainer.current_epoch)
        self.test_outs = []
        self.test_labels = []


# 학습 실행 코드
if __name__ == '__main__':

    qm9_tasks = list(qm9_name2idx.keys())
    model_types = list(model_settings.keys())

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/total_loss',
        mode='min',
        save_top_k=1,
        filename='best-model'
    )
    validation_callback = ValidationCallback()
    tokenizer = SelfiesTokenizer()

    for model_type in model_types:
        data_module = DataModule(tokenizer, batch_size=64, max_seq_len=model_settings[model_type]['max_seq_len'])
        pl_model = TaskModel(tokenizer=tokenizer, **model_settings[model_type])
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator='auto',
            devices=[0,1,2,3,4,5,6,7],
            # devices=[3],
            default_root_dir=f'./gnn_ablation/custom_gnn_train_5M/{model_type}',
            # callbacks=[checkpoint_callback],
            callbacks=[checkpoint_callback, validation_callback],
            strategy='ddp_find_unused_parameters_true',
            precision="bf16-mixed",
            gradient_clip_val=0.5,
            log_every_n_steps=600,
            val_check_interval=0.25,
            num_sanity_val_steps=0
        )
        # Load gnn_ablation/custom_gnn_train/MoleculeSTM/lightning_logs/version_2/checkpoints/best-model.ckpt
        # checkpoint_path = 'gnn_ablation/custom_gnn_train/MoleculeSTM/lightning_logs/version_2/checkpoints/best-model.ckpt'
        # print(f"Loading checkpoint from {checkpoint_path}")
        # trainer.fit(pl_model, datamodule=data_module, ckpt_path=checkpoint_path)
        trainer.fit(pl_model, datamodule=data_module)
        
        
        
        # best_model_path = checkpoint_callback.best_model_path
        # trainer.test(ckpt_path=best_model_path, datamodule=data_module)
        # if trainer.global_rank == 0:
        #     if os.path.exists(checkpoint_callback.best_model_path):
        #         os.remove(checkpoint_callback.best_model_path)
        #         print(f"Deleted checkpoint file: {checkpoint_callback.best_model_path}")
        