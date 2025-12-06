import torch
import torch.nn as nn

# from transformers import BertTokenizer
from model.gin_model import GNN, GNN_MoleculeSTM
from model.tokenGT import BERTTokenGT



class GINE_TokenGT(nn.Module):
    def __init__(self, args):
        super(GINE_TokenGT, self).__init__()
        self.graph_encoder_gine = GNN_MoleculeSTM(
            num_layer=args.gine.gin_num_layers,
            emb_dim=args.gine.gnn_hidden_dim,
            gnn_type="gin",
            drop_ratio=args.gine.drop_ratio,
            JK=args.gine.gnn_jk,
            args=args,
        )
        self.graph_encoder_tokengt = BERTTokenGT(
            input_feat_dim=args.tokengt.input_feat_dim,
            hidden_dim=args.tokengt.gnn_hidden_dim,
            num_layers=args.tokengt.num_layers,
            num_heads=args.tokengt.num_heads,
            method=args.tokengt.method,
            d_p=args.tokengt.d_p,
            d_e=args.tokengt.d_e,
            use_graph_token=args.tokengt.use_graph_token,
            max_position_embeddings=args.tokengt.max_position_embeddings
        )
        ##### load pretrained GINE #####
        print(args.gine.graph_encoder_ckpt, "-args.gine.graph_encoder_ckpt")
        ckpt = torch.load(
            args.gine.graph_encoder_ckpt, map_location=torch.device("cpu"), weights_only=False
        )
        renamed_state_dict = {}
        for param, value in ckpt['state_dict'].items():
            if param.startswith('blip2model.graph_encoder.graph_encoder_gine'):
                renamed_state_dict[param.replace("blip2model.graph_encoder.graph_encoder_gine.", "")] = value
        self.graph_encoder_gine.load_state_dict(renamed_state_dict, strict=True)
        print(f"load graph encoder from {args.gine.graph_encoder_ckpt}")

        ##### load pretrained TokenGT #####
        ckpt = torch.load(
            args.tokengt.graph_encoder_ckpt, map_location=torch.device("cpu"), weights_only=False
        )
        renamed_state_dict = {}
        for param, value in ckpt['state_dict'].items():
            if param.startswith('blip2model.graph_encoder.graph_encoder_tokengt'):
                renamed_state_dict[param.replace("blip2model.graph_encoder.graph_encoder_tokengt.", "")] = value
        self.graph_encoder_tokengt.load_state_dict(renamed_state_dict, strict=True)
        print(f"load graph encoder from {args.tokengt.graph_encoder_ckpt}")

        self.layer_norm_gine = nn.LayerNorm(args.gine.gnn_hidden_dim)
        self.layer_norm_tokengt = nn.LayerNorm(args.tokengt.gnn_hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        gine_output, gine_mask = self.graph_encoder_gine(x, edge_index, edge_attr, batch)
        tokengt_output, tokengt_mask = self.graph_encoder_tokengt(x, edge_index, edge_attr, batch)

        # apply layer normalization
        gine_output = self.layer_norm_gine(gine_output)
        tokengt_output = self.layer_norm_tokengt(tokengt_output)

        # concatenate the outputs
        output = torch.concat((gine_output, tokengt_output), dim=1)
        mask = torch.concat((gine_mask, tokengt_mask), dim=1)

        return output, mask
