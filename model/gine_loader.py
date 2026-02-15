import torch
import torch.nn as nn
from model.gin_model import GNN_MoleculeSTM
from model.tokenGT import BERTTokenGT


def load_gine_tokengt(
    args,
    gine_ckpt_path="/app/Mol-LLM_Custom/checkpoint/GraphEncoder/epochepoch=45-stepstep=880440_MoleculeSTM.ckpt",
    tokengt_ckpt_path="/app/Mol-LLM_Custom/checkpoint/GraphEncoder/epochepoch=49-stepstep=476106_tokengt_best.ckpt",
    device="cpu",
    debug=False,
):
    """
    GINE과 TokenGT checkpoint를 로드하여 GINE_TokenGT 모델을 생성하는 함수

    Args:
        args: Configuration 객체 (gine, tokengt 파라미터 포함)
        gine_ckpt_path: GINE MoleculeSTM checkpoint 경로
        tokengt_ckpt_path: TokenGT checkpoint 경로
        device: 모델을 로드할 device ('cpu' 또는 'cuda')
        debug: 디버그 정보 출력 여부

    Returns:
        model: 두 checkpoint가 로드된 GINE_TokenGT 모델 (nn.Module)
    """

    # 1. GINE encoder 생성
    if debug:
        print(f"[1/4] Creating GINE encoder...")

    graph_encoder_gine = GNN_MoleculeSTM(
        num_layer=args.gine.gin_num_layers,
        emb_dim=args.gine.gnn_hidden_dim,
        gnn_type="gin",
        drop_ratio=args.gine.drop_ratio,
        JK=args.gine.gnn_jk,
        args=args,
    )

    if debug:
        print(f"  ✓ GINE encoder created")

    # 2. TokenGT encoder 생성
    if debug:
        print(f"[2/4] Creating TokenGT encoder...")

    graph_encoder_tokengt = BERTTokenGT(
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

    if debug:
        print(f"  ✓ TokenGT encoder created")

    # 3. GINE checkpoint 로드
    if debug:
        print(f"[3/4] Loading GINE checkpoint: {gine_ckpt_path}")

    gine_ckpt = torch.load(gine_ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    gine_state_dict = {}

    for param_name, param_value in gine_ckpt['state_dict'].items():
        if param_name.startswith('gnn.'):
            # "gnn." 프리픽스 제거 (독립적인 GNN 모델 checkpoint)
            new_param_name = param_name.replace("gnn.", "")
            gine_state_dict[new_param_name] = param_value

    graph_encoder_gine.load_state_dict(gine_state_dict, strict=True)

    if debug:
        print(f"  ✓ GINE checkpoint loaded ({len(gine_state_dict)} parameters)")

    # 4. TokenGT checkpoint 로드
    if debug:
        print(f"[4/4] Loading TokenGT checkpoint: {tokengt_ckpt_path}")

    tokengt_ckpt = torch.load(tokengt_ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    tokengt_state_dict = {}

    for param_name, param_value in tokengt_ckpt['state_dict'].items():
        if param_name.startswith('gnn.'):
            # "gnn." 프리픽스 제거 (독립적인 TokenGT 모델 checkpoint)
            new_param_name = param_name.replace("gnn.", "")
            tokengt_state_dict[new_param_name] = param_value

    graph_encoder_tokengt.load_state_dict(tokengt_state_dict, strict=True)

    if debug:
        print(f"  ✓ TokenGT checkpoint loaded ({len(tokengt_state_dict)} parameters)")

    # 5. GINE_TokenGT 모델 조립
    model = nn.Module()
    model.graph_encoder_gine = graph_encoder_gine
    model.graph_encoder_tokengt = graph_encoder_tokengt
    model.layer_norm_gine = nn.LayerNorm(args.gine.gnn_hidden_dim)
    model.layer_norm_tokengt = nn.LayerNorm(args.tokengt.gnn_hidden_dim)

    # Forward 메서드 추가
    def forward(x, edge_index, edge_attr, batch):
        gine_output, gine_mask = model.graph_encoder_gine(x, edge_index, edge_attr, batch)
        tokengt_output, tokengt_mask = model.graph_encoder_tokengt(x, edge_index, edge_attr, batch)

        gine_output = model.layer_norm_gine(gine_output)
        tokengt_output = model.layer_norm_tokengt(tokengt_output)

        output = torch.concat((gine_output, tokengt_output), dim=1)
        mask = torch.concat((gine_mask, tokengt_mask), dim=1)

        return output, mask

    model.forward = forward

    # Device 이동
    model = model.to(device)

    if debug:
        print(f"\n[✓] Complete! Model loaded on {device}")
        print(f"    GINE hidden dim: {args.gine.gnn_hidden_dim}")
        print(f"    TokenGT hidden dim: {args.tokengt.gnn_hidden_dim}")
        print(f"    Output dim: {args.gine.gnn_hidden_dim + args.tokengt.gnn_hidden_dim}")

    return model
