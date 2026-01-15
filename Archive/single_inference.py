import torch
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
import os
import sys
from datasets import load_from_disk  # [추가] 데이터셋 로드용

# [추가] 태스크 이름 확인을 위해 import
from data_utils import id2task 

# ==============================================================================
# 1. Flash Attention Patch
# ==============================================================================
try:
    import flash_attn.models.bert
    OriginalBertConfig = flash_attn.models.bert.BertConfig
    class PatchedBertConfig(OriginalBertConfig):
        def __init__(self, *args, **kwargs):
            kwargs['use_flash_attn'] = False 
            super().__init__(*args, **kwargs)
            self.use_flash_attn = False
    flash_attn.models.bert.BertConfig = PatchedBertConfig
    print("[System] Patched BertConfig to disable Flash Attention.")
except:
    pass

from model.blip2_stage3 import Blip2Stage3
from data_module import Stage3DM

# ==============================================================================
# 2. Config Helper & Data Verification
# ==============================================================================
def flatten_dictconfig(config: DictConfig) -> DictConfig:
    items = []
    for k, v in config.items():
        if isinstance(v, DictConfig):
            for kk, vv in v.items():
                items.append((f"{kk}", vv))
        else:
            items.append((k, v))
    return OmegaConf.create(dict(items))

def verify_dataset_integrity(dataset_path):
    """
    지정된 경로의 데이터셋을 로드하여 키 값과 데이터 상태를 검증하는 함수
    """
    print(f"\n" + "="*30 + " [VERIFICATION START] " + "="*30)
    print(f"Target Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path does not exist!")
        return False

    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset Loaded Successfully.")
        print(f"   - Total Rows: {len(dataset)}")
        print(f"   - Column Names: {dataset.column_names}")
        
        # 첫 번째 샘플 검사
        sample = dataset[0]
        print(f"\n[Sample #0 Inspection]")
        
        # 1. Task 확인
        task_info = sample.get('task', 'Unknown')
        print(f"   - Task: {task_info}")
        if 'task_subtask_pair' in sample:
            print(f"   - Task Subtask Pair: {sample['task_subtask_pair']}")

        # 2. Graph Data (x) 확인
        if 'x' in sample and sample['x'] is not None:
            # 리스트인지 텐서인지 확인
            x_len = len(sample['x'])
            print(f"   - Graph Node Feature (x): Found (Length={x_len})")
        else:
            print(f"   - ⚠️ Graph Node Feature (x): MISSING or None")

        # 3. Additional Graph Data (additional_x) 확인
        if 'additional_x' in sample and sample['additional_x'] is not None:
            add_x_len = len(sample['additional_x'])
            print(f"   - Additional Graph (additional_x): Found (Length={add_x_len})")
        else:
            print(f"   - Additional Graph (additional_x): None (Normal for Single Task)")

        # 4. Text & Token Check
        mol_token = "<mol>"
        check_keys = ['input_mol_string', 'prompt_text', 'instruction']
        
        for key in check_keys:
            if key in sample:
                text_content = sample[key]
                token_count = text_content.count(mol_token)
                print(f"   - [{key}] length: {len(text_content)}, <mol> count: {token_count}")
                # 내용 일부 출력
                # print(f"     Snippet: {text_content[:100]} ...")
            else:
                print(f"   - [{key}]: Not Found")

        print("="*30 + " [VERIFICATION END] " + "="*30 + "\n")
        return True

    except Exception as e:
        print(f"❌ Exception during dataset verification: {e}")
        return False

def get_config_and_setup():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="test_CHJ.yaml")
    cfg = flatten_dictconfig(cfg)
    OmegaConf.set_struct(cfg, False)

    root_dir = "Mol-LLM_Custom"
    cfg.ckpt_path = os.path.join(root_dir, "checkpoint/Custom/mol-llm.ckpt")
    cfg.selfies_token_path = os.path.join(root_dir, "model/selfies_dict.txt")
    cfg.filename = "HJChoi"
    cfg.data_tag = "3.3M_0415" 
    cfg.mode = "test"
    
    if hasattr(cfg, "gine"): cfg.gine.graph_encoder_ckpt = cfg.ckpt_path
    if hasattr(cfg, "tokengt"): cfg.tokengt.graph_encoder_ckpt = cfg.ckpt_path
    return cfg

# ==============================================================================
# 3. Main Logic
# ==============================================================================
def main():
    # [STEP 0] 데이터셋 무결성 검증 (요청하신 부분)
    target_dataset_path = "Mol-LLM_Custom/dataset/real_train/mistralai-Mistral-7B-Instruct-v0.3_string+graph_q32_test_3.3M_0415"
    verify_dataset_integrity(target_dataset_path)

    # [STEP 1] 설정 및 모델 준비
    cfg = get_config_and_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    print(f"Loading weights from {cfg.ckpt_path}...")
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    if "epoch" in checkpoint: cfg.current_epoch = checkpoint["epoch"]

    print("Initializing Model...")
    model = Blip2Stage3(cfg)
    tokenizer = model.blip2model.llm_tokenizer
    
    mol_token_str = "<mol>" 
    mol_token_id = tokenizer.convert_tokens_to_ids(mol_token_str)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    if hasattr(model, 'blip2model'): model.blip2model.to(device)
    model.float() 
    model.eval()

    # [STEP 2] 데이터 로더 준비
    dm = Stage3DM(mode=cfg.mode, num_workers=0, tokenizer=tokenizer, args=cfg)
    dm.setup("test")
    test_loader = dm.test_dataloader()
    
    try:
        batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Empty DataLoader")
        return

    # [STEP 3] Inference 전처리 및 실행
    
    # 1. Task 확인
    current_task_name = "Unknown"
    if "tasks" in batch:
        task_id = batch["tasks"][0].item() 
        current_task_name = id2task(task_id)
        print(f"[Run] Current Task: {current_task_name} (ID: {task_id})")
    
    # 2. Graph Type 결정 (Multi vs Single)
    MULTI_GRAPH_TASKS = ["reagent_prediction", "reaction_prediction", "retrosynthesis"]
    is_multi_graph = any(t in current_task_name for t in MULTI_GRAPH_TASKS)
    
    graphs = batch.get("graphs")
    additional_graphs = batch.get("additional_graphs")

    if is_multi_graph:
        num_graphs_input = 2
        print(f"[Logic] Multi-Graph Task. Keeping additional_graphs.")
    else:
        num_graphs_input = 1
        # ★ Single Task면 더미 데이터 제거
        additional_graphs = None 
        print(f"[Logic] Single-Graph Task. Ignoring additional_graphs.")

    # 3. Text Token 검증 및 수정
    input_ids = batch["input_ids"]
    current_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    current_mol_count = current_text.count(mol_token_str)
    expected_mol_count = num_graphs_input * 32
    
    print(f"[Token Check] Found: {current_mol_count}, Expected: {expected_mol_count}")

    if current_mol_count != expected_mol_count:
        print(f"⚠️ [FIXING] Token mismatch detected!")
        if current_mol_count == 0:
            mol_string = mol_token_str * expected_mol_count
            new_text = current_text + f" <GRAPH>{mol_string}</GRAPH>"
        elif current_mol_count == 32 and expected_mol_count == 64:
            new_text = current_text.replace("</GRAPH>", f"{mol_token_str * 32}</GRAPH>")
        elif current_mol_count > expected_mol_count:
            print("   -> Removing excess tokens...")
            import re
            correct_graph_part = "<GRAPH>" + (mol_token_str * expected_mol_count) + "</GRAPH>"
            new_text = re.sub(r"<GRAPH>.*?</GRAPH>", correct_graph_part, current_text)
        else:
            new_text = current_text 

        # Re-tokenize
        inputs = tokenizer([new_text], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        is_mol_token = (input_ids == mol_token_id)
    else:
        attention_mask = batch["attention_mask"]
        if "is_mol_token" in batch:
            is_mol_token = batch["is_mol_token"]
        else:
            is_mol_token = (input_ids == mol_token_id)

    # 4. Generate
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    is_mol_token = is_mol_token.to(device)
    
    if graphs is not None: graphs = graphs.to(device)
    if additional_graphs is not None: additional_graphs = additional_graphs.to(device)
    
    graph_inputs = (graphs, additional_graphs)

    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.blip2model.generate(
            graphs=graph_inputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_mol_token=is_mol_token,
            num_beams=5,
            max_length=256,
            min_length=1,
            do_sample=False,
            repetition_penalty=1.0
        )

    # 5. Output Decoding (Safe Access)
    prediction_text = ""
    if hasattr(outputs, "predictions"):
        prediction_text = outputs.predictions[0]
    elif isinstance(outputs, dict) and "predictions" in outputs:
        prediction_text = outputs["predictions"][0]
    elif isinstance(outputs, list):
        prediction_text = outputs[0]
    else:
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        if hasattr(sequences, "cpu"):
             prediction_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        else:
             prediction_text = str(outputs)

    clean_prediction = prediction_text.replace(tokenizer.pad_token, "").replace("</s>", "").strip()
    
    print("\n" + "="*50)
    print(f"[Task]: {current_task_name}")
    print(f"[Prediction]: {clean_prediction}")
    print("="*50)

if __name__ == "__main__":
    main()