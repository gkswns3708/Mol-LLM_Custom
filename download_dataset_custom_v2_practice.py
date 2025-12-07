# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import re
import traceback

import deepchem as dc
import numpy as np
import selfies as sf
import torch
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import time
from typing import List, Dict, Any
import pandas as pd
import datasets
import yaml
import multiprocessing as mp

import instructions_smol
import model.added_tokens as added_tokens
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)


# util function
def from_dict(d):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**d)

def get_task_subtask_info(target_benchmarks):
    task_subtask_dict = {}
    for task in target_benchmarks:
        if isinstance(task, str):
            task_subtask_dict[task] = [0]
        else:
            task_subtask_dict.update(task)
    task_subtask_pairs = [
        (task, subtask)
        for task, subtasks in task_subtask_dict.items()
        for subtask in subtasks
    ]
    return task_subtask_dict, task_subtask_pairs

def get_dataset(task_name, raw_data_root):
    if "smol" in task_name:
        script_path = "./InstructGraph.py"
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=True,
            insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)  # remove smol- from smol-<task_name>
        
        train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
        valid_dataset = smol_dataset["validation"].filter(lambda x: x["task"] == _task)
        test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
        tasks = [task_name]
    elif task_name in [
        "toxcast", 
        "tox21",
        "hopv"
    ]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}")
    elif task_name in ['qm9_additional_label']:
        loading_fc = dc.molnet.load_qm9
    elif task_name == "bace":
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_test.csv"))
        tasks = [task_name]
    elif "chebi-20" in task_name:
        # load data from csv
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_valid.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_test.csv"))
        tasks = [task_name]
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs/download/")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--train_procs", type=int, default=32)
    parser.add_argument("--test_procs", type=int, default=32)
    args = parser.parse_args()
    
    #!  수정이 없다면 ./config/download/default.yaml을 load해서 download 관련된 내용을 진행함.
    arg_path = os.path.join(args.config_dir, args.config) + ".yaml" 
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = from_dict(cfg)
    
    raw_data_root = cfg.raw_data_root
    if not os.path.exists(raw_data_root):
        os.makedirs(raw_data_root)
    
    # 'target_benchmarks'를 읽어서 [('task 명' , 0) 형태의 list를 반환함.
    # {'forward_reaction_prediction': [0], 'qm9_homo': [0], 'qm9_homo_lumo_gap': [0], 'qm9_lumo': [0]}  - task_subtask_dict
    # [('forward_reaction_prediction', 0), ('qm9_homo', 0), ('qm9_homo_lumo_gap', 0), ('qm9_lumo', 0)]  - task_subtask_pairs
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)

    # download하고 싶은 모든 task를 append함.
    downloading_task_subtask_pairs = []
    for task_subtask_pair in task_subtask_pairs:
        downloading_task_subtask_pairs.append(task_subtask_pair)
        
    for task_subtask_pair in tqdm(downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"):
        task_name, subtask_idx = task_subtask_pair
        try:
            new_dataset = get_dataset(task_name=task_name, raw_data_root=raw_data_rpoot)
        except Exception as e:
            print(f"[Error] Failed to get dataset for {task_name}: {e}")
            # 단순 error message외에, 에러가 발생하기까지의 모든 과정(Stack Trace)을 출력해라.
            traceback.print_exc()
            continue
        subtasks = new_dataset[0]
        print(subtasks, "- subtasks")
        