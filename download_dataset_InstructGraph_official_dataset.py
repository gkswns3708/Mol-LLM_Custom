# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import re
import traceback
import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import yaml
import deepchem as dc
import selfies as sf
import torch
import datasets
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

import instructions_smol
import model.added_tokens as added_tokens
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)

# -----------------------------------------------------------------------------
# [Helper Functions]
# -----------------------------------------------------------------------------

def wrap_label(label, task):
    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = added_tokens.BOOL
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = added_tokens.FLOAT
    elif task in ["smol-name_conversion-s2f", "smol-name_conversion-i2f"]:
        label_tokens = added_tokens.MOLFORMULA
    elif task == "smol-name_conversion-s2i":
        label_tokens = added_tokens.IUPAC
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = added_tokens.DESCRIPTION
    elif task == "smol-name_conversion-i2s":
        label_tokens = added_tokens.SELFIES
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = added_tokens.SELFIES
    else:
        raise NotImplementedError(f"Task {task} is not implemented in wrap_label")

    if task in CLASSIFICATION_BENCHMARKS:
        if isinstance(label, str):
            if "true" in label.lower() or "yes" in label.lower():
                label = "True"
            elif "false" in label.lower() or "no" in label.lower():
                label = "False"
            else:
                label = "False"
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        elif isinstance(label, list):
            label_language = ", ".join(label)
            label_boolean = "True" * len(label)
            label = label_language + label_tokens[0] + " " + label_boolean + " " + label_tokens[1]
        else:
            label = "True" if label else "False"
            label = label_tokens[0] + " " + label + " " + label_tokens[1]
        return label
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, float) or isinstance(label, int):
            label = "{:.10f}".format(float(label))
        else:
            try:
                label = format(float(label), ".10f")
            except:
                label = str(label)

        if "-" not in label and "+" not in label:
            label = "+" + label
        label = label[:7]
        converted_label = "".join([f"<|{char}|>" for char in label])
        return label_tokens[0] + " " + converted_label + " " + label_tokens[1]
    
    elif task in REACTION_BENCHMARKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS + ["smol-name_conversion-i2s"]:
        return label_tokens[0] + str(label) + label_tokens[1]
    else:
        return str(label)


def smiles2data(smiles):
    from ogb.utils import smiles2graph
    try:
        graph = smiles2graph(smiles)
        x = torch.from_numpy(graph["node_feat"])
        edge_index = torch.from_numpy(graph["edge_index"])
        edge_attr = torch.from_numpy(graph["edge_feat"])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to graph: {smiles}. Error: {e}")

def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None

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

# -----------------------------------------------------------------------------
# [Dataset Classes]
# -----------------------------------------------------------------------------

class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, task_subtask_pair, subtask_idx=0, prompt=None):
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = str(subtask_idx)
        
        if self.task in CLASSIFICATION_BENCHMARKS:
            self.instruction_templates = getattr(instructions_smol, self.task)
        elif self.task in REGRESSION_BENCHMARKS:
            if self.task in ["qm9_additional_label"]:
                subtask_full_name_dict = {
                    "mu": "dipole_moment", "alpha": "isotropic_polarizability", "r2": "electronic_spatial_extent",
                    "zpve": "zero_point_vibrational_energy", "cv": "heat_capacity_298K",
                    "u298": "internal_energy_298K", "h298": "enthalpy_298K", "g298": "free_energy_298K",
                }
                task = self.task.replace("_additional_label", "")
                subtask_full_name = subtask_full_name_dict[self.subtask]
                self.instruction_templates = getattr(instructions_smol, f"{task}_{subtask_full_name}")
            else:
                self.instruction_templates = getattr(instructions_smol, f"{self.task}_{self.subtask}".lower())
        else:
            raise NotImplementedError
        self.set_necessary_data()

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        smiles = self.smiles_list[index]
        try:
            input_mol_string = sf.encoder(smiles)
            if input_mol_string is None: input_mol_string = smiles
        except:
             raise ValueError(f"Selfies encoding failed for {smiles}")

        input_mol_string = added_tokens.SELFIES[0] + " " + input_mol_string + " " + added_tokens.SELFIES[1]
        
        if self.subtask_idx == "multi_label_classification":
            label = self.raw_outputs[index]
            label = [self.label_full_name[i] for i in range(len(label)) if label[i]]
            if len(label) == 0:
                label = "No toxicity identified. " + wrap_label("False", self.task)
            else:
                label = wrap_label(label, self.task)
        else:
            label = self.raw_outputs[index]
            label = wrap_label(label, self.task)
        
        graph = [smiles2data(smiles), smiles2data('CC')]
        return graph, label, input_mol_string, instruction

    def set_label_fullname(self):
        self.label_full_name = None
        if self.task == "tox21":
            self.label_full_name = [
                "androgen receptor, full (AR, full)", "androgen receptor, LBD (AR, LBD)", "aryl hydrocarbon receptor (AhR)",
                "aromatase", "estrogen receptor alpha, full (ER, full)", "estrogen receptor alpha, LBD (ER, LBD)",
                "peroxisome proliferator-activated receptor gamma (PPAR-gamma)",
                "nuclear factor (erythroid-derived 2)-like 2/antioxidant responsive element (Nrf2/ARE)",
                "ATPase family AAA domain containing 5 (ATAD5)", "heat shock factor response element (HSE)",
                "mitochondrial membrane potential (MMP)", "tumor suppressor protein p53",
            ]

    def set_necessary_data(self):
        self.raw_inputs = self.data.X
        if self.subtask_idx == "multi_label_classification":
            self.set_label_fullname()
            self.raw_outputs = self.data.y
        else:
            self.raw_outputs = self.data.y[:, self.subtask_idx]
            
        self.smiles_list = []
        for mol in self.raw_inputs:
            try:
                s = Chem.MolToSmiles(mol)
                self.smiles_list.append(s)
            except:
                self.smiles_list.append(None) 

        processed_label_list = []
        processed_input_mol_string_list = []
        processed_graph_list = []
        processed_instruction_list = []
        
        self.count_invalid_smiles = 0
        
        for i in tqdm(range(len(self.raw_inputs)), desc=f"{self.task}-{self.subtask_idx}"):
            if self.smiles_list[i] is None:
                self.count_invalid_smiles += 1
                continue
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_graph_list.append(graph)
                processed_instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.graph_list = processed_graph_list
        self.instruction_list = processed_instruction_list
                
    def __len__(self): return len(self.label_list)
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]

class MolInstructionDatset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        self.set_necesary_data()

    def set_necesary_data(self):
        if self.task == "bace":
            self.input_list = self.data["SELFIES"][:]
            self.label_list = self.data["label"][:]
        else:
            self.input_list = self.data["input"][:]
            self.label_list = self.data["output"][:]
            
        self.instruction_templates = getattr(instructions_smol, self.task)
        
        processed_input_list = [] 
        processed_label_list = []
        processed_input_mol_string_list = []
        processed_graph_list = []
        processed_instruction_list = []
        
        self.count_invalid_smiles = 0
        
        for i in tqdm(range(len(self.input_list)), desc=self.task):
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                processed_input_list.append(self.input_list[i])
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_graph_list.append(graph)
                processed_instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1

        self.input_list = processed_input_list
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.graph_list = processed_graph_list
        self.instruction_list = processed_instruction_list

    def __len__(self): return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        input_ = self.input_list[index]
        label_ = self.label_list[index]
        
        if pd.isna(input_) or pd.isna(label_):
             raise ValueError("Input or Label is NaN")

        input_ = str(input_)
        label_ = str(label_)

        if self.task in REACTION_BENCHMARKS:
            if self.task in ["reagent_prediction"]:
                if ">>" in input_:
                    list_selfies = input_.split(">>")
                    input_mol_string = input_.replace(">>", f"{added_tokens.SELFIES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SELFIES[0]}")
                    list_smiles = [sf.decoder(s.strip()) for s in list_selfies]
                    graph = [smiles2data(s) for s in list_smiles]
                elif "|>>|" in input_:
                    list_selfies = input_.split("|>>|")
                    input_mol_string = input_
                    list_smiles = [sf.decoder(s.strip()) for s in list_selfies]
                    graph = [smiles2data(s) for s in list_smiles]
                else:
                    raise ValueError(f"Invalid reagent format: {input_}")
            else:
                input_mol_string = input_
                smiles = sf.decoder(input_mol_string.strip())
                graph = [smiles2data(smiles), smiles2data('CC')]
        
        elif self.task in CLASSIFICATION_BENCHMARKS:
            input_mol_string = input_
            smiles = sf.decoder(input_mol_string.strip())
            if smiles is None: raise ValueError(f"SELFIES decoding failed")
            graph = [smiles2data(smiles), smiles2data('CC')]
            
        else: 
            input_mol_string = input_
            smiles = sf.decoder(input_mol_string.strip())
            if smiles is None: raise ValueError(f"SELFIES decoding failed")
            graph = [smiles2data(smiles), smiles2data('CC')]
            
        label = wrap_label(label_, self.task)
        input_mol_string = added_tokens.SELFIES[0] + " " + input_mol_string + " " + added_tokens.SELFIES[1]
        return graph, label, input_mol_string, instruction
        
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task, self.instruction_list[index]

# [수정] ChEBI-20 표준 데이터셋 (alxfgh/Text2Mol 등) 호환을 위해 클래스 수정
class ChEBIDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        self.set_necesary_data()

    def set_necesary_data(self):
        # [수정] 데이터셋의 컬럼 목록 확인
        if hasattr(self.data, "column_names"):
            cols = self.data.column_names
        elif hasattr(self.data, "columns"):
            cols = self.data.columns
        else:
            cols = []

        print(f"[Debug] Processing Task: {self.task}")
        print(f"[Debug] Available columns: {cols}")

        # -------------------------------------------------------------------------
        # 1. Description 컬럼 매핑
        # -------------------------------------------------------------------------
        if "description" in cols:
            self.description_list = self.data["description"]
        elif "text" in cols:
            self.description_list = self.data["text"]
        elif "ChEBI descriptions" in cols:  # [수정] 사용자 데이터셋 대응
            self.description_list = self.data["ChEBI descriptions"]
        elif "input" in cols and "text2mol" in self.task:
            self.description_list = self.data["input"]
        elif "output" in cols and "mol2text" in self.task:
            self.description_list = self.data["output"]
        else:
            raise ValueError(f"Column 'description' (or 'text', 'ChEBI descriptions') not found. Available: {cols}")

        # -------------------------------------------------------------------------
        # 2. Molecule (SELFIES or SMILES) 컬럼 매핑
        # -------------------------------------------------------------------------
        self.is_selfies = False # 기본값은 SMILES라고 가정

        if "SELFIES" in cols:
            self.mol_list = self.data["SELFIES"]
            self.is_selfies = True
        elif "smiles" in cols:
            self.mol_list = self.data["smiles"]
        elif "SMILES" in cols:
            self.mol_list = self.data["SMILES"]
        elif "graph" in cols:
            self.mol_list = self.data["graph"]
        elif "input" in cols and "mol2text" in self.task:
            self.mol_list = self.data["input"]
        elif "output" in cols and "text2mol" in self.task:
            self.mol_list = self.data["output"]
        else:
            # [중요] SMILES가 없고 CIDs만 있는 경우에 대한 경고
            if "CIDs" in cols and "SMILES" not in cols:
                raise ValueError(
                    f"Dataset contains 'CIDs' but missing 'SMILES' column. "
                    f"Graph construction requires SMILES strings. Available: {cols}"
                )
            raise ValueError(f"Molecule column (SELFIES or smiles) not found. Available: {cols}")

        # Instruction Template 설정
        if "mol2text" in self.task: 
            self.instruction_templates = getattr(instructions_smol, "molecule_captioning")
        elif "text2mol" in self.task: 
            self.instruction_templates = getattr(instructions_smol, "molecule_generation")
        
        # 데이터 처리 루프
        processed_graph_list = []
        processed_label_list = []
        processed_input_mol_string_list = []
        processed_instruction_list = []
        
        self.count_invalid_smiles = 0
        total_len = len(self.description_list)
        
        for i in tqdm(range(total_len), desc=f"Processing {self.task}"):
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_graph_list.append(graph)
                processed_instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
                if self.count_invalid_smiles < 3: # 로그 폭탄 방지
                    print(f"[Warn] Error at index {i}: {e}")
        
        print(f"[{self.task}] Finished. Valid: {len(processed_label_list)}, Invalid: {self.count_invalid_smiles}")

        self.graph_list = processed_graph_list
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.instruction_list = processed_instruction_list

    def __len__(self): return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        
        desc = self.description_list[index]
        mol_data = self.mol_list[index]
        
        if pd.isna(desc) or pd.isna(mol_data): raise ValueError("NaN data detected")
        
        # 분자 데이터 처리
        if self.is_selfies:
            selfies = mol_data
            smiles = sf.decoder(selfies)
        else:
            smiles = mol_data
            try:
                selfies = sf.encoder(smiles)
                if selfies is None: selfies = smiles 
            except:
                selfies = smiles

        if smiles is None: raise ValueError(f"Selfies decode failed for: {selfies}")

        # 태스크별 Input/Label 설정
        if self.task in TEXT2MOL_BENCHMARKS: # Text -> Mol
            label = selfies
            description = added_tokens.DESCRIPTION[0] + str(desc) + added_tokens.DESCRIPTION[1]
            instruction = instruction.replace("<INPUT>", description)
            graph = [smiles2data("CC"), smiles2data("CC")] # Text2Mol은 입력 그래프 없음(Dummy)
            input_mol_string = "<None>"
            
        elif self.task in MOL2TEXT_BENCHMARKS: # Mol -> Text
            label = str(desc)
            input_mol_string = selfies
            graph = [smiles2data(smiles), smiles2data('CC')]
            
        else:
            label = str(desc)
            input_mol_string = selfies
            graph = [smiles2data(smiles), smiles2data('CC')]

        label = wrap_label(label, self.task)
        input_mol_string = added_tokens.SELFIES[0] + " " + input_mol_string + " " + added_tokens.SELFIES[1]
        
        return graph, label, input_mol_string, instruction
    
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]
class SMolInstructDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        if "/" in task_subtask_pair:
            self.task, self.subtask = task_subtask_pair.split("/", 1)
        else:
            self.task = task_subtask_pair
            self.subtask = "0"
        if "forward_synthesis" in self.task:
            self.instruction_templates = getattr(instructions_smol, "forward_reaction_prediction")
        else:
            self.instruction_templates = getattr(instructions_smol, self.task.replace("smol-", "").replace("-", "_"))
        self.set_necesary_data()

    def set_necesary_data(self):
        processed_graph_list = []
        processed_label_list = []
        processed_input_mol_string_list = []
        processed_instruction_list = []
        
        raw_inputs = self.data["raw_input"][:]
        raw_outputs = self.data["raw_output"][:]
        
        self.count_invalid_smiles = 0
        success_count = 0
        
        print(f"[{self.task}] Start Processing. Raw Data Size: {len(self.data)}")

        for i in tqdm(range(len(self.data)), desc=self.task):
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i, raw_inputs[i], raw_outputs[i])
                processed_graph_list.append(graph)
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_instruction_list.append(instruction)
                success_count += 1
            except Exception as e:
                self.count_invalid_smiles += 1
                if self.count_invalid_smiles <= 5:
                    print(f"\n[FAIL LOG] Task: {self.task} | Index: {i} | Error: {e}")

        print(f"[{self.task}] Finished. Total: {len(self.data)} | Success: {success_count} | Failed: {self.count_invalid_smiles}")
        
        self.graph_list = processed_graph_list
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.instruction_list = processed_instruction_list

    def __len__(self): return len(self.label_list)
    
    def get_necessary_data(self, index, raw_input, raw_output):
        label = raw_output
        
        if self.task in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"]:
            s_token, e_token = added_tokens.IUPAC
            description = s_token + str(raw_input) + e_token
            instruction = np.random.choice(self.instruction_templates).replace("<INPUT>", description)
            
            try:
                dummy_graph = smiles2data("CC")
                graph = [dummy_graph, dummy_graph]
            except:
                x = torch.zeros((1, 1), dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                graph = [data, data]

            input_mol_string = "<None>" 
            label = re.sub(r"\s*;\s*", ".", str(label))

        elif self.task in ["smol-name_conversion-s2i", "smol-name_conversion-s2f"]:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = str(raw_input) 

            try:
                smiles = sf.decoder(input_mol_string)
                if smiles is None: 
                    smiles = "CC" 
                
                graph = [smiles2data(smiles), smiles2data('CC')]
            except:
                graph = [smiles2data("CC"), smiles2data("CC")]

            label = str(label)

        elif self.task in TEXT2MOL_BENCHMARKS:
            s_token, e_token = added_tokens.DESCRIPTION
            description = s_token + raw_input + e_token
            instruction = np.random.choice(self.instruction_templates).replace("<INPUT>", description)
            
            graph = [smiles2data("CC"), smiles2data("CC")]
            input_mol_string = "<None>"
            label = re.sub(r"\s*;\s*", ".", label)
            
        elif self.task in REACTION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = raw_input
            try:
                smiles = sf.decoder(input_mol_string)
                if smiles is None: raise ValueError("Decode failed")
                graph = [smiles2data(smiles), smiles2data('CC')]
            except:
                raise ValueError(f"Reaction decode failed: {input_mol_string}")
            
        elif self.task in ["smol-property_prediction-sider"]:
            instance_input = self.data[index]["input"]
            instruction = re.sub(r"\[.*\]", "<INPUT>", instance_input)
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = [smiles2data(smiles), smiles2data('CC')]
            
        elif self.task in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = [smiles2data(smiles), smiles2data('CC')]
            
        else:
            raise NotImplementedError
            
        label = wrap_label(label, self.task)
        input_mol_string = added_tokens.SELFIES[0] + " " + input_mol_string + " " + added_tokens.SELFIES[1]
        return graph, label, input_mol_string, instruction
        
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]

def get_dataset(task_name, raw_data_root):
    # ---------------------------------------------------------
    # [수정] ChEBI-20: 표준 데이터셋 (alxfgh/Text2Mol) 사용
    # ---------------------------------------------------------
    if "chebi-20" in task_name:
        print(f"[Info] Loading Standard ChEBI-20 from 'liupf/ChEBI-20-MM' for {task_name}")
        try:
            # liupf/ChEBI-20-MM contains 'SMILES', 'caption', 'SELFIES' etc.
            # It follows the standard split sizes.
            dataset = load_dataset("liupf/ChEBI-20-MM")
            
            train_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            
            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset
            
        except Exception as e:
            print(f"[Warning] Failed to load liupf/ChEBI-20-MM: {e}. Fallback to local CSV.")
            # Fallback to local CSV if download fails
            train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_train.csv"))
            valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_valid.csv"))
            test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_chebi20_test.csv"))
            tasks = [task_name]
            return tasks, train_dataset, valid_dataset, test_dataset

    # [수정] SMolInstruct 등 다른 데이터셋은 기존 Hugging Face 로직 유지
    elif "smol" in task_name:
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=True,
            insert_core_tags=False,
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)

        train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
        valid_dataset = smol_dataset["validation"].filter(lambda x: x["task"] == _task)
        test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
        tasks = [task_name]

    elif task_name in ["toxcast", "tox21", "hopv", "qm9_additional_label"]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}" if task_name != "qm9_additional_label" else "load_qm9")
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets_, transformers = loading_fn(featurizer="Raw", splitter="scaffold", save_dir=base_path, data_dir=base_path, reload=True)
        train_dataset, valid_dataset, test_dataset = datasets_

    elif task_name == "bace":
        # BACE도 MolNet 표준 데이터셋으로 로드하는 것이 좋으나, 
        # DeepChem MolNet은 BACE Classification/Regression 스플릿 이슈가 있을 수 있어 
        # 일단 기존 CSV 유지 (사용자 선택에 따라 수정 가능)
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_test.csv"))
        tasks = [task_name]

    elif task_name in ["reagent_prediction", "forward_reaction_prediction", "retrosynthesis", "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]:
        # Mol-Instructions도 Hugging Face Hub에서 로드
        mol_instruction_dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", trust_remote_code=True)
        if "qm9_" in task_name:
            dataset = mol_instruction_dataset["property_prediction"]
            subtask_name = task_name.replace("qm9_", "")
            subtask_instruction_templates = getattr(instructions_smol, "filtering_template_" + subtask_name)
            dataset = dataset.filter(lambda x: x["instruction"] in subtask_instruction_templates)
        else:
            dataset = mol_instruction_dataset[task_name]
        
        train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
        # Mol-Instructions에는 별도 Valid가 없으므로 Train을 쪼개서 사용
        split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
        train_dataset, valid_dataset = split["train"], split["test"]
        test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
        tasks = [task_name]
    else:
        raise NotImplementedError(f"Task {task_name} not supported in get_dataset")
    
    return tasks, train_dataset, valid_dataset, test_dataset


def prepare_data_instance(
        data_instance,
        system_prompt,
        llm_model_name="mistral",  
        mol_token="<mol>",
        num_query_tokens=32,
):
    input_mol_string = data_instance["input_mol_string"]    
    input_mol_string = re.sub(r"<SELFIES>\s*", "<SELFIES> ", input_mol_string)
    input_mol_string = re.sub(r"\s*</SELFIES>", " </SELFIES>", input_mol_string)
    
    input_prompt = data_instance["instruction"]

    if "<INPUT>" in input_prompt:
        input_prompt = input_prompt.replace("<INPUT>", input_mol_string)

    graph_sequence = "<GRAPH> " + mol_token * num_query_tokens + " </GRAPH>"
    input_prompt += graph_sequence

    is_llada = "llada" in llm_model_name.lower() or "llama-3" in llm_model_name.lower()
    
    if is_llada:
        formatted_prompt_text = (
            "<|startoftext|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + system_prompt + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + input_prompt + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted_target_text = data_instance["label"] + "<|eot_id|>"
        
    else:
        formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [/INST] "
        formatted_target_text = data_instance["label"] + " </s>"

    raw_task = data_instance["task_subtask_pair"]

    if "qm9_additional_label" in raw_task:
        convert_dict = {
            'qm9_additional_label/mu' : "qm9_dipole_moment",
            'qm9_additional_label/alpha' : "qm9_isotropic_polarizability",
            'qm9_additional_label/r2' : "qm9_electronic_spatial_extent",
            'qm9_additional_label/zpve' : "qm9_zero_point_vibrational_energy",
            'qm9_additional_label/cv' : "qm9_heat_capacity_298K",
            'qm9_additional_label/u298' : "qm9_internal_energy_298K",
            'qm9_additional_label/h298' : "qm9_enthalpy_298K",
            'qm9_additional_label/g298' : "qm9_free_energy_298K",
        }
        task = convert_dict.get(raw_task, raw_task)
    elif raw_task.endswith("/0"):
        task = raw_task[:-2]
    elif "/multi_label_classification" in raw_task:
         task = raw_task.split("/")[0]
    else:
        task = raw_task

    data = {
        "task": task,
        "x": data_instance["x"],
        "edge_index": data_instance["edge_index"],
        "edge_attr": data_instance["edge_attr"],
        "additional_x": data_instance["additional_x"],
        "additional_edge_index": data_instance["additional_edge_index"],
        "additional_edge_attr": data_instance["additional_edge_attr"],
        "prompt_text": formatted_prompt_text,
        "target_text": formatted_target_text,
        "input_mol_string": input_mol_string, 
    }
    return data

# -----------------------------------------------------------------------------
# [Main Execution Block]
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs/download/")
    parser.add_argument("--config", type=str, default="default_llada")
    parser.add_argument("--train_procs", type=int, default=64)
    parser.add_argument("--test_procs", type=int, default=64)
    args = parser.parse_args()

    arg_path = os.path.join(args.config_dir, args.config) + ".yaml"
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = from_dict(cfg)

    raw_data_root = cfg.raw_data_root
    if not os.path.exists(raw_data_root):
        os.makedirs(raw_data_root)
    start_time =  time.time()
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)
    
    downloading_task_subtask_pairs = []
    for task_subtask_pair in task_subtask_pairs:
        downloading_task_subtask_pairs.append(task_subtask_pair)

    # --- Dataset Downloading & Saving (First Stage) ---
    for task_subtask_pair in tqdm(downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"):
        task_name = task_subtask_pair[0]
        subtask_idx = task_subtask_pair[1] 

        # [수정] 이미 처리된 데이터셋(Train Split 기준)이 존재하면 건너뛰기
        check_train_path = f"{raw_data_root}/{task_name}_subtask-{subtask_idx}_train"
        if os.path.exists(check_train_path):
            continue
        try:
            new_dataset = get_dataset(task_name=task_name, raw_data_root=raw_data_root)
        except Exception as e:
            print(f"[Error] Failed to get dataset for {task_name}: {e}")
            traceback.print_exc()
            continue

        subtasks = new_dataset[0]
        subtask_idx = task_subtask_pair[1]

        if subtask_idx == "multi_label_classification":
            pair_name = f"{task_name}/multi_label_classification"
        elif task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"]:
            pair_name = f"{task_name}/{subtasks[subtask_idx]}"
        else:
            pair_name = f"{task_name}/0"

        data_split = new_dataset[1:]
        
        def _task_arg_for(dataset_cls):
            return task_name if dataset_cls is MolInstructionDatset else pair_name

        dataset_cls = (
            SMolInstructDataset if "smol" in task_name else
            MoleculeNetDatasetDeepChem if task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"] else
            ChEBIDataset if task_name in ["chebi-20-mol2text", "chebi-20-text2mol"] else
            MolInstructionDatset
        )

        dataset_wrapper = dataset_cls
        
        dataset_splits = {
            "train": data_split[0],
            "val": data_split[1], 
            "test": data_split[2]
        }
        
        process_order = ["train", "val", "test"]
        common_features = None

        for split in process_order:
            if task_name in ["smol-name_conversion-i2s", "smol-name_conversion-s2i"] and split != "train":
                continue

            raw_data = dataset_splits[split]
            
            if raw_data is None or len(raw_data) == 0:
                continue

            ds = dataset_wrapper(data=raw_data, task_subtask_pair=_task_arg_for(dataset_cls), subtask_idx=subtask_idx)
            
            list_dict_data = []
            for i in range(len(ds)):
                try:
                    graph, label, input_mol_string, task_pair_or_name, instruction = ds[i]
                    if hasattr(instruction, "item"): instruction = instruction.item()
                    instruction = str(instruction)

                    if isinstance(graph, list) and len(graph) >= 2:
                        g0, g1 = graph[0], graph[1]
                    elif isinstance(graph, list) and len(graph) == 1:
                        g0 = graph[0]
                        g1 = smiles2data('CC') 
                    else:
                        g0 = graph
                        g1 = smiles2data('CC')

                    list_dict_data.append({
                        "x": g0.x, "edge_index": g0.edge_index, "edge_attr": g0.edge_attr,
                        "label": label, "input_mol_string": input_mol_string,
                        "task_subtask_pair": task_pair_or_name, "instruction": instruction,
                        "additional_x": g1.x, "additional_edge_index": g1.edge_index, "additional_edge_attr": g1.edge_attr,
                    })
                except Exception:
                    continue
            
            save_path = f"{raw_data_root}/{task_name}_subtask-{subtask_idx}_{split}"
            
            if len(list_dict_data) > 0:
                output_dataset = datasets.Dataset.from_list(list_dict_data)
                if common_features is None:
                    common_features = output_dataset.features
                output_dataset.save_to_disk(save_path)
            else:
                if common_features is not None:
                    output_dataset = datasets.Dataset.from_dict({}, features=common_features)
                    output_dataset.save_to_disk(save_path)
                else:
                    pass

    # --- Concatenate & Mapping (Second Stage) ---
    print("\n--- Starting Concatenation and Mapping ---")
    trainsets, testsets = [], []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        
        train_path = f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        try:
            if os.path.exists(train_path):
                ds = datasets.Dataset.load_from_disk(train_path)
                if len(ds) > 0: trainsets.append(ds)
            else:
                pass 
        except Exception as e:
            print(f"Error loading train path {train_path}: {e}")

        test_path = f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"
        try:
            if os.path.exists(test_path):
                ds = datasets.Dataset.load_from_disk(test_path)
                if len(ds) > 0: testsets.append(ds)
        except Exception as e:
            print(f"Error loading test path {test_path}: {e}")
            
        val_path = f"{raw_data_root}/{task}_subtask-{subtask_idx}_val"
        try:
            if os.path.exists(val_path):
                ds = datasets.Dataset.load_from_disk(val_path)
                if len(ds) > 0: testsets.append(ds)
        except Exception as e:
            print(f"Error loading val path {val_path}: {e}")
    
    system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."

    llm_model = cfg.llm_model
    mol_representation = "string+graph"
    num_query_token = 32
    base_model = llm_model.replace("/", "-")
    tags = [base_model, mol_representation]
    if "graph" in mol_representation:
        tags += [f"q{num_query_token}"]
    processed_file_name = "_".join(tags)

    if trainsets:
        concat_trainset = datasets.concatenate_datasets(trainsets)
        print(f"Mapping TRAIN dataset ({len(concat_trainset)} examples)...")
        
        mapped_trainset = concat_trainset.map(
            prepare_data_instance,
            fn_kwargs={
                "system_prompt": system_prompt,
                "llm_model_name": llm_model 
            },
            num_proc=args.train_procs
        )
        save_name_train = f"{raw_data_root}/{processed_file_name}_train_{cfg.data_tag}"
        mapped_trainset.save_to_disk(save_name_train)
        print(f"Saved Final Train Dataset: {save_name_train}")
    else:
        print("Warning: No train datasets were concatenated.")
    
    if testsets:
        concat_testset = datasets.concatenate_datasets(testsets)
        print(f"Mapping TEST dataset ({len(concat_testset)} examples)...")
        
        mapped_testset = concat_testset.map(
            prepare_data_instance,
            fn_kwargs={
                "system_prompt": system_prompt,
                "llm_model_name": llm_model 
            },
            num_proc=args.test_procs
        )
        save_name_test = f"{raw_data_root}/{processed_file_name}_test_{cfg.data_tag}"
        mapped_testset.save_to_disk(save_name_test)
        print(f"Saved Final Test Dataset: {save_name_test}")

        save_name_val = f"{raw_data_root}/{processed_file_name}_validation_{cfg.data_tag}"
        mapped_testset.save_to_disk(save_name_val)
        print(f"Saved Final Validation Dataset: {save_name_val}")
    else:
        print("Warning: No test datasets were concatenated.")

    end_time = time.time()
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")