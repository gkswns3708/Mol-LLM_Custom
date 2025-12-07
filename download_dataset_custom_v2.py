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

start_time = time.time()

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
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS or task == 'smol-name_conversion-i2s':
        label_tokens = added_tokens.SELFIES
    else:
        raise NotImplementedError

    if task in CLASSIFICATION_BENCHMARKS:
        if isinstance(label, str):
            if "true" in label.lower() or "yes" in label.lower():
                label = "True"
            elif "false" in label.lower() or "no" in label.lower():
                label = "False"
            else:
                label = "False" 
            label = label_tokens[0] + label + label_tokens[1]
        elif isinstance(label, list):
            label_language = ", ".join(label)
            label_boolean = "True" * len(label)
            label = label_language + label_tokens[0] + label_boolean + label_tokens[1]
        else:
            label = "True" if label else "False"
            label = label_tokens[0] + label + label_tokens[1]
        return label
        
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, float) or isinstance(label, int):
            label = "{:.10f}".format(float(label))
        else:
            try:
                label = format(float(label), ".10f")
            except:
                raise ValueError(f"Invalid label for regression: {label}")

        if "-" not in label and "+" not in label:
            label = "+" + label
        label = label[:7]
        converted_label = "".join([f"<|{char}|>" for char in label])
        return label_tokens[0] + " " + converted_label + " " + label_tokens[1]
        
    elif task in REACTION_BENCHMARKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS + NAME_CONVERSION_BENCHMARKS:
        return label_tokens[0] + str(label) + label_tokens[1]
    else:
        raise NotImplementedError

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
        self.task, self.subtask = task_subtask_pair.split("/")
        
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
            if input_mol_string is None: 
                 input_mol_string = smiles 
        except:
             raise ValueError(f"Selfies encoding failed for {smiles}")

        input_mol_string = added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        
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
        
        graph = smiles2data(smiles)
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
                if self.count_invalid_smiles == 0:
                    print(f"\n[Error] Task: {self.task}, Index: {i}")
                    print(traceback.format_exc())
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
        self.task = task_subtask_pair
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
                if self.count_invalid_smiles == 0:
                    print(f"\n[Error] Task: {self.task}, Index: {i}")
                    print(traceback.format_exc())
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
                assert ">>" in input_
                list_selfies = input_.split(">>")
                input_mol_string = input_.replace(">>", f"{added_tokens.SELFIES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SELFIES[0]}")
                list_smiles = [sf.decoder(s.strip()) for s in list_selfies]
                graph = [smiles2data(s) for s in list_smiles]
            else:
                input_mol_string = input_
                smiles = sf.decoder(input_mol_string.strip())
                graph = smiles2data(smiles)
        
        elif self.task in CLASSIFICATION_BENCHMARKS:
            input_mol_string = input_
            smiles = sf.decoder(input_mol_string.strip())
            if smiles is None: 
                raise ValueError(f"SELFIES decoding failed for: {input_mol_string}")
            graph = smiles2data(smiles)
            
        else: 
            input_mol_string = input_
            smiles = sf.decoder(input_mol_string.strip())
            if smiles is None:
                raise ValueError(f"SELFIES decoding failed for: {input_mol_string}")
            graph = smiles2data(smiles)
            
        label = wrap_label(label_, self.task)
        input_mol_string = added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        return graph, label, input_mol_string, instruction
        
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task, self.instruction_list[index]

class ChEBIDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")
        self.set_necesary_data()

    def set_necesary_data(self):
        self.description_list = self.data["description"]
        self.selfies_list = self.data["SELFIES"]
        if "mol2text" in self.task: self.instruction_templates = getattr(instructions_smol, "molecule_captioning")
        elif "text2mol" in self.task: self.instruction_templates = getattr(instructions_smol, "molecule_generation")
        
        processed_graph_list = []
        processed_label_list = []
        processed_input_mol_string_list = []
        processed_instruction_list = []
        
        self.count_invalid_smiles = 0
        for i in tqdm(range(len(self.description_list)), desc=self.task):
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_graph_list.append(graph)
                processed_instruction_list.append(instruction)
            except Exception as e:
                if self.count_invalid_smiles == 0:
                    print(f"\n[Error] Task: {self.task}, Index: {i}")
                    print(traceback.format_exc())
                self.count_invalid_smiles += 1
        
        self.graph_list = processed_graph_list
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.instruction_list = processed_instruction_list

    def __len__(self): return len(self.label_list)
    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        desc = self.description_list[index]
        selfies = self.selfies_list[index]
        
        if pd.isna(desc) or pd.isna(selfies): raise ValueError("NaN data")
        
        smiles = sf.decoder(selfies)
        if smiles is None: raise ValueError(f"Selfies decode failed: {selfies}")

        if self.task in TEXT2MOL_BENCHMARKS:
            label = selfies
            description = added_tokens.DESCRIPTION[0] + desc + added_tokens.DESCRIPTION[1]
            instruction = instruction.replace("<INPUT>", description)
            graph = smiles2data("CC")
            input_mol_string = "<None>"
        elif self.task in MOL2TEXT_BENCHMARKS:
            label = desc
            input_mol_string = selfies
            graph = smiles2data(smiles)
        label = wrap_label(label, self.task)
        input_mol_string = added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        return graph, label, input_mol_string, instruction
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]

class SMolInstructDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")
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
        for i in tqdm(range(len(self.data)), desc=self.task):
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i, raw_inputs[i], raw_outputs[i])
                processed_graph_list.append(graph)
                processed_label_list.append(label)
                processed_input_mol_string_list.append(input_mol_string)
                processed_instruction_list.append(instruction)
            except Exception as e:
                if self.count_invalid_smiles == 0:
                    print(f"\n[Error] Task: {self.task}, Index: {i}")
                    print(traceback.format_exc())
                self.count_invalid_smiles += 1
        
        self.graph_list = processed_graph_list
        self.label_list = processed_label_list
        self.input_mol_string_list = processed_input_mol_string_list
        self.instruction_list = processed_instruction_list

    def __len__(self): return len(self.label_list)
    def get_necessary_data(self, index, raw_input, raw_output):
        label = raw_output
        if self.task in TEXT2MOL_BENCHMARKS or self.task in NAME_CONVERSION_BENCHMARKS:
            s_token, e_token = (added_tokens.IUPAC if self.task in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"] else added_tokens.DESCRIPTION)
            description = s_token + raw_input + e_token
            instruction = np.random.choice(self.instruction_templates).replace("<INPUT>", description)
            graph = smiles2data("CC")
            input_mol_string = "<None>"
            label = re.sub(r"\s*;\s*", ".", label)
        elif self.task in REACTION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = raw_input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        elif self.task in ["smol-property_prediction-sider"]:
            instance_input = self.data[index]["input"]
            instruction = re.sub(r"\[.*\]", "<INPUT>", instance_input)
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        elif self.task in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        else:
            raise NotImplementedError
        label = wrap_label(label, self.task)
        input_mol_string = added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        return graph, label, input_mol_string, instruction
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index], self.input_mol_string_list[index], self.task_subtask_pair, self.instruction_list[index]

def get_dataset(task_name, raw_data_root):
    if "smol" in task_name:
        script_path = "./InstructGraph.py"
        smol_dataset = load_dataset(
            script_path,
            # use_selfies=True, 
            insert_core_tags=False,
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)
        train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
        valid_dataset = smol_dataset["validation"].filter(lambda x: x["task"] == _task)
        test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
        tasks = [task_name]

    elif task_name in ["toxcast", "tox21", "hopv"]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}")
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets_, transformers = loading_fn(featurizer="Raw", splitter="scaffold", save_dir=base_path, data_dir=base_path, reload=True)
        train_dataset, valid_dataset, test_dataset = datasets_

    elif task_name in ["qm9_additional_label"]:
        loading_fn = dc.molnet.load_qm9
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets_, transformers = loading_fn(featurizer="Raw", splitter="scaffold", save_dir=base_path, data_dir=base_path, reload=True)
        train_dataset, valid_dataset, test_dataset = datasets_

    elif task_name == "bace":
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/BioT5_bace_test.csv"))
        tasks = [task_name]

    elif "chebi-20" in task_name:
        train_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/chebi20_mol2text_train.csv"))
        valid_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/chebi20_mol2text_validation.csv"))
        test_dataset = pd.read_csv(os.path.join(raw_data_root, "raw/chebi20_mol2text_test.csv"))
        tasks = [task_name]

    elif task_name in ["reagent_prediction", "forward_reaction_prediction", "retrosynthesis", "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]:
        mol_instruction_dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", trust_remote_code=True)
        if "qm9_" in task_name:
            dataset = mol_instruction_dataset["property_prediction"]
            subtask_name = task_name.split("_")[1]
            subtask_instruction_templates = getattr(instructions_smol, "filtering_template_" + subtask_name)
            dataset = dataset.filter(lambda x: x["instruction"] in subtask_instruction_templates)
        else:
            dataset = mol_instruction_dataset[task_name]
        
        train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
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
        mol_token="<mol>",
        num_query_tokens=32,
):
    input_mol_string = data_instance["input_mol_string"]
    input_mol_string = input_mol_string.replace("<SELFIES>", "<SELFIES> ").replace("</SELFIES>", " </SELFIES>")
    input_prompt = data_instance["instruction"]

    graph_sequence = "<GRAPH>" + mol_token * num_query_tokens + "</GRAPH>"
    input_mol_string += graph_sequence
    
    if "<INPUT>" in input_prompt:
        input_prompt = input_prompt.replace("<INPUT>", input_mol_string)

    formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [INST]"
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
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--train_procs", type=int, default=32)
    parser.add_argument("--test_procs", type=int, default=32)
    args = parser.parse_args()

    arg_path = os.path.join(args.config_dir, args.config) + ".yaml"
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = from_dict(cfg)

    raw_data_root = cfg.raw_data_root
    if not os.path.exists(raw_data_root):
        os.makedirs(raw_data_root)

    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(cfg.target_benchmarks)
    
    downloading_task_subtask_pairs = []
    # Always try to regenerate to fix the empty data issue, or check file validity
    # For now, let's keep the check but assume the user deletes bad folders if needed.
    # To be safe, we can just append all.
    for task_subtask_pair in task_subtask_pairs:
        downloading_task_subtask_pairs.append(task_subtask_pair)

    # --- Dataset Downloading & Saving (First Stage) ---
    for task_subtask_pair in tqdm(downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"):
        task_name = task_subtask_pair[0]
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
        print(f"Using dataset class: {dataset_cls} for {task_name}")
        
        splits = {
            "train": data_split[0],
            "val": data_split[1], 
            "test": data_split[2]
        }

        for split_name, raw_data in splits.items():
            # Skip if raw data is None/empty
            if raw_data is None or len(raw_data) == 0:
                print(f"[warn] Raw data for {task_name} {split_name} is empty.")
                continue

            save_path = f"{raw_data_root}/{task_name}_subtask-{subtask_idx}_{split_name}"
            # if os.path.exists(save_path):
            #    print(f"Skipping {save_path}, exists.")
            #    continue

            # Instantiate Wrapper (preprocessing happens here)
            ds = dataset_wrapper(data=raw_data, task_subtask_pair=_task_arg_for(dataset_cls), subtask_idx=subtask_idx)
            
            list_dict_data = []
            for i in range(len(ds)):
                graph, label, input_mol_string, task_pair_or_name, instruction = ds[i]
                if hasattr(instruction, "item"): instruction = instruction.item()
                instruction = str(instruction)

                if isinstance(graph, list):
                    if len(graph) >= 2: g0, g1 = graph[0], graph[1]
                    elif len(graph) == 1: g0 = g1 = graph[0]
                    else: continue
                else:
                    g0 = g1 = graph

                list_dict_data.append({
                    "x": g0.x, "edge_index": g0.edge_index, "edge_attr": g0.edge_attr,
                    "label": label, "input_mol_string": input_mol_string,
                    "task_subtask_pair": task_pair_or_name, "instruction": instruction,
                    "additional_x": g1.x, "additional_edge_index": g1.edge_index, "additional_edge_attr": g1.edge_attr,
                })
            
            if not list_dict_data:
                print(f"[warn] {task_name} split={split_name} empty after processing.")
                continue

            hf_ds = datasets.Dataset.from_list(list_dict_data)
            print(f"Saving to {save_path} ...")
            hf_ds.save_to_disk(save_path)

    # --- Concatenate & Mapping (Second Stage) ---
    print("\n--- Starting Concatenation and Mapping ---")
    trainsets, testsets = [], []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        
        train_path = f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        if os.path.exists(train_path):
            try:
                trainsets.append(datasets.Dataset.load_from_disk(train_path))
            except Exception as e:
                print(f"Failed to load {train_path}: {e}")

        # Name Conversion 제외
        if task in NAME_CONVERSION_BENCHMARKS:
            print(f"[info] skipping test split for NAME CONVERSION task: {task}")
        else:
            test_path = f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"
            if os.path.exists(test_path):
                try:
                    testsets.append(datasets.Dataset.load_from_disk(test_path))
                except Exception as e:
                    print(f"Failed to load {test_path}: {e}")
    
    # Mapping with corrected prepare_data_instance
    system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."

    if trainsets:
        concat_trainset = datasets.concatenate_datasets(trainsets)
        
        print("Mapping TRAIN dataset...")
        mapped_trainset = concat_trainset.map(
            prepare_data_instance,
            fn_kwargs={"system_prompt": system_prompt},
            num_proc=args.train_procs
        )
        
        llm_model = "mistralai/Mistral-7B-Instruct-v0.3"
        mol_representation = "string+graph"
        num_query_token = 32
        base_model = llm_model.replace("/", "-")
        tags = [base_model, mol_representation]
        if "graph" in mol_representation:
            tags += [f"q{num_query_token}"]
        processed_file_name = "_".join(tags)

        mapped_trainset.save_to_disk(f"{raw_data_root}/{processed_file_name}_train_{cfg.data_tag}")
    else:
        print("No train datasets found/processed.")

    if testsets:
        concat_testset = datasets.concatenate_datasets(testsets)
        print("Mapping TEST dataset...")
        mapped_testset = concat_testset.map(
            prepare_data_instance,
            fn_kwargs={"system_prompt": system_prompt},
            num_proc=args.test_procs
        )
        mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_test_{cfg.data_tag}")
        # Validation은 Test와 동일하게 저장 (사용자 요구)
        mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_validation_{cfg.data_tag}")
    else:
        print("No test datasets found/processed.")

    end_time = time.time()
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")