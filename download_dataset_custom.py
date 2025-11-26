# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import re

import deepchem as dc
import numpy as np
import selfies as sf
import torch
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
import time
from typing import List, Dict, Any

import instructions_smol
import model.added_tokens as added_tokens
from pytorch_lightning import LightningDataModule
import pandas as pd
import datasets

from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.




import multiprocessing as mp
start_time = time.time() 


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
                raise NotImplementedError(
                    f"Label: {label} is not supported in classification task"
                )
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
        if isinstance(label, float):
            label = "{:.10f}".format(label)
        else:
            label = format(float(label), ".10f")

        # force to predict the sign of label first
        if "-" not in label and "+" not in label:
            label = "+" + label
        # unify the length of label to 7
        label = label[:7]
        converted_label = "".join([f"<|{char}|>" for char in label])
        return label_tokens[0] + " " + converted_label + " " + label_tokens[1]
    elif task in REACTION_BENCHMARKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS + NAME_CONVERSION_BENCHMARKS:
        return label_tokens[0] + label + label_tokens[1]
    else:
        raise NotImplementedError


from torch_geometric.data import InMemoryDataset, Data


# TODO use task or refactor it
# getitem shoul return enough information that what is label, and what is the label meaning (to format instruction)
class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, task_subtask_pair, subtask_idx=0, prompt=None):
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        if self.task in CLASSIFICATION_BENCHMARKS:
            self.instruction_templates = getattr(instructions_smol, self.task)
            self.label_tokens = added_tokens.BOOL
        elif self.task in REGRESSION_BENCHMARKS:
            if self.task in ["qm9_additional_label"]:
                subtask_full_name_dict = {
                    "mu": "dipole_moment",
                    "alpha": "isotropic_polarizability",
                    "r2": "electronic_spatial_extent",
                    "zpve": "zero_point_vibrational_energy",
                    "cv": "heat_capacity_298K",
                    "u298": "internal_energy_298K",
                    "h298": "enthalpy_298K",
                    "g298": "free_energy_298K",
                }
                task = self.task.replace("_additional_label", "")
                subtask_full_name = subtask_full_name_dict[self.subtask]
                self.instruction_templates = getattr(
                    instructions_smol, f"{task}_{subtask_full_name}"
                )
                assert len(self.instruction_templates) > 1, "Instruction is not enough"
            else:
                self.instruction_templates = getattr(instructions_smol, f"{self.task}_{self.subtask}".lower())
            self.label_tokens = added_tokens.FLOAT
        else:
            raise NotImplementedError

        self.set_necessary_data()

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        smiles = self.smiles_list[index]
        # set molecule string representation as selfies
        input_mol_string = sf.encoder(smiles)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )
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
        # randomly select one instruction from list
        return graph, label, input_mol_string, instruction

    def set_label_fullname(self):
        self.label_full_name = None
        if self.task == "tox21":
            self.label_full_name = [
                "androgen receptor, full (AR, full)",  # AR
                "androgen receptor, LBD (AR, LBD)",  # AR, LBD
                "aryl hydrocarbon receptor (AhR)",  # AhR
                "aromatase",
                "estrogen receptor alpha, full (ER, full)",  # ER
                "estrogen receptor alpha, LBD (ER, LBD)",  # ER, LBD
                "peroxisome proliferator-activated receptor gamma (PPAR-gamma)",  # PPAR-gamma
                "nuclear factor (erythroid-derived 2)-like 2/antioxidant responsive element (Nrf2/ARE)",
                "ATPase family AAA domain containing 5 (ATAD5)",
                "heat shock factor response element (HSE)",  # HSE
                "mitochondrial membrane potential (MMP)",  # MMP
                "tumor suppressor protein p53",
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
            self.smiles_list.append(Chem.MolToSmiles(mol))

        self.label_list = []
        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []

        self.count_invalid_smiles = 0

        iter_bar = tqdm(
            range(len(self.raw_inputs)),
            total=len(self.raw_inputs),
            desc=f"{self.task}-{self.subtask_idx}",
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.graph_list.append(graph)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.raw_inputs)}"
            )

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class MolInstructionDatset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task = task_subtask_pair
        #self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        if self.task == "bace":
            self.input_list = self.data["SELFIES"][:]
            self.label_list = self.data["label"][:]
        else:
            self.input_list = self.data["input"][:]
            self.label_list = self.data["output"][:]
        self.instruction_templates = getattr(instructions_smol, self.task)

        input_list = []
        label_list = []
        input_mol_string_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.input_list)), total=len(self.input_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                input_list.append(self.input_list[i])
                label_list.append(label)
                input_mol_string_list.append(input_mol_string)
                graph_list.append(graph)
                instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.input_list)}"
            )

        self.input_list = input_list
        self.label_list = label_list
        self.input_mol_string_list = input_mol_string_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        input = self.input_list[index]  # if mol_string, representation is selfies
        label = self.label_list[index]  # if mol_string, representation is selfies

        if self.task in REACTION_BENCHMARKS:
            # two smiles in input0
            if self.task in ["reagent_prediction"]:
                assert ">>" in input
                list_selfies = input.split(
                    ">>"
                )  # reagent prediction has two selfies in input
                input_mol_string = input.replace(
                    ">>",
                    f"{added_tokens.SELFIES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SELFIES[0]}",
                )
                list_smiles = [sf.decoder(s) for s in list_selfies]
                graph = [smiles2data(s) for s in list_smiles]
            # one smiles in input and one smiles in output
            else:
                input_mol_string = input
                smiles = sf.decoder(input_mol_string)
                graph = smiles2data(smiles)
        elif self.task in CLASSIFICATION_BENCHMARKS:
            input_mol_string = input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        else:
            # one selfies in input
            input_mol_string = input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )

        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task, instruction


class ChEBIDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        self.description_list = self.data["description"]
        self.selfies_list = self.data["SELFIES"]
        if "mol2text" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "molecule_captioning"
            )
        elif "text2mol" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "molecule_generation"
            )
        else:
            raise NotImplementedError

        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []
        self.label_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.description_list)),
            total=len(self.description_list),
            desc=self.task,
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.graph_list.append(graph)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.label_list)}"
            )

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        descriptiopn = self.description_list[index]
        selfies = self.selfies_list[index]
        smiles = sf.decoder(selfies)
        print(instruction, descriptiopn, selfies, smiles, "- sample data")

        if self.task in TEXT2MOL_BENCHMARKS:
            label = selfies
            description = (
                added_tokens.DESCRIPTION[0] + descriptiopn + added_tokens.DESCRIPTION[1]
            )
            instruction = instruction.replace("<INPUT>", description)
            graph = smiles2data(
                "CC"
            )  # null smiles, just input dummy graph for batch processing
            input_mol_string = "<None>"
        elif self.task in MOL2TEXT_BENCHMARKS:
            label = descriptiopn
            input_mol_string = selfies
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )
        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class SMolInstructDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")
        # print(f"data: {self.data}, task: {self.task}, subtask: {self.subtask}")
        if "forward_synthesis" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "forward_reaction_prediction"
            )
        else:
            self.instruction_templates = getattr(
                instructions_smol, self.task.replace("smol-", "").replace("-", "_")
            )
        self.set_necesary_data()

    def set_necesary_data(self):
        self.semi_colon_count_input = 0
        self.semi_colon_count_label = 0

        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []
        self.label_list = []

        # pre-load data
        raw_inputs = self.data["raw_input"][:]
        raw_outputs = self.data["raw_output"][:]

        iter_bar = tqdm(
            range(len(self.data)),
            total=len(self.data),
            desc=self.task,
        )
        self.count_invalid_smiles = 0

        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(
                    i, raw_inputs[i], raw_outputs[i]
                )
                self.graph_list.append(graph)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}-smol_set: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}-smol_set: Invalid smiles ratio: {1.0 - len(self.label_list)/len(self.data)}"
            )

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index, raw_input, raw_output):
        raw_input = raw_input
        label = raw_output

        if ";" in raw_input:
            self.semi_colon_count_input += 1
        if ";" in raw_output:
            self.semi_colon_count_label += 1
        if self.task in TEXT2MOL_BENCHMARKS or self.task in NAME_CONVERSION_BENCHMARKS:
            s_token, e_token = (
                added_tokens.IUPAC
                if self.task in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"]
                else added_tokens.DESCRIPTION
            )
            description = raw_input
            description = s_token + description + e_token
            instruction = np.random.choice(self.instruction_templates)
            instruction = instruction.replace("<INPUT>", description)
            # print("TEXT2MOL_BENCHMARKS: raw_input:", raw_input, " raw_output:", raw_output, " label:", label)
            # print("instruction:", instruction, " description:", description)
            graph = smiles2data(
                "CC"
            )  # null smiles, just input dummy graph for batch processing
            input_mol_string = "<None>"
            label = re.sub(r"\s*;\s*", ".", label)
            
        elif self.task in REACTION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = raw_input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        # multi labeled property prediction datasets
        elif self.task in ["smol-property_prediction-sider"]:
            instance_input = self.data[index]["input"]
            assert re.search(r"\[.*\]", instance_input) is not None
            instruction = re.sub(r"\[.*\]", "<INPUT>", instance_input)

            # use re sub to replace ";" with "."
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        elif (
            self.task
            in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS
        ):
            instruction = np.random.choice(self.instruction_templates)
            # use re sub to replace ";" with "."
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        else:
            raise NotImplementedError(f"Task: {self.task} is not supported")

        label = wrap_label(label, self.task)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )
        
        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


from ogb.utils import smiles2graph


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# torch_geometric.data.Data variants for paired graph data, i.e. reagent prediction
class PairData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index":
            return self.x.size(0)
        elif key == "additional_edge_index":
            return self.additional_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class PackedData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "edge_index" in key:
            prefix = key.split("edge_index")[0]
            return getattr(self, f"{prefix}x.size")(0)
        return super().__inc__(key, value, *args, **kwargs)

from rdkit import Chem


def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None


import yaml


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
    # get dataset from deepchem
    if "smol" in task_name:
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=True,
            insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)  # remove smol- from smol-<task_name>

        # DEBUG: to avoid lengthy processing time
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
    elif task_name in ["qm9_additional_label"]:
        loading_fn = dc.molnet.load_qm9
    # TODO: address biot5 datasets
    elif task_name == "bace":
        train_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_train.csv")
        )
        valid_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv")
        )
        test_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_test.csv")
        )
        tasks = [task_name]
    elif "chebi-20" in task_name:
        # load data from csv
        train_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/chebi20_mol2text_train.csv")
        )
        valid_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/chebi20_mol2text_validation.csv")
        )
        test_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/chebi20_mol2text_test.csv")
        )

        tasks = [task_name]

    # mol-instruction datasets
    elif task_name in [
        "reagent_prediction",
        "forward_reaction_prediction",
        "retrosynthesis",
        "qm9_homo",
        "qm9_lumo",
        "qm9_homo_lumo_gap",
    ]:
        mol_instruction_dataset = load_dataset(
            "zjunlp/Mol-Instructions",
            "Molecule-oriented Instructions",
            trust_remote_code=True,
        )
        if "qm9_" in task_name:
            dataset = mol_instruction_dataset["property_prediction"]
            subtask_name = task_name.split("_")[1]
            subtask_instruction_templates = getattr(
                instructions_smol, "filtering_template_" + subtask_name
            )
            dataset = dataset.filter(
                lambda x: x["instruction"] in subtask_instruction_templates
            )
            assert len(dataset) > 0, f"len(dataset) = {len(dataset)}"
        else:
            dataset = mol_instruction_dataset[task_name]

        train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
        split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
        train_dataset, valid_dataset = split["train"], split["test"]

        test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
        tasks = [task_name]
    else:
        print()
        raise NotImplementedError

    # dataset from deepchem
    if (
        task_name in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS
        and task_name not in ["qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap", "bace"]
        and "smol" not in task_name
    ):
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
        tasks, datasets, transformers = loading_fn(
            featurizer="Raw",
            splitter="scaffold",
            save_dir=base_path,
            data_dir=base_path,
            reload=True,
        )
        train_dataset, valid_dataset, test_dataset = datasets
    else:
        # dataset from mol-instruction is already loaded
        pass
    return tasks, train_dataset, valid_dataset, test_dataset


def from_dict(dict):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Struct(**dict)

if __name__ == "__main__":
    import argparse
    import os
    import random

    # get arg replace_ratio, dataset_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs/download/")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--train_procs", type=int, default=32)
    parser.add_argument("--test_procs", type=int, default=32)
    args = parser.parse_args()

    
    arg_path = os.path.join(args.config_dir, args.config) + ".yaml"
    # read config file
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # convert args to be accessible its values by attributes
    cfg = from_dict(cfg)

    raw_data_root = cfg.raw_data_root

    if not os.path.exists(raw_data_root):
        os.makedirs(raw_data_root)

    start, end = added_tokens.SELFIES
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(
        cfg.target_benchmarks
    )
    data_tag = cfg.data_tag

    downloading_task_subtask_pairs = []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        if os.path.exists(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        ) and os.path.exists(f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"):
            print(f"{task}_{subtask_idx} already exists")
        else:
            downloading_task_subtask_pairs.append(task_subtask_pair)

    for task_subtask_pair in tqdm(
        downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"
    ):
        task_name = task_subtask_pair[0]
        new_dataset = get_dataset(task_name=task_name, raw_data_root=raw_data_root)

        subtasks = new_dataset[0]
        subtask_idx = task_subtask_pair[1]
        if subtask_idx == "multi_label_classification":
            task_subtask_pair = f"{task_name}/multi_label_classification"
        elif task_name in ["toxcast", "tox21", "qm9_additional_label", "hopv"]:
            task_subtask_pair = f"{task_name}/{subtasks[subtask_idx]}"
        else:
            task_subtask_pair = f"{task_name}/0"

        data_split = new_dataset[1:]  # train_set, val_set, test_set
        
        def _task_arg_for(dataset_cls):
            # MolInstructionDatset만은 'task' 단독 문자열이어야 함
            return task_name if dataset_cls is MolInstructionDatset else task_subtask_pair

        dataset_cls = (
            SMolInstructDataset if "smol" in task_name else
            MoleculeNetDatasetDeepChem if task_name in ["toxcast","tox21","qm9_additional_label","hopv"] else
            ChEBIDataset if task_name in ["chebi-20-mol2text","chebi-20-text2mol"] else
            MolInstructionDatset
        )
        
        if "smol" in task_name:
            dataset = SMolInstructDataset
        elif task_name in [
            "toxcast",
            "tox21",
            "qm9_additional_label",
            "hopv"
        ]:
            dataset = MoleculeNetDatasetDeepChem
        elif task_name in ["chebi-20-mol2text", "chebi-20-text2mol"]:
            dataset = ChEBIDataset
        # qm9 in regression benchmark is processed via MolInstructionDataset
        elif task_name in [
            "chebi-20-text2mol",
            "chebi-20-mol2text",
            "reagent_prediction",
            "forward_reaction_prediction",
            "retrosynthesis",
            "qm9_homo",
            "qm9_lumo",
            "qm9_homo_lumo_gap",
            "bace",
        ]:
            dataset = MolInstructionDatset
        print(f"Using dataset class: {dataset_cls}")
        valid_dataset = dataset(
            data=data_split[1],
            task_subtask_pair=_task_arg_for(dataset_cls),
            subtask_idx=subtask_idx,
        )
        test_dataset = dataset(
            data=data_split[2],
            task_subtask_pair=_task_arg_for(dataset_cls),
            subtask_idx=subtask_idx,
        )
        train_dataset = dataset(
            data=data_split[0],
            task_subtask_pair=_task_arg_for(dataset_cls),
            subtask_idx=subtask_idx,
        )
        dataset_splits = {
            "val": valid_dataset,
            "test": test_dataset,
            "train": train_dataset,
        }

        for split in dataset_splits.keys():
            dataset = dataset_splits[split]
            list_dict_data = []
            for i in range(len(dataset)):
                # 원래 튜플 해체
                graph, label, input_mol_string, task_pair_or_name, instruction = dataset[i]

                # 1) instruction 이 numpy.str_ 일 수도 있어서 문자열로 강제
                if hasattr(instruction, "item"):
                    instruction = instruction.item()
                instruction = str(instruction)

                # 2) reagent_prediction 등: graph 가 list[Data, Data]
                #    -> 첫 번째를 main, 두 번째를 additional 로 넣어줍니다
                if isinstance(graph, list):
                    if len(graph) >= 2:
                        g0, g1 = graph[0], graph[1]
                    elif len(graph) == 1:
                        g0 = g1 = graph[0]
                    else:
                        # 그래프가 비면 스킵
                        continue
                else:
                    g0 = g1 = graph  # 단일 그래프인 대부분의 태스크

                # 3) dict_data 생성
                dict_data = {
                    "x": g0.x,
                    "edge_index": g0.edge_index,
                    "edge_attr": g0.edge_attr,
                    "label": label,
                    "input_mol_string": input_mol_string,
                    "task_subtask_pair": task_pair_or_name,
                    "instruction": instruction,
                    # reagent_prediction 처럼 2개짜리인 경우를 지원
                    "additional_x": g1.x,
                    "additional_edge_index": g1.edge_index,
                    "additional_edge_attr": g1.edge_attr,
                }
                list_dict_data.append(dict_data)

            # 만약 list_dict_data 가 비면 save_to_disk 가 또 실패하니 방어
            if not list_dict_data:
                print(f"[warn] {task_name} split={split} produced no valid samples; skipping save.")
                continue

            dataset = datasets.Dataset.from_list(list_dict_data)
            dataset.save_to_disk(f"{raw_data_root}/{task_name}_subtask-{subtask_idx}_{split}")


    trainsets = []
    testsets = []
    valsets = []
    trainsets_dict = {}
    testsets_dict = {}
    valsets_dict = {}

    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        trainset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        )
        trainsets.append(trainset)
        trainsets_dict[task_subtask_pair] = trainset
        valset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_val"
        )
        valsets.append(valset)
        valsets_dict[task_subtask_pair] = valset
        testset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"
        )
        #! Name Conversion Task는 Test에 포함하지 않음.
        if task in NAME_CONVERSION_BENCHMARKS:
            print(f"[info] skipping test split for NAME CONVERSION task: {task}")
        else:
            testsets.append(testset)
            testsets_dict[task_subtask_pair] = testset

        print(f"{task}_{subtask_idx} loaded")
        print(f"{task}_{subtask_idx} loaded")

    # ------------------ Step 2: Deduplication following the paper ------------------
    from datasets import concatenate_datasets

    SELFIES_START, SELFIES_END = added_tokens.SELFIES

    def _extract_single_selfies_block(text: str):
        """
        첫 번째 <SELFIES> ... </SELFIES> 사이의 내용을 추출.
        못 찾으면 None.
        """
        if text is None:
            return None
        try:
            start_idx = text.index(SELFIES_START)
            end_idx = text.index(SELFIES_END, start_idx + len(SELFIES_START))
            inner = text[start_idx + len(SELFIES_START): end_idx]
            return inner.strip()
        except ValueError:
            return None

    def _mol_key_from_caption_input(input_mol_string: str):
        """
        Molecule Captioning 용 분자 키:
        input_mol_string 안의 SELFIES -> SMILES -> canonical SMILES
        """
        selfies = _extract_single_selfies_block(input_mol_string)
        if selfies is None:
            return None
        try:
            smiles = sf.decoder(selfies)
        except Exception:
            return None
        return get_canonical_smiles(smiles)

    def _mol_key_from_text2mol_label(label: str):
        """
        Description-Guided Molecule Generation (Text2Mol) 용 분자 키:
        label 안의 SELFIES -> SMILES -> canonical SMILES
        """
        selfies = _extract_single_selfies_block(label)
        if selfies is None:
            return None
        try:
            smiles = sf.decoder(selfies)
        except Exception:
            return None
        return get_canonical_smiles(smiles)

    def _mol_key_from_reaction_input(input_mol_string: str):
        """
        Forward / Retro 용 반응 키:
        일단 전체 입력 문자열을 그대로 키로 사용 (반응 단위 dedup).
        """
        s = input_mol_string.strip()
        return s or None

    # 그룹 정의 (dataset 내부 field "task_subtask_pair" 기준)
    CAPTION_TASK_TAGS = {"chebi-20-mol2text/0", "smol-molecule_captioning/0"}
    TEXT2MOL_TASK_TAGS = {"chebi-20-text2mol/0", "smol-molecule_generation/0"}
    RETRO_TASK_TAGS   = {"retrosynthesis", "smol-retrosynthesis/0"}
    FORWARD_TASK_TAGS = {"forward_reaction_prediction", "smol-forward_synthesis/0"}

    def _build_key_to_tag_dict(split_dict):
        """
        split_dict: {(task, subtask_idx) -> Dataset}
        각 Dataset의 첫 샘플에서 'task_subtask_pair' 문자열을 읽어서
        key -> tag(str) 매핑을 만든다.
        """
        mapping = {}
        for k, ds in split_dict.items():
            if len(ds) == 0:
                continue
            tag = ds[0]["task_subtask_pair"]  # 예: 'smol-molecule_captioning/0'
            mapping[k] = tag
        return mapping

    def _collect_split_mol_keys(split_dict, key_to_tag, target_tags, mode):
        """
        하나의 split(dict)에 대해 분자/반응 키를 모은다.
        mode: 'caption' | 'text2mol' | 'reaction'
        """
        keys = set()
        for k, tag in key_to_tag.items():
            if tag not in target_tags:
                continue
            ds = split_dict.get(k, None)
            if ds is None or len(ds) == 0:
                continue

            if mode in ("caption", "reaction"):
                ims_list = ds["input_mol_string"]
                for ims in ims_list:
                    if mode == "caption":
                        mk = _mol_key_from_caption_input(ims)
                    else:
                        mk = _mol_key_from_reaction_input(ims)
                    if mk is not None:
                        keys.add(mk)

            elif mode == "text2mol":
                labels = ds["label"]
                for lab in labels:
                    mk = _mol_key_from_text2mol_label(lab)
                    if mk is not None:
                        keys.add(mk)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return keys

    def _filter_train_by_mol_keys(trainsets_dict, key_to_tag, target_tags, mode, forbidden_keys):
        """
        target_tags 에 해당하는 task들의 train split에서
        forbidden_keys(= eval에서 나온 분자/반응 키)에 해당하는 샘플을 제거.
        -> 'eval(test+val)에 나온 분자는 train에서 안 쓰인다' 보장.
        """
        for k, tag in list(key_to_tag.items()):
            if tag not in target_tags:
                continue
            ds = trainsets_dict.get(k, None)
            if ds is None or len(ds) == 0:
                continue

            keep_indices = []
            if mode in ("caption", "reaction"):
                ims_list = ds["input_mol_string"]
                for idx, ims in enumerate(ims_list):
                    if mode == "caption":
                        mk = _mol_key_from_caption_input(ims)
                    else:
                        mk = _mol_key_from_reaction_input(ims)
                    # mk == None 이면 기준이 애매하므로 그냥 유지
                    if mk is None or mk not in forbidden_keys:
                        keep_indices.append(idx)

            elif mode == "text2mol":
                labels = ds["label"]
                for idx, lab in enumerate(labels):
                    mk = _mol_key_from_text2mol_label(lab)
                    if mk is None or mk not in forbidden_keys:
                        keep_indices.append(idx)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if len(keep_indices) != len(ds):
                print(f"[dedup] {tag}: train size {len(ds)} -> {len(keep_indices)} "
                      f"after removing molecules that appear in eval (test+val)")

            trainsets_dict[k] = ds.select(keep_indices)

    # ---- 실제 적용: test + val 모두 eval로 간주 ----
    train_key_to_tag = _build_key_to_tag_dict(trainsets_dict)
    test_key_to_tag  = _build_key_to_tag_dict(testsets_dict)
    val_key_to_tag   = _build_key_to_tag_dict(valsets_dict)

    def _collect_eval_keys(target_tags, mode):
        keys = set()
        # test에서 키 수집
        keys |= _collect_split_mol_keys(testsets_dict, test_key_to_tag, target_tags, mode)
        # validation에서도 동일하게 수집
        keys |= _collect_split_mol_keys(valsets_dict,  val_key_to_tag,  target_tags, mode)
        return keys

    # 1) Molecule Captioning (ChEBI-20 + SMolInstruct)
    caption_eval_keys = _collect_eval_keys(CAPTION_TASK_TAGS, mode="caption")
    _filter_train_by_mol_keys(
        trainsets_dict, train_key_to_tag, CAPTION_TASK_TAGS, mode="caption",
        forbidden_keys=caption_eval_keys,
    )

    # 2) Description-Guided Molecule Generation (Text2Mol)
    text2mol_eval_keys = _collect_eval_keys(TEXT2MOL_TASK_TAGS, mode="text2mol")
    _filter_train_by_mol_keys(
        trainsets_dict, train_key_to_tag, TEXT2MOL_TASK_TAGS, mode="text2mol",
        forbidden_keys=text2mol_eval_keys,
    )

    # 3) Retrosynthesis (Mol-Instructions + SMolInstruct)
    retro_eval_keys = _collect_eval_keys(RETRO_TASK_TAGS, mode="reaction")
    _filter_train_by_mol_keys(
        trainsets_dict, train_key_to_tag, RETRO_TASK_TAGS, mode="reaction",
        forbidden_keys=retro_eval_keys,
    )

    # 4) Forward Reaction Prediction (Mol-Instructions + SMolInstruct)
    forward_eval_keys = _collect_eval_keys(FORWARD_TASK_TAGS, mode="reaction")
    _filter_train_by_mol_keys(
        trainsets_dict, train_key_to_tag, FORWARD_TASK_TAGS, mode="reaction",
        forbidden_keys=forward_eval_keys,
    )

    # --- dedup 반영된 dict로부터 다시 trainsets / testsets 리스트 재구성 ---
    trainsets = []
    testsets = []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        trainsets.append(trainsets_dict[task_subtask_pair])
        # Name Conversion Task는 여전히 test에서 제외
        if task in NAME_CONVERSION_BENCHMARKS:
            continue
        if task_subtask_pair in testsets_dict:
            testsets.append(testsets_dict[task_subtask_pair])

    concat_trainset = datasets.concatenate_datasets(trainsets)
    concat_testset  = datasets.concatenate_datasets(testsets)


        
    from transformers import AutoTokenizer
    system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."


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
        # assert "<INPUT>" in input_prompt, f"llm_prompt should contain <INPUT_MOL>, {input_prompt}"
        if "<INPUT>" in input_prompt:
            input_prompt = input_prompt.replace("<INPUT>", input_mol_string)
        else:
            input_prompt = input_prompt

        formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [INST]"
        formatted_target_text = data_instance["label"] + " </s>"

        if "additional" in data_instance["task_subtask_pair"]:
            convert_dict = {
                'qm9_additional_label/mu' : "qm9_dipole_moment",
                'qm9_additional_label/alpha' : "qm9_isotropic_polarizability",
                'qm9_additional_label/r2' : "qm9_electronic_spatial_extent",
                'qm9_additional_label/zpve' : "qm9_zero_point_vibrational_energy",

            }
            task = convert_dict[data_instance["task_subtask_pair"]]
        else:
            task = data_instance["task_subtask_pair"]

        data ={
            "task": task,
            "x": data_instance["x"],
            "edge_index": data_instance["edge_index"],
            "edge_attr": data_instance["edge_attr"],
            "additional_x": data_instance["additional_x"],
            "additional_edge_index": data_instance["additional_edge_index"],
            "additional_edge_attr": data_instance["additional_edge_attr"],
            "prompt_text": formatted_prompt_text,
            "target_text": formatted_target_text,
        }
        return data

    
    data_instance = concat_testset[0]

    remove_keys = set(concat_testset.column_names)
    remove_keys -= {
        "task",
        "x",
        "edge_index",
        "edge_attr",
        "additional_x",
        "additional_edge_index",
        "additional_edge_attr",
    }
    mapped_trainset = concat_trainset.map(
        lambda x: prepare_data_instance(
            x, system_prompt=system_prompt,
        ),
            # use 36 processes
            num_proc=args.train_procs
    )
    mapped_testset = concat_testset.map(
        lambda x: prepare_data_instance(
            x, system_prompt=system_prompt,
        ),
            # use 36 processes
            num_proc=args.test_procs
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
    mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_test_{cfg.data_tag}")
    mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_validation_{cfg.data_tag}")

    a = 17
    end_time = time.time()
    print(f"End time: {(end_time - start_time) / 60} minutes {(end_time - start_time) % 60} seconds")