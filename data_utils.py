import torch
from transformers import DataCollatorForSeq2Seq
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

from collections import Counter

import selfies as sf

import rdkit.Chem as Chem
import re
import copy

CLASSIFICATION_BENCHMARKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
]
REGRESSION_BENCHMARKS = [
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "qm9_dipole_moment",
    "qm9_isotropic_polarizability",
    "qm9_electronic_spatial_extent",
    "qm9_zero_point_vibrational_energy",
    "qm9_heat_capacity_298K",
    "qm9_internal_energy_298K",
    "qm9_enthalpy_298K",
    "qm9_free_energy_298K",
    "alchemy_homo",
    "alchemy_lumo",
    "alchemy_homo_lumo_gap",
    "aqsol-logS",
    "pcqm_homo_lumo_gap",
]
REACTION_BENCHMARKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
    "presto-forward_reaction_prediction",
    "presto-retrosynthesis",
    "presto-reagent_prediction",
    "orderly-forward_reaction_prediction",
    "orderly-retrosynthesis",
    "orderly-reagent_prediction",
]
TEXT2MOL_BENCHMARKS = [
    "chebi-20-text2mol",
    "smol-molecule_generation",
]
MOL2TEXT_BENCHMARKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]
NAME_CONVERSION_BENCHMARKS = [
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
]


tasks = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)

input_mol_string_pattern = re.compile("<SELFIES>.*?</SELFIES>")
# [\s\S]는 공백과 공백이 아닌 모든 문자를 의미하므로 줄바꿈 포함 모든 것을 잡습니다.
graph_sequence = re.compile(r"<GRAPH>.*?</GRAPH>", re.DOTALL)


def task2id(task):
    # task name to task id
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]


def id2task(task_id):
    # task id to task name
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        train=True,
        args=None,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.train = train
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.mol_representation = args.mol_representation

        self.apply_molpo = args.train_molpo if self.train else args.eval_molpo

        self.projector_type = args.projector_type
        self.current_epoch = args.current_epoch
        self.args = args

        if self.mol_representation in ["string+graph", "graph_only"]:
            self.graph_collator = GraphCollater([], [])

    def select_mol_representation(self, prompt_text, mol_representation="string+graph"):
        print(mol_representation, "- data_utils/mol_representation")
        if mol_representation == "string+graph":
            return prompt_text
        elif mol_representation == "string_only":
            string_only_prompt_text = [graph_sequence.sub("", p) for p in prompt_text]
            return string_only_prompt_text
        elif mol_representation == "graph_only":
            graph_only_prompt_text = [
                input_mol_string_pattern.sub("", p) for p in prompt_text
            ]
            return graph_only_prompt_text
        else:
            raise ValueError(
                "mol_representation should be one of ['string+graph', 'string_only', 'graph_only']"
            )

    def enumerate_selfies(
        self,
        origin_selfies,
    ):
        origin_smiles = sf.decoder(origin_selfies)

        isomericSmiles = bool(self.args.isomericSmiles)
        canonical = bool(self.args.canonical)
        allHsExplicit = bool(self.args.allHsExplicit)

        processed_smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(origin_smiles),
            isomericSmiles=isomericSmiles,
            canonical=canonical,
            doRandom=not canonical,
            allHsExplicit=allHsExplicit,
            allBondsExplicit=False,
            kekuleSmiles=False,
        )
        processed_selfies = sf.encoder(processed_smiles)
        return processed_selfies

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # tasks = [task2id(sample["task"]) for sample in batch]  # task id
        temp = [sample for sample in batch]
        tasks = [task2id(sample["task"].split("/", 1)[0]) for sample in batch]
        task_names = [id2task(task) for task in tasks]
        prompt_text = [sample["prompt_text"] for sample in batch]
        target_text = [sample["target_text"] for sample in batch]
        input_mol_strings = [sample["input_mol_string"] for sample in batch]
        list_selfies = [
            i.replace("<SELFIES> ", "").replace(" </SELFIES>", "")
            for i in input_mol_strings
        ]
        list_graphs = [
            Data(
                x=torch.tensor(sample["x"], dtype=torch.int64),
                edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
            )
            for sample in batch
        ]
        # for reagent prediction
        list_additional_graphs = [
            Data(
                x=torch.tensor(sample["additional_x"], dtype=torch.int64),
                edge_index=torch.tensor(
                    sample["additional_edge_index"], dtype=torch.int64
                ),
                edge_attr=torch.tensor(
                    sample["additional_edge_attr"], dtype=torch.int64
                ),
            )
            for sample in batch
        ]

        prompt_text = self.select_mol_representation(
            prompt_text, mol_representation=self.mol_representation
        )

        if not self.train and self.args.eval_modality_util in [
            "string",
            "graph",
        ]:
            shuffled_idx = []
            # shuffle the selfies_idx, guarantee that the selfies_idx is not in order
            for i in range(len(list_selfies)):
                idxs = np.random.choice(
                    range(len(list_selfies)), size=2, replace=False
                ).tolist()
                if i in idxs:
                    idxs.remove(i)
                shuffled_idx.append(idxs[0])

            if self.args.eval_modality_util == "string":
                processed_selfies = [list_selfies[i] for i in shuffled_idx]
                for i in range(len(prompt_text)):
                    assert (
                        list_selfies[i] in prompt_text[i]
                    ), f"{list_selfies[i]} not in {prompt_text[i]}"
                    prompt_text[i] = prompt_text[i].replace(
                        list_selfies[i], processed_selfies[i]
                    )

            if self.args.eval_modality_util == "graph":
                list_graphs = [list_graphs[i] for i in shuffled_idx]
                list_additional_graphs = [
                    list_additional_graphs[i] for i in shuffled_idx
                ]

        if self.args.selfies_enumeration:
            processed_selfies = [
                self.enumerate_selfies(list_selfies[i])
                for i in range(len(list_selfies))
            ]
            for i in range(len(prompt_text)):
                assert (
                    list_selfies[i] in prompt_text[i]
                ), f"{list_selfies[i]} not in {prompt_text[i]}"
                prompt_text[i] = prompt_text[i].replace(
                    list_selfies[i], processed_selfies[i]
                )
                list_selfies = processed_selfies

        if self.apply_molpo:
            if self.train:
                self.reject_cardinal = self.current_epoch
            else:
                self.reject_cardinal = 0

            prompt_text_reject = prompt_text.copy()

            if self.args.apply_preference_system_prompt:
                for i in range(len(prompt_text_reject)):
                    preference_system_prompt = "In the following problems, molecular graph is either accurate or inaccurate. Your predictions should be based primarily on careful understanding of the provided graph."
                    prompt_text_reject[i] = re.sub(
                        r"(?<=\[INST\]).*(?=\n\n)",
                        preference_system_prompt,
                        prompt_text_reject[i],
                    )

            prompt_text = prompt_text + prompt_text_reject * (
                self.args.molpo_batch_division - 1
            )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                reject_target_text = [sample[f"{self.reject_cardinal}-th_rejected_target_text"] for sample in batch]
                target_text = target_text + reject_target_text
            else:
                target_text = target_text * self.args.molpo_batch_division
            tasks = tasks * self.args.molpo_batch_division
            task_names = task_names * self.args.molpo_batch_division

            if "graph" in self.mol_representation:
                list_rejected_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_index"],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_attr"],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]
                # for reagent prediction
                list_rejected_additional_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_additional_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_index"
                            ],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_attr"
                            ],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]

                list_graphs = (
                    list_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_graphs
                )
                list_additional_graphs = (
                    list_additional_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_additional_graphs
                )

        # address <mol> token in prompt_text, for the case of using graph modality
        if self.projector_type == "mlp" and "graph" in self.mol_representation:
            for i in range(len(prompt_text)):
                if task_names[i] in ["reagent_prediction"]:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile(r"(<mol>)+(?=</GRAPH>\|>>\|)")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

                    num_additional_nodes_in_graph = list_additional_graphs[i].x.size(0)
                    num_additional_nodes_mol = "<mol>" * num_additional_nodes_in_graph
                    additional_mol_tokens_pattern = re.compile(
                        r"(?<=\|>>\|<GRAPH>)(<mol>)+"
                    )
                    assert additional_mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = additional_mol_tokens_pattern.sub(
                        num_additional_nodes_mol, prompt_text[i]
                    )
                elif task_names[i] in TEXT2MOL_BENCHMARKS:
                    # there is no input <mol> token
                    pass
                else:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile("(<mol>)+")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

        self.tokenizer.padding_side = "left"
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        target_tokenized = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        full_input_ids = [
            p + t
            for p, t in zip(
                prompt_tokenized["input_ids"], target_tokenized["input_ids"]
            )
        ]
        full_attention_mask = [
            p + t
            for p, t in zip(
                prompt_tokenized["attention_mask"], target_tokenized["attention_mask"]
            )
        ]

        prompt_length = [len(p) for p in prompt_tokenized["input_ids"]]
        full_input_ids = [f_ids[: self.max_length] for f_ids in full_input_ids]
        full_attention_mask = [
            f_ids[: self.max_length] for f_ids in full_attention_mask
        ]

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p for p in prompt_tokenized["input_ids"]],
                    "attention_mask": [p for p in prompt_tokenized["attention_mask"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

            self.tokenizer.padding_side = "right"
            gen_features = self.tokenizer.pad(
                {
                    "input_ids": [t for t in target_tokenized["input_ids"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            gen_features.input_ids = gen_features.input_ids.masked_fill(
                gen_features.input_ids == self.tokenizer.pad_token_id, -100
            )
            features["gen_labels"] = gen_features.input_ids

            input_mol_strings_tokenized = self.tokenizer(
                input_mol_strings,
                truncation=False,
                max_length=self.max_length,
                padding=True,
                return_tensors=return_tensors,
                add_special_tokens=False,
            )

            features["input_mol_strings"] = input_mol_strings_tokenized.input_ids

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized["input_ids"]):
            label = target
            if prompt_length[i] >= self.max_length:
                continue
            else:
                len_label = min(len(label), self.max_length - prompt_length[i])
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        labels_ids = labels_ids.masked_fill(
            labels_ids == self.tokenizer.pad_token_id, -100
        )
        features["labels"] = labels_ids
        if self.apply_molpo:
            molpo_labels_ids = labels_ids.clone()
            for molpo_mask_id in self.tokenizer.molpo_mask_ids:
                molpo_labels_ids = molpo_labels_ids.masked_fill(
                    molpo_labels_ids == molpo_mask_id, -100
                )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                num_chosen = molpo_labels_ids.shape[0] // self.args.molpo_batch_division
                chosen_molpo_labels_ids = molpo_labels_ids.clone()[:num_chosen]
                reject_molpo_labels_ids = molpo_labels_ids.clone()[num_chosen:]

                chosen_molpo_labels_ids = chosen_molpo_labels_ids.masked_fill(
                    chosen_molpo_labels_ids == reject_molpo_labels_ids, -100
                )
                molpo_labels_ids = torch.cat(
                    (chosen_molpo_labels_ids, reject_molpo_labels_ids), dim=0
                )
            features["molpo_labels"] = molpo_labels_ids

        assert (
            features.input_ids.size(1) <= self.max_length
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > self.max_length={self.max_length}"
        assert (
            features.labels.size(1) <= self.max_length
        ), f"features.labels.size(1)={features.labels.size(1)} > self.max_length={self.max_length}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        if "graph" in self.mol_representation:
            graphs = self.graph_collator(list_graphs)
            additional_graphs = self.graph_collator(list_additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs
            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    features["prompt_input_ids"].clone().detach() # .clone().detach() 수정됨
                    == self.tokenizer.mol_token_id
                )

        return features


def random_noise_selfies(selfies, tokenizer, sl_noise_ratio=0.3):
    selfies_ids = tokenizer.encode(selfies, add_special_tokens=False)
    total_selfies_token_ids = tokenizer.selfies_token_ids
    num_ids_to_replace = int(sl_noise_ratio * len(selfies_ids))
    replacing_random_ids = np.random.choice(
        total_selfies_token_ids, num_ids_to_replace, replace=True
    )

    # replace selfies_ids with randomly selected total_selfies_token_ids as many as num_ids_to_replace
    position_to_replace = np.random.choice(
        len(selfies_ids), num_ids_to_replace, replace=False
    )
    noised_selfies_ids = copy.deepcopy(selfies_ids)
    for i, replance_idx in enumerate(position_to_replace):
        noised_selfies_ids[replance_idx] = replacing_random_ids[i]

    noised_selfies = tokenizer.decode(
        noised_selfies_ids, skip_special_tokens=True
    ).replace(" ", "")
    return noised_selfies
