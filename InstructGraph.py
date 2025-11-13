# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SMolInstruct: A Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset for Small Molecules"""


import json
import os

import datasets

from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
# _CITATION = """\
# @article{yu2024llasmol,
#     title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
#     author={Botao Yu and Frazier N. Baker and Ziqi Chen and Xia Ning and Huan Sun},
#     journal={arXiv preprint arXiv:2402.09391},
#     year={2024}
# }
# """

# Add description of the dataset here
# You can copy an official description
# _DESCRIPTION = """\
# SMolInstruct is a large-scale instruction tuning dataset for chemistry tasks and centers around small molecules. It contains a total of 14 chemistry tasks and over 3 million samples. It is designed to be large-scale, comprehensive, and high-quality.
# """

# Add a link to an official homepage for the dataset here
# _HOMEPAGE = "https://osu-nlp-group.github.io/LLM4Chem/"

# Add the licence for the dataset here if you can find it
_LICENSE = "cc-by-4.0"

# Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }


TrainTASKS = (    
    # name conversion
    'smol-name_conversion-i2s',
    'smol-name_conversion-i2f',
    'smol-name_conversion-s2f',
    'smol-name_conversion-s2i',
    
    # propertry classification
    'smol-property_prediction-bbbp',
    'smol-property_prediction-clintox',
    'smol-property_prediction-hiv',
    'smol-property_prediction-sider',
    'bace',
    # 'tox21',
    # 'toxcast',
    
    # propertry regression
    'smol-property_prediction-esol',
    'smol-property_prediction-lipo',
    'qm9_homo',
    'qm9_lumo',
    'qm9_homo_lumo_gap',
    'qm9_dipole_moment',
    'qm9_isotropic_polarizability',
    'qm9_electronic_spatial_extent',
    'qm9_zero_point_vibrational_energy',
    'qm9_heat_capacity_298K',
    'qm9_internal_energy_298K',
    'qm9_enthalpy_298K',
    'qm9_free_energy_298K',
    
    'forward_reaction_prediction',
    'smol-forward_synthesis',
    'retrosynthesis',
    'smol-retrosynthesis',
    'reagent_prediction',
    
    'chebi-20-text2mol',
    'smol-molecule_generation',
    'chebi-20-mol2text',
    'smol-molecule_captioning',
)

TestTASKS = (
    # propertry classification
    'smol-property_prediction-bbbp',
    'smol-property_prediction-clintox',
    'smol-property_prediction-hiv',
    'smol-property_prediction-sider',
    'bace',
    # 'tox21',
    # 'toxcast',
    
    # propertry regression
    'smol-property_prediction-esol',
    'smol-property_prediction-lipo',
    'qm9_homo',
    'qm9_lumo',
    'qm9_homo_lumo_gap',
    
    'smol-forward_synthesis',
    'forward_reaction_prediction',
    'smol-retrosynthesis',
    'retrosynthesis',
    'reagent_prediction',
    
    'chebi-20-text2mol',
    'smol-molecule_generation',
    'chebi-20-mol2text',
    'smol-molecule_captioning',
)

Task2Category = {    
    # name conversion
    'smol-name_conversion-i2s': 'smol-name_conversion-i2s',
    'smol-name_conversion-i2f': 'smol-name_conversion-i2f',
    'smol-name_conversion-s2f': 'smol-name_conversion-s2f',
    'smol-name_conversion-s2i': 'smol-name_conversion-s2i',
    
    # propertry classification
    'smol-property_prediction-bbbp': 'classification',
    'smol-property_prediction-clintox': 'classification',
    'smol-property_prediction-hiv': 'classification',
    'smol-property_prediction-sider': 'classification',
    'bace': 'classification',
    # 'tox21': 'classification',
    # 'toxcast': 'classification',
    
    # propertry regression
    'smol-property_prediction-esol': 'regression',
    'smol-property_prediction-lipo': 'regression',
    'qm9_homo': 'regression',
    'qm9_lumo': 'regression',
    'qm9_homo_lumo_gap': 'regression',
    'qm9_dipole_moment': 'regression',
    'qm9_isotropic_polarizability': 'regression',
    'qm9_electronic_spatial_extent': 'regression',
    'qm9_zero_point_vibrational_energy': 'regression',
    'qm9_heat_capacity_298K': 'regression',
    'qm9_internal_energy_298K': 'regression',
    'qm9_enthalpy_298K': 'regression',
    'qm9_free_energy_298K': 'regression',
    
    'smol-forward_synthesis': 'reaction_prediction',
    'forward_reaction_prediction': 'reaction_prediction',
    'smol-retrosynthesis': 'reaction_prediction',
    'retrosynthesis': 'reaction_prediction',
    'reagent_prediction': 'reaction_prediction',
    
    'chebi-20-text2mol': 'mol_generation',
    'smol-molecule_generation': 'mol_generation',
    'chebi-20-mol2text': 'mol_captioning',
    'smol-molecule_captioning': 'mol_captioning',
}

default_system_prompt = 'You are a helpful assistant for molecular chemistry, \
to address tasks including molecular property classification, \
molecular property regression, chemical reaction prediction, \
molecule captioning, molecule generation . \n\n'


class InstructGraphDatasetConfig(datasets.BuilderConfig):
    def __init__(
        self, 
        tasks=None, 
        sample_group='instruction_tuning', 
        insert_core_tags=True, 
        use_smiles=False,
        use_test_subset=False,
        use_first=None, 
        system_prompt=None,
        **kwargs
    ):
        """BuilderConfig for MyDataset
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(InstructGraphDatasetConfig, self).__init__(
            **kwargs,
        )
        if tasks is None:
            # tasks = TASKS
            tasks = TrainTASKS
            test_tasks = TestTASKS
        else:
            tasks = set(tasks)
            all_tasks = set(TrainTASKS)
            assert len(tasks - all_tasks) == 0, 'Unsupported task(s): {tasks}'.format(tasks=(tasks - all_tasks))
            test_tasks = tasks & set(TestTASKS)
            
        self.train_tasks = tasks
        self.test_tasks = test_tasks
        print('train_tasks:', self.train_tasks)
        
        self.sample_group = sample_group
        self.insert_core_tags = insert_core_tags
        self.use_smiles = use_smiles
        if 'split' in kwargs:
            self.split = kwargs['split']
        else:
            self.split = None
        self.use_test_subset = use_test_subset
        if use_first is not None:
            assert use_first > 0, "use_first must be a positive integer."
            use_first = int(use_first)
        self.use_first = use_first
        self.system_prompt = system_prompt if system_prompt is not None else default_system_prompt


import torch
from torch_geometric.data import Data
from ogb.utils import smiles2graph
import selfies as sf

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
    # graph = smiles2data(smiles)


class InstructGraph(datasets.GeneratorBasedBuilder):
    """SMolInstruct: A large-scale chemistry instruction tuning dataset."""

    VERSION = datasets.Version("1.2.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = InstructGraphDatasetConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="instruction_tuning", version=VERSION, description="Default set for instruction tuning."),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]
    # BUILDER_CONFIGS = [
    #     CheMIDatasetConfig(
    #         name='instruction',
    #         tasks=TASKS,
    #         sample_group='instruction_tuning',
    #         description="Molecule instructions.",
    #     ),
    #     CheMIDatasetConfig(
    #         name='raw',
    #         tasks=TASKS,
    #         sample_group=None,
    #         description="Molecule raw data.",
    #     ),
    # ]

    # DEFAULT_CONFIG_NAME = "instruction"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "raw_input": datasets.Value("string"),
                "raw_output": datasets.Value("string"),
                "split": datasets.Value("string"),
                "task": datasets.Value("string"),
                'input_core_tag_left': datasets.Value("string"),
                'input_core_tag_right': datasets.Value("string"),
                'output_core_tag_left': datasets.Value("string"),
                'output_core_tag_right': datasets.Value("string"),
                'target': datasets.Value("string"),
                
                # TODO check the type of the graph data
                # graph data
                "x": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "edge_index": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "edge_attr": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                
                "additional_x": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "additional_edge_index": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "additional_edge_attr": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            }
        )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            # description=_DESCRIPTION,
            
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # # Homepage of the dataset for documentation
            # homepage=_HOMEPAGE,
            
            # License for the dataset if available
            license=_LICENSE,
            
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        root = dl_manager.download_and_extract('./data.zip')
        print("root:", root)
        sample_group = self.config.sample_group
        insert_core_tags = self.config.insert_core_tags
        use_smiles = self.config.use_smiles
        use_test_subset = self.config.use_test_subset
        use_first = self.config.use_first

        print('train tasks:\n', self.config.train_tasks)
        print('val tasks:\n', self.config.test_tasks)
        print('test tasks:\n', self.config.test_tasks)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "train",
                    "tasks": self.config.train_tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_smiles": use_smiles,
                    "use_test_subset": False,
                    "use_first": use_first,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "val",
                    "tasks": self.config.test_tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_smiles": use_smiles,
                    "use_test_subset": False,
                    "use_first": use_first,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "root": root,
                    "sample_group": sample_group,
                    "split": "test",
                    "tasks": self.config.test_tasks,
                    "insert_core_tags": insert_core_tags,
                    "use_smiles": use_smiles,
                    "use_test_subset": use_test_subset,
                    "use_first": use_first,
                },
            ),
        ]
    
    def _generate_instruction_examples(self, root, sample_group, split, tasks, insert_core_tags, use_smiles, use_test_subset, use_first):
        key = 0

        if split == 'test' and use_test_subset is True:
            real_split = 'test_subset'
        else:
            real_split = split

        # for task in tasks:
        for task in tqdm(tasks):
            print('\ntask:', task)
            with open(os.path.join(root, 'sample', sample_group, real_split, task + '.json'), 'r') as fs:
                sample_record = json.load(fs)
                assert sample_record['task'] == task, (sample_record['task'], task, os.path.join(root, 'sample', sample_group, real_split, task + '.json'))
                assert sample_record['split'] == real_split
                template_name = sample_record['template_name']
                samples = sample_record['samples']

            with open(os.path.join(root, 'template', template_name, task + '.json'), 'r') as f:
                templates = json.load(f)
            if use_smiles:
                raise NotImplementedError('SELFIES to SMILES conversion is not implemented yet.')
                
                for template in templates:
                    input_template = template['input']
                    output_template = template['output']
                    input_template = input_template.replace("SELFIES", "SMILES")
                    output_template = output_template.replace("SELFIES", "SMILES")
                    template['input'] = input_template
                    template['output'] = output_template

            data = []
            with open(os.path.join(root, 'raw_smiles' if use_smiles else 'raw', split, task + '.jsonl'), 'r') as fr:
                for line in fr:
                    item = json.loads(line)
                    data.append(item)

            with open(os.path.join(root, 'core_tag', Task2Category[task] + '.json'), 'r') as f:
                core_tags = json.load(f)
            input_core_tag_left = core_tags['input'][0]
            input_core_tag_right = core_tags['input'][1]
            
            if use_smiles and input_core_tag_left == '<SELFIES>':
                assert input_core_tag_right == '</SELFIES>'
                input_core_tag_left = '<SMILES>'
                input_core_tag_right = '</SMILES>'
            output_core_tag_left = core_tags['output'][0]
            output_core_tag_right = core_tags['output'][1]
            if use_smiles and output_core_tag_left == '<SELFIES>':
                assert output_core_tag_right == '</SELFIES>'
                output_core_tag_left = '<SMILES>'
                output_core_tag_right = '</SMILES>'
            
            for sample_item in (samples if use_first is None else samples[:use_first]):
                # TODO transformer number to token for regression task
                
                
                try:
                    data_item = data[sample_item['idx']]
                except IndexError:
                    raise IndexError('In %s for %s, data index exceeds the number of samples. The data size is %d, while the index is %d.' % (real_split, task, len(data), sample_item['idx']))
                assert data_item['task'] == task
                assert data_item['split'] == split
                template_id = sample_item['template_id']
                template = templates[template_id]
                input_template = template['input']
                # output_template = template['output']  # it can have more natural language in the output
                output_template = "<OUTPUT>"
                input_data = data_item['input']
                try:
                    if task in ['chebi-20-text2mol', 
                                'smol-molecule_generation', 
                                'smol-name_conversion-i2s',
                                'smol-name_conversion-i2f',
                                ]:
                        input_graph = [smiles2data('CC'), smiles2data('CC')]
                    else:
                        # no_space_input = input_data.replace(' ', '')
                        if task == 'reagent_prediction':
                            assert '|>>|' in input_data, f'Invalid reagent format: {input_data} needs to have >>'
                            list_selfies = input_data.split('|>>|')
                                                        
                            list_smiles = [sf.decoder(s) for s in list_selfies]
                            input_graph = [smiles2data(s) for s in list_smiles]
                            
                        else:
                            smiles = sf.decoder(input_data)
                            # smiles = sf.decoder(no_space_input)
                            # input_graph = smiles2data(smiles)
                            input_graph = [smiles2data(smiles), smiles2data('CC')]
                            

                except:
                    continue
                
                if insert_core_tags and input_core_tag_left is not None:
                    assert input_core_tag_right is not None
                    if task == 'reagent_prediction':
                        input_data_str = ' |>>| '.join([f'{input_core_tag_left} {s} {input_core_tag_right}' for s in list_selfies])
                    else:
                        input_data_str = '%s %s %s' % (input_core_tag_left, input_data, input_core_tag_right)
                else:
                    input_data_str = input_data
                
                # input_str = input_template.replace('<INPUT>', input_data_str)
                # print(self.config.system_prompt)
                # print('input_template:', input_template)
                # print('input_data_str:', input_data_str)
                # input_str = self.config.system_prompt + input_template.replace('<INPUT>', input_data_str)
                input_str = input_template.replace('<INPUT>', input_data_str)
                output_data = data_item['output']
                if isinstance(output_data, str):
                    target = None
                elif isinstance(output_data, dict):
                    target = sample_item['target']
                    output_data = output_data[target]
                else:
                    raise ValueError
                if insert_core_tags and output_core_tag_left is not None:
                    assert output_core_tag_right is not None
                    output_data_str = '%s %s %s' % (output_core_tag_left, output_data, output_core_tag_right)
                else:
                    output_data_str = output_data
                output_str = output_template.replace('<OUTPUT>', output_data_str)
                output_sample = {
                    # TODO tokenize 하고 이것저것 하는데, 필요한거 뭔지 정리 후 수정
                    # 그리고 여기서 할지, 전처리 단계에서 할지, collator에서 할 지
                    
                    
                    'input': input_str,
                    'output': output_str,
                    'raw_input': input_data,
                    'raw_output': output_data,
                    'task': task,
                    'split': real_split,
                    'input_core_tag_left': input_core_tag_left,
                    'input_core_tag_right': input_core_tag_right,
                    'output_core_tag_left': output_core_tag_left,
                    'output_core_tag_right': output_core_tag_right,
                    'target': target,  # if there are multiple targets 
                    
                    # graph data
                    "x": input_graph[0].x.tolist(),
                    "edge_index": input_graph[0].edge_index.tolist(),
                    "edge_attr": input_graph[0].edge_attr.tolist(),
                    # for reagent_prediction
                    "additional_x": input_graph[1].x.tolist(),
                    "additional_edge_index": input_graph[1].edge_index.tolist(),
                    "additional_edge_attr": input_graph[1].edge_attr.tolist(),
                    
                }
                yield key, output_sample

                key += 1

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, *args, **kwargs):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        return self._generate_instruction_examples(*args, **kwargs)
