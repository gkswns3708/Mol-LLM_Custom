from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    REACTION_BENCHMARKS,
)
import model.added_tokens as added_tokens
import re
from Levenshtein import distance as lev


def caption_evaluate(predictions, targets, tokenizer, prompts, input_mol_strings):
    references = []
    hypotheses = []
    ref_sentences = []
    hyp_sentences = []
    failure_idxs = []

    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    patterns = {
        "DESCRIPTION": {
            "dual_side": re.compile(r"(?<=<DESCRIPTION>).*?(?=</DESCRIPTION>)"),
            "left_side": re.compile(r"(?<=<DESCRIPTION>).*"),
        },
        "IUPAC": {
            "dual_side": re.compile(r"(?<=<IUPAC>).*?(?=</IUPAC>)"),
            "left_side": re.compile(r"(?<=<IUPAC>).*"),
        },
        "MOLFORMULA": {
            "dual_side": re.compile(r"(?<=<MOLFORMULA>).*?(?=</MOLFORMULA>)"),
            "left_side": re.compile(r"(?<=<MOLFORMULA>).*"),
        },
    }

    for i in range(len(targets)):
        target = targets[i]
        prediction = predictions[i]

        pattern = None
        for key, matching_pattern in patterns.items():
            if matching_pattern["left_side"].search(targets[i]):
                pattern = matching_pattern
                break
        if pattern is None:
            print(targets[i])
            continue
        assert pattern is not None

        if pattern["dual_side"].search(target):
            ref = pattern["dual_side"].search(target).group()
        else:
            ref = pattern["left_side"].search(target).group()
        ref_sentences.append(ref)
        ref_tokens = tokenizer.tokenize(ref)

        try:
            if pattern["dual_side"].search(prediction):
                pred = pattern["dual_side"].search(prediction).group()
            else:
                pred = pattern["left_side"].search(prediction).group()
            hyp_sentences.append(pred)
            pred_tokens = tokenizer.tokenize(pred)

            references.append([ref_tokens])
            hypotheses.append(pred_tokens)

        except:
            failure_idxs.append(i)
            pred = None
            pred_tokens = None

    if hypotheses:
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        bleu2 *= 100
        bleu4 *= 100

        for ref, hyp in tqdm(zip(references, hypotheses)):
            mscore = meteor_score(ref, hyp)
            meteor_scores.append(mscore)

        _meteor_score = np.mean(meteor_scores)
        _meteor_score *= 100

        for ref_sen, hyp_sen in tqdm(zip(ref_sentences, hyp_sentences)):
            lscore = scorer.score(hyp_sen, ref_sen)
            rouge_scores.append(lscore)

        rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
        rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
        rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100

    else:
        bleu2 = 0
        bleu4 = 0
        _meteor_score = 0
        rouge_1 = 0
        rouge_2 = 0
        rouge_l = 0

    evaluation_results = {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": _meteor_score,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies


def molecule_evaluate(
    predictions, targets, tokenizer, prompts, input_mol_strings, morgan_r=2
):
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    levs = []
    morgan_r = 2

    failure_idxs = []
    exact_matches = []
    ref_selfies_list = []
    ref_smiles_list = []
    pred_selfies_list = []
    pred_smiles_list = []

    for i in tqdm(range(len(targets))):
        target = targets[i].replace(" ", "")
        prediction = predictions[i].replace(" ", "")

        try:
            if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", target):
                target_selfies = re.search(
                    r"(?<=<SELFIES>).*?(?=</SELFIES>)", target
                ).group()
            else:
                target_selfies = re.search(r"(?<=<SELFIES>).*", target).group()
            target_smiles = selfies.decoder(target_selfies)
            target_mol = Chem.MolFromSmiles(target_smiles)
            target_canonical_smiles = Chem.CanonSmiles(target_smiles)
            target_canonical_selfies = selfies.encoder(target_canonical_smiles)

            if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction) is not None:
                prediction_selfies = re.search(
                    r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction
                ).group()
            else:
                prediction_selfies = re.search(r"(?<=<SELFIES>).*", prediction).group()

            prediction_selfies = prediction_selfies.split("<SELFIES>")[-1]
            prediction_selfies = prediction_selfies.split("</SELFIES>")[0]

            assert (
                "<SELFIES>" not in prediction_selfies
                and "</SELFIES>" not in prediction_selfies
            )

            prediction_smiles = selfies.decoder(prediction_selfies)
            prediction_mol = Chem.MolFromSmiles(prediction_smiles)
            prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)
            prediction_canonical_selfies = selfies.encoder(prediction_canonical_smiles)

            exact_matches.append(
                Chem.MolToInchi(target_mol) == Chem.MolToInchi(prediction_mol)
            )
        except:
            failure_idxs.append(i)
            prediction_mol = None
            # [수정됨] 리스트 전체가 아니라 현재 인덱스 i의 값만 출력
            print(f"\n=== [Conversion Error at index {i}] ===") 
            print(f"Target      : {target}")
            print(f"Prediction  : {prediction}")
            # input_mol_strings가 리스트인지 확인하고 인덱싱
            if isinstance(input_mol_strings, list) and len(input_mol_strings) > i:
                print(f"Input Mol   : {input_mol_strings[i]}") 
            else:
                print(f"Input Mol   : {input_mol_strings}") # 리스트가 아닐 경우 대비
            
            if isinstance(prompts, list) and len(prompts) > i:
                print(f"Prompt      : {prompts[i]}")
            else:
                print(f"Prompt      : {prompts}")
            print("======================================\n")
            continue

        if prediction_mol is not None:

            levs.append(lev(target_canonical_smiles, prediction_canonical_smiles))

            pred_selfies = tokenizer.tokenize(prediction_canonical_selfies)
            pred_smiles = tokenizer.tokenize(prediction_canonical_smiles)
            pred_selfies_list.append(pred_selfies)
            pred_smiles_list.append(pred_smiles)

            ref_selfies = tokenizer.tokenize(target_canonical_selfies)
            ref_smiles = tokenizer.tokenize(target_canonical_smiles)
            ref_selfies_list.append([ref_selfies])
            ref_smiles_list.append([ref_smiles])

            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(
                    MACCSkeys.GenMACCSKeys(target_mol),
                    MACCSkeys.GenMACCSKeys(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            RDK_sims.append(
                DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(target_mol),
                    Chem.RDKFingerprint(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            morgan_sims.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(target_mol, morgan_r),
                    AllChem.GetMorganFingerprint(prediction_mol, morgan_r),
                )
            )

    validity_ratio = 1 - len(failure_idxs) / len(predictions)
    MACCS_sim = np.mean(MACCS_sims)
    RDK_sim = np.mean(RDK_sims)
    morgan_sim = np.mean(morgan_sims)
    exact_match_ratio = np.mean(exact_matches)
    levenshtein_score = np.mean(levs)

    if pred_smiles_list:
        bleu_smiles = corpus_bleu(
            ref_smiles_list, pred_smiles_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_smiles *= 100
    else:
        bleu_smiles = 0

    if pred_selfies_list:
        bleu_selfies = corpus_bleu(
            ref_selfies_list, pred_selfies_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_selfies *= 100
    else:
        bleu_selfies = 0

    results = {
        "validity_ratio": validity_ratio,
        "MACCS_FTS": MACCS_sim,
        "RDK_FTS": RDK_sim,
        "morgan_FTS": morgan_sim,
        "exact_match_ratio": exact_match_ratio,
        "levenshtein_score": levenshtein_score,
        "bleu_smiles": bleu_smiles,
        "bleu_selfies": bleu_selfies,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return results, failed_cases


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    """
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    """
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full(
            (sum_dim0, max_dim1, tensor_list[0].shape[-1]),
            fill_value=fill_value,
            device=device,
            dtype=dtype,
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full(
            (sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()


def get_task_specific_list(predictions, targets, tasks, prompts, input_mol_strings):
    unique_tasks = list(set(tasks))
    task_specific_predictions = {t: [] for t in unique_tasks}
    task_specific_targets = {t: [] for t in unique_tasks}
    task_specific_prompts = {t: [] for t in unique_tasks}
    task_specific_input_mol_strings = {t: [] for t in unique_tasks}
    for i, t in enumerate(tasks):
        task_specific_predictions[t].append(predictions[i])
        task_specific_targets[t].append(targets[i])
        task_specific_prompts[t].append(prompts[i])
        task_specific_input_mol_strings[t].append(input_mol_strings[i])
    return (
        task_specific_predictions,
        task_specific_targets,
        task_specific_prompts,
        task_specific_input_mol_strings,
    )


# correspond to all the tasks other than classification
def per_device_evaluate(
    predictions,
    targets,
    tasks,
    prompts,
    input_mol_strings,
    tokenizer,
    total_task_subtask_pairs,
):
    # get unique items from all_tasks
    unique_tasks = list(set(tasks))
    # remove tasks_to_be_removed
    tasks_to_be_removed = [
        "smol-name_conversion-i2f",
        "smol-name_conversion-s2f",
        "smol-name_conversion-i2s",
        "smol-name_conversion-s2i",
    ] + CLASSIFICATION_BENCHMARKS

    unique_tasks = [
        t for t in unique_tasks if t.split("/")[0] not in tasks_to_be_removed
    ]

    evaluation_results = {task: dict() for task in unique_tasks}

    (
        task_specific_predictions,
        task_specific_targets,
        task_specific_prompts,
        task_specific_input_mol_strings,
    ) = get_task_specific_list(predictions, targets, tasks, prompts, input_mol_strings)
    failed_cases = {
        "predictions": [],
        "targets": [],
        "prompts": [],
        "tasks": [],
        "input_mol_strings": [],
    }

    # initialize evaluation results for all tasks with null values
    # necessary to make uniform shape of evaluation results
    for t in total_task_subtask_pairs:
        if t.split("/")[0] in tasks_to_be_removed:
            continue

        task_name = t.split("/")[0]
        null_value = 0
        if task_name in REGRESSION_BENCHMARKS:
            results = {
                "mae": null_value,
                "mse": null_value,
                "rmse": null_value,
                "failure_rate": null_value,
            }
        elif (
            task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results = {
                "validity_ratio": null_value,
                "MACCS_FTS": null_value,
                "RDK_FTS": null_value,
                "morgan_FTS": null_value,
                "exact_match_ratio": null_value,
                "levenshtein_score": null_value,
                "bleu_smiles": null_value,
                "bleu_selfies": null_value,
            }
        elif task_name in MOL2TEXT_BENCHMARKS:
            results = {
                "bleu2": null_value,
                "bleu4": null_value,
                "rouge1": null_value,
                "rouge2": null_value,
                "rougeL": null_value,
                "meteor": null_value,
            }
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = 0
        evaluation_results[t] = results

    for t in task_specific_predictions.keys():
        if t.split("/")[0] in tasks_to_be_removed:
            continue

        task_predictions = task_specific_predictions[t]
        task_targets = task_specific_targets[t]
        task_prompts = task_specific_prompts[t]
        task_input_mol_strings = task_specific_input_mol_strings[t]
        task_name = t.split("/")[0]

        if task_name in REGRESSION_BENCHMARKS:
            results, _failed_cases = regression_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
            )
        elif (
            task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results, _failed_cases = molecule_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
            )
        elif task_name in MOL2TEXT_BENCHMARKS:
            results, _failed_cases = caption_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
            )
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = len(task_predictions)
        evaluation_results[t] = results

        if task_name not in CLASSIFICATION_BENCHMARKS:
            for k in _failed_cases.keys():
                failed_cases[k].extend(_failed_cases[k])
            failed_cases["tasks"].extend(
                [t.split("/")[0] for _ in range(len(_failed_cases["predictions"]))]
            )

    return evaluation_results, failed_cases


def classification_evaluate(total_labels, total_probs):
    total_preds = total_probs.argmax(dim=-1)

    # Convert tensors to numpy arrays for use with scikit-learn metrics
    total_preds_np = total_preds.numpy()
    total_labels_np = total_labels.numpy()

    # Calculate metrics
    acc = accuracy_score(y_true=total_labels_np, y_pred=total_preds_np)
    f1 = f1_score(y_true=total_labels_np, y_pred=total_preds_np)
    prec = precision_score(y_true=total_labels_np, y_pred=total_preds_np)
    rec = recall_score(y_true=total_labels_np, y_pred=total_preds_np)
    try:
        roc_auc = roc_auc_score(
            y_true=total_labels_np,
            y_score=total_probs[
                :, 1
            ].numpy(),  # Use y_score here because roc_auc_score expects probability scores
        )
    except:
        roc_auc = float("nan")

    evaluation_results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
    }
    return evaluation_results


# correspond to classification tasks
def total_device_evaluate(
    total_labels, total_tasks, total_probs, classification_task_subtask_pairs
):
    evaluation_results = {}

    # initialize evaluation results for classification tasks with null values
    for t in classification_task_subtask_pairs:
        task_name = t.split("/")[0]
        null_value = float("nan")
        results = {
            "accuracy": null_value,
            "f1": null_value,
            "precision": null_value,
            "recall": null_value,
            "roc_auc": null_value,
            "num_instances": 0,
        }
        evaluation_results[t] = results

    unique_tasks = list(set(total_tasks))
    task_specific_labels = {t: [] for t in unique_tasks}
    task_specific_probs = {t: [] for t in unique_tasks}

    for i, t in enumerate(total_tasks):
        task_specific_labels[t].append(total_labels[i])
        task_specific_probs[t].append(total_probs[i])

    for t in task_specific_labels.keys():
        task_probs = task_specific_probs[t]
        task_probs = torch.stack(task_probs, dim=0)
        task_labels = task_specific_labels[t]
        task_labels = torch.stack(task_labels, dim=0)
        task_name = t.split("/")[0]
        if task_name in CLASSIFICATION_BENCHMARKS:
            results = classification_evaluate(
                total_probs=task_probs,
                total_labels=task_labels,
            )
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = len(total_labels)
        evaluation_results[t] = results

    return evaluation_results


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def convert_logit2binary_prob(logits, predictions, tokenizer):
    # WARNING: for specific LLM tokenizer, behaviour might be different
    # below code is for mistral7B tokenizer
    # if you want to use this function for other tokenizer, you should check working, and modify if necessary
    True_token_id = tokenizer.encode("True")[-1]
    False_token_id = tokenizer.encode("False")[-1]

    bos_token, eos_token = added_tokens.BOOL
    boolean_bos_id = tokenizer.encode([bos_token])[-1]

    prediction_position_ids = torch.zeros(logits.shape[:-1], dtype=torch.bool)
    is_using_prediction_position_ids = torch.zeros(
        (logits.shape[0], 2), dtype=torch.bool
    ).to(logits.device)

    for idx, pred in enumerate(predictions):
        # first, inspect that pred includes boolean tokens
        # second, inspect that there is only one token between boolean tokens
        # third, get position id of the prediction token between the boolean tokens
        pred_token_ids = tokenizer.encode(pred, add_special_tokens=False)
        try:
            assert re.search(
                f"{bos_token}.+{eos_token}", pred
            ).group(), f"pred should be searched by re pattern {bos_token}.+{eos_token}"
            boolean_bos_position = pred_token_ids.index(boolean_bos_id)
            prediction_position_ids[idx, boolean_bos_position + 1] = True
            is_using_prediction_position_ids[idx, :] = True
        except:
            prediction_position_ids[idx, 0] = True
            is_using_prediction_position_ids[idx, :] = False

    true_logits = logits[prediction_position_ids][:, True_token_id]
    false_logits = logits[prediction_position_ids][:, False_token_id]

    total_logits = torch.cat(
        [false_logits.unsqueeze(1), true_logits.unsqueeze(1)], dim=-1
    )
    total_probs = total_logits.softmax(-1)

    total_probs = torch.where(
        is_using_prediction_position_ids,
        total_probs,
        torch.full_like(total_probs, -1),
    )

    total_probs = [p.tolist() for p in total_probs]
    return total_probs


def regression_evaluate(predictions, targets, prompts, input_mol_strings):

    total_labels = []
    total_predictions = []
    failure_idxs = []

    for i in range(len(predictions)):
        label = (
            re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", targets[i])
            .group()
            .replace(" ", "")
        )
        label = label.replace("<|", "").replace("|>", "")
        label = float(label)

        # only calculate metrics if the prediction is a float
        # else, increment the failure count
        try:
            assert (
                "<|.|>" in predictions[i]
            ), f"Prediction should include <|.|> token for proper magnitude of order, but {predictions[i]}"
            prediction = (
                re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", predictions[i])
                .group()
                .replace(" ", "")
            )

            prediction = prediction.replace("<|", "").replace("|>", "")
            prediction = float(prediction)

            total_labels.append(label)
            total_predictions.append(prediction)
        except:
            failure_idxs.append(i)
    failure_rate = len(failure_idxs) / len(predictions)

    # Calculate regression metrics: mae, mse, rmse
    total_labels = np.array(total_labels)
    total_predictions = np.array(total_predictions)

    mae = np.mean(np.abs(total_labels - total_predictions))
    mse = np.mean((total_labels - total_predictions) ** 2)
    rmse = np.mean((total_labels - total_predictions) ** 2) ** 0.5

    evaluation_results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "failure_rate": failure_rate,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases
