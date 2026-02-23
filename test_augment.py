#!/usr/bin/env python3
"""
Quick test script for augment_dataset.py
Tests the map function with a small subset before running the full pipeline
"""

from datasets import load_from_disk
from augment_dataset import map_by_substructure_replacement
from functools import partial
import os
import random

def test_augmentation(num_proc=112, test_size=100000):
    # Load dataset
    data_dir = "/home/jovyan/CHJ/Mol-LLM_Custom/dataset/train_official"
    dataset_path = os.path.join(
        data_dir,
        "GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_train_512_Truncation"
    )

    print("[TEST] Loading dataset...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Dataset loaded: {len(dataset):,} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Take only first N samples for testing
    test_dataset = dataset.select(range(min(test_size, len(dataset))))
    print(f"✓ Using {len(test_dataset)} samples for testing")

    # Test with map function
    print(f"\n[TEST] Testing map function with num_proc={num_proc}...")
    try:
        map_func = partial(
            map_by_substructure_replacement,
            replace_ratio=0.3,
            num_rejected_graphs=6
        )

        # Use specified num_proc
        test_mapped = test_dataset.map(
            map_func,
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Testing augmentation on small subset"
        )
        print(f"✓ Map completed successfully")

    except Exception as e:
        print(f"✗ Map function failed: {type(e).__name__}: {e}")
        return False

    # Check for failed samples
    print("\n[TEST] Checking for failed samples...")
    if '_processing_failed' in test_mapped.column_names:
        failed_count = sum(1 for x in test_mapped if x.get('_processing_failed', False))
        print(f"  Failed samples: {failed_count}/{len(test_mapped)}")

        # Filter out failed samples
        test_filtered = test_mapped.filter(lambda x: not x.get('_processing_failed', False))
        print(f"✓ Filtered dataset: {len(test_filtered)} samples")

        # Remove marker column
        test_filtered = test_filtered.remove_columns('_processing_failed')
        print(f"✓ Marker column removed")
    else:
        print("✗ _processing_failed column not found!")
        return False

    # Check output structure
    print("\n[TEST] Checking output structure...")
    sample = test_filtered[0]
    required_fields = [
        '0-th_rejected_x', '0-th_rejected_edge_index', '0-th_rejected_edge_attr',
        '0-th_additional_rejected_x', '0-th_additional_rejected_edge_index', '0-th_additional_rejected_edge_attr'
    ]

    missing_fields = [f for f in required_fields if f not in sample or sample[f] is None]
    if missing_fields:
        print(f"✗ Missing fields: {missing_fields}")
        return False
    else:
        print(f"✓ All required fields present")

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Ready for full pipeline!")
    print("="*60)
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=112, help="Number of processes for map")
    parser.add_argument("--test_size", type=int, default=100000, help="Number of samples to test")
    args = parser.parse_args()

    random.seed(42)
    success = test_augmentation(num_proc=args.num_proc, test_size=args.test_size)
    exit(0 if success else 1)
