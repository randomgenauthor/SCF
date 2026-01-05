#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates semantic cluster IDs for a model's vocabulary using hierarchical clustering.

This script loads a specified Hugging Face model, extracts its token embeddings,
performs hierarchical clustering on them, and saves the resulting cluster assignments
to a file. This output is intended for use with the SSQSelector.

Usage:
    python cluster_embeddings.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --distance_threshold_ratios 0.7 0.8 0.9

This will create a `cluster_ids.pt` file in the current directory.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Perform hierarchical clustering on model token embeddings.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",#"Qwen/Qwen2.5-7B-Instruct",
        help="The Hugging Face model to load embeddings from."
    )
    parser.add_argument(
        "--output_path",
        "--output_prefix",
        type=str,
        default="cluster_ids",
        help="Prefix for the output file(s). Each file will be named <prefix>_ratio_<value>.pt."
    )
    parser.add_argument(
        "--distance_threshold_ratios",
        nargs='+',
        type=float,
        default=None,
        help="One or more relative distance thresholds. If not provided, a default sequence from 0.9 to 1-10^-8 will be used."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on ('cuda' or 'cpu')."
    )
    args = parser.parse_args()

    # If no ratios are provided via command line, use the default sequence.
    if args.distance_threshold_ratios is None:
        # Generate ratios from 0.9, 0.99, ... up to 1 - 10^-8
        ratios_to_process = [0.01*i for i in range(1, 21)]
        print(f"No ratios provided. Using default sequence: {ratios_to_process}")
    else:
        ratios_to_process = args.distance_threshold_ratios

    print(f"Loading model '{args.model_name_or_path}' to extract embeddings...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract embeddings and move to CPU for numpy/scipy processing
    embeddings = model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    vocab_size = embeddings.shape[0]
    print(f"Extracted {vocab_size} embeddings of dimension {embeddings.shape[1]}.")

    print("\nPerforming hierarchical clustering (this may take a while)...")
    # Using 'ward' method, which minimizes the variance of the clusters being merged.
    # Metric is 'euclidean' distance.
    linkage_matrix = linkage(embeddings, method='ward', metric='euclidean')
    print("Linkage matrix calculated.")

    # The last entry in the linkage matrix contains the final merge,
    # and its distance is the maximum distance in the dendrogram.
    max_distance = linkage_matrix[-1, 2]
    print(f"Max merge distance in dendrogram: {max_distance:.4f}")

    for ratio in ratios_to_process:
        if not (0.0 < ratio < 1.0):
            print(f"Skipping invalid ratio {ratio}. Must be between 0.0 and 1.0.", file=sys.stderr)
            continue

        print("-" * 20)
        # Calculate the absolute distance threshold based on the user-provided ratio.
        absolute_threshold = max_distance * ratio
        print(f"Processing for ratio: {ratio} (absolute threshold: {absolute_threshold:.4f})")

        # Form flat clusters from the linkage matrix using the distance threshold.
        cluster_labels = fcluster(linkage_matrix, t=absolute_threshold, criterion='distance')

        # Convert numpy array to a PyTorch tensor (0-based).
        cluster_ids_tensor = torch.from_numpy(cluster_labels - 1).long()

        num_clusters = len(np.unique(cluster_labels))
        print(f"Clustering resulted in {num_clusters} clusters.")

        output_filename = f"{args.output_path}_ratio_{ratio}.pt"
        print(f"Saving cluster IDs to '{output_filename}'...")
        torch.save(cluster_ids_tensor, output_filename)

    print("Done.")

if __name__ == "__main__":
    main()