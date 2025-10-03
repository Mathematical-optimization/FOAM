# condition-number_v2.py (개선된 버전)
"""
Enhanced condition number analysis for Shampoo preconditioners.
Includes bias correction, comprehensive statistics, and improved visualization.
"""

import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
import gc
from typing import Dict, List, Tuple, Any, Optional

# Import common utilities
from shampoo_analysis_utils import (
    CheckpointAnalyzer,
    compute_eigenvalue_statistics,
    diagnose_conditioning,
    export_to_csv,
    print_ill_conditioning_report
)


def compute_condition_number(matrix: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    Compute the condition number of a matrix with improved numerical stability.
    
    Args:
        matrix: Input square matrix
        epsilon: Small regularization value
        
    Returns:
        Condition number (float)
    """
    try:
        matrix = matrix.detach().double()

        # Check if square
        if matrix.shape[0] != matrix.shape[1]:
            return float('inf')

        # Add small regularization
        matrix = matrix + torch.eye(
            matrix.shape[0], 
            dtype=torch.float64, 
            device=matrix.device
        ) * epsilon

        # Compute condition number
        try:
            cond_num = torch.linalg.cond(matrix).item()
        except:
            # Fallback to SVD
            try:
                U, S, V = torch.linalg.svd(matrix)
                cond_num = (S[0] / S[-1]).item() if S[-1] > epsilon else float('inf')
            except:
                return float('inf')

        # Check for NaN/Inf
        if np.isnan(cond_num) or np.isinf(cond_num):
            return float('inf')

        return cond_num

    except Exception as e:
        print(f"  ⚠️ Condition number computation failed: {e}")
        return float('inf')


def apply_bias_correction(matrix: torch.Tensor, beta2: float, step: int) -> torch.Tensor:
    """
    Apply bias correction to the factor matrix.
    
    Args:
        matrix: Factor matrix
        beta2: Exponential moving average parameter
        step: Current training step/epoch
        
    Returns:
        Bias-corrected matrix
    """
    if beta2 < 1.0 and step > 0:
        bias_correction = 1.0 - (beta2 ** step)
        if bias_correction > 1e-9:
            return matrix / bias_correction
    return matrix


def extract_beta2_from_checkpoint(checkpoint: Dict) -> float:
    """Extract beta2 parameter from checkpoint with multiple fallback strategies."""
    
    # Strategy 1: Check shampoo_config
    if 'shampoo_config' in checkpoint:
        beta2 = checkpoint['shampoo_config'].get('beta2', None)
        if beta2 is not None:
            return float(beta2)
    
    # Strategy 2: Check args
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'beta2'):
        return float(checkpoint['args'].beta2)
    
    # Strategy 3: Check optimizer state dict
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
            # Handle both dict and list formats
            if isinstance(opt_state['param_groups'], dict):
                # distributed_state_dict() saves as dict
                param_group = next(iter(opt_state['param_groups'].values()))
            else:
                # Standard state_dict() saves as list
                param_group = opt_state['param_groups'][0]
            
            if 'betas' in param_group:
                betas = param_group['betas']
                if isinstance(betas, (list, tuple)) and len(betas) > 1:
                    return float(betas[1])
    
    # Default value
    print("  ⚠️ Could not find beta2 in checkpoint, using default 0.99")
    return 0.99


def analyze_condition_numbers(
    checkpoint_dir: str,
    output_dir: str,
    memory_efficient: bool = False,
    export_csv: bool = True,
    condition_threshold: float = 1e6
):
    """
    Analyze condition numbers across all checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Directory to save results
        memory_efficient: Use memory-efficient processing
        export_csv: Export data to CSV
        condition_threshold: Threshold for flagging problematic preconditioners
    """
    # Find checkpoints
    checkpoint_files = CheckpointAnalyzer.find_checkpoints(checkpoint_dir)
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in '{checkpoint_dir}'")
        return

    print(f"📊 Analyzing condition numbers from {len(checkpoint_files)} checkpoints...\n")

    # Results storage: results[layer_type][key][metric] = [values]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Additional statistics storage
    all_condition_numbers = []

    # Process each checkpoint
    for epoch, checkpoint_path in checkpoint_files:
        print(f"--- Epoch {epoch} ({os.path.basename(checkpoint_path)}) ---")

        checkpoint = CheckpointAnalyzer.load_checkpoint_safe(checkpoint_path)
        if checkpoint is None:
            continue

        if 'optimizer_state_dict' not in checkpoint:
            print("  ⚠️ No optimizer_state_dict found")
            continue

        param_states = checkpoint['optimizer_state_dict'].get('state', {})
        beta2 = extract_beta2_from_checkpoint(checkpoint)
        print(f"  Using beta2 = {beta2}")

        analyzed_count = 0
        encoder_count = 0
        decoder_count = 0

        for param_name, param_state in param_states.items():
            param_info = CheckpointAnalyzer.parse_param_info(param_name)
            if not param_info:
                continue

            layer_type = param_info['layer_type']
            layer_idx = param_info['layer_idx']
            attn_type = param_info['attn_type']
            proj_name = param_info['proj_name']

            # Process factor matrices
            for state_key, state_value in param_state.items():
                if isinstance(state_key, str) and state_key.startswith('['):
                    try:
                        key_parts = json.loads(state_key)
                        
                        if (isinstance(key_parts, list) and len(key_parts) == 4 and
                            key_parts[1] == 'shampoo' and
                            key_parts[2] == 'factor_matrices' and
                            isinstance(key_parts[3], int)):

                            factor_idx = key_parts[3]

                            if isinstance(state_value, torch.Tensor):
                                if state_value.shape[0] == state_value.shape[1] and state_value.shape[0] > 1:
                                    # Apply bias correction
                                    corrected_matrix = apply_bias_correction(state_value, beta2, epoch)
                                    cond_num = compute_condition_number(corrected_matrix)

                                    if cond_num != float('inf'):
                                        factor_name = 'L' if factor_idx == 0 else 'R'
                                        
                                        # Construct result key
                                        result_key = f"Block{layer_idx}_{attn_type}_{proj_name}_{factor_name}"
                                        
                                        results[layer_type][result_key]['epochs'].append(epoch)
                                        results[layer_type][result_key]['cond_nums'].append(cond_num)
                                        
                                        # Store additional info for statistics
                                        all_condition_numbers.append({
                                            'layer_type': layer_type,
                                            'block': layer_idx,
                                            'attn_type': attn_type,
                                            'projection': proj_name,
                                            'factor': factor_name,
                                            'epoch': epoch,
                                            'condition_number': cond_num
                                        })
                                        
                                        analyzed_count += 1
                                        if layer_type == 'encoder_layers' or layer_type == 'encoder_blocks':
                                            encoder_count += 1
                                        else:
                                            decoder_count += 1

                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

        print(f"  ✓ Analyzed {analyzed_count} attention parameters")
        print(f"    (Encoder: {encoder_count}, Decoder: {decoder_count})")
        
        # Memory cleanup
        del checkpoint
        if memory_efficient:
            gc.collect()

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"📊 ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total unique parameter keys: {sum(len(v) for v in results.values())}")
    print(f"Total condition number measurements: {len(all_condition_numbers)}")
    
    for layer_type, layer_data in results.items():
        print(f"\n{layer_type}:")
        print(f"  - Parameters tracked: {len(layer_data)}")
        
    if not results:
        print("❌ No condition number data collected.")
        return

    # Export to CSV
    if export_csv:
        print(f"\n💾 Exporting data to CSV...")
        export_condition_numbers_to_csv(all_condition_numbers, output_dir)

    # Diagnose problematic preconditioners
    print(f"\n🔍 Diagnosing ill-conditioned preconditioners...")
    diagnose_high_condition_numbers(all_condition_numbers, threshold=condition_threshold)

    # Generate plots
    print(f"\n🎨 Creating visualizations...")
    plot_condition_number_trends(results, output_dir)
    plot_condition_number_statistics(all_condition_numbers, output_dir)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


def export_condition_numbers_to_csv(data: List[Dict], output_dir: str):
    """Export condition number data to CSV."""
    import pandas as pd
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, 'condition_numbers.csv')
    df.to_csv(output_path, index=False)
    print(f"  ✓ Exported to {output_path}")


def diagnose_high_condition_numbers(data: List[Dict], threshold: float = 1e6):
    """Identify and report high condition numbers."""
    
    problematic = [d for d in data if d['condition_number'] > threshold]
    
    if not problematic:
        print(f"  ✅ No condition numbers exceed threshold {threshold:.0e}")
        return
    
    print(f"\n  ⚠️  Found {len(problematic)} measurements exceeding threshold {threshold:.0e}")
    print(f"\n  {'Layer':<15} {'Block':<6} {'Attn':<10} {'Proj':<8} {'Factor':<8} {'Epoch':<8} {'Cond#':<12}")
    print(f"  {'-'*80}")
    
    # Sort by condition number (worst first)
    problematic.sort(key=lambda x: x['condition_number'], reverse=True)
    
    for item in problematic[:20]:  # Show top 20
        print(f"  {item['layer_type']:<15} {item['block']:<6} {item['attn_type']:<10} "
              f"{item['projection']:<8} {item['factor']:<8} {item['epoch']:<8} "
              f"{item['condition_number']:<12.2e}")


def plot_condition_number_trends(results: Dict, output_dir: str):
    """Generate condition number trend plots for each layer type."""
    
    for layer_type, layer_data in results.items():
        if not layer_data:
            continue
        
        # Determine number of unique blocks
        blocks = set()
        for key in layer_data.keys():
            match = re.search(r'Block(\d+)', key)
            if match:
                blocks.add(int(match.group(1)))
        
        max_block = max(blocks) if blocks else 0
        num_blocks = max_block + 1
        
        # Grid size
        rows = max(2, (num_blocks + 1) // 2)
        cols = 2

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows), squeeze=False)
        axes = axes.flatten()

        color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
        marker_map = {'L': 'o', 'R': 's'}
        linestyle_map = {'L': '-', 'R': '--'}

        for block_idx in range(num_blocks):
            ax = axes[block_idx]
            has_data = False

            # Determine attention types in this layer
            attn_types = set()
            for key in layer_data.keys():
                if f'Block{block_idx}_' in key:
                    match = re.search(r'_(self_attn|cross_attn)_', key)
                    if match:
                        attn_types.add(match.group(1))

            for attn_type in sorted(attn_types):
                for proj_type in ['Query', 'Key', 'Value']:
                    for factor_type in ['L', 'R']:
                        key = f"Block{block_idx}_{attn_type}_{proj_type}_{factor_type}"

                        if key in layer_data and layer_data[key]['epochs']:
                            has_data = True
                            epochs = layer_data[key]['epochs']
                            cond_nums = layer_data[key]['cond_nums']

                            label_prefix = f"{attn_type.replace('_attn', '')} " if len(attn_types) > 1 else ""
                            label = f"{label_prefix}{proj_type} ({factor_type})"

                            ax.semilogy(
                                epochs, cond_nums,
                                marker=marker_map[factor_type],
                                linestyle=linestyle_map[factor_type],
                                label=label,
                                color=color_map[proj_type],
                                linewidth=1.5,
                                markersize=5,
                                alpha=0.8
                            )

            if has_data:
                ax.set_title(f"Layer {block_idx}", fontsize=12, fontweight='bold')
                ax.grid(True, which="both", ls="--", alpha=0.3)
                ax.legend(loc='best', fontsize=8, ncol=2 if len(attn_types) > 1 else 1)
                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel("Condition Number (log scale)", fontsize=10)
                
                # Add threshold line
                ax.axhline(y=1e6, color='red', linestyle=':', alpha=0.5, linewidth=1)
            else:
                ax.set_title(f"Layer {block_idx} (No Data)", fontsize=12, color='gray')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_axis_off()

        # Hide unused subplots
        for idx in range(num_blocks, len(axes)):
            axes[idx].set_visible(False)

        title = layer_type.replace('_', ' ').title()
        fig.suptitle(
            f"Condition Number Evolution - {title}",
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save
        save_path = os.path.join(output_dir, f"{layer_type}_condition_numbers.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close(fig)


def plot_condition_number_statistics(data: List[Dict], output_dir: str):
    """Plot statistical distributions of condition numbers."""
    
    if not data:
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Condition Number Statistics', fontsize=16, fontweight='bold')
    
    # 1. Distribution by layer type
    ax = axes[0, 0]
    for layer_type in df['layer_type'].unique():
        layer_df = df[df['layer_type'] == layer_type]
        ax.hist(
            np.log10(layer_df['condition_number']),
            bins=50,
            alpha=0.6,
            label=layer_type
        )
    ax.set_xlabel('log₁₀(Condition Number)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution by Layer Type', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Evolution over epochs
    ax = axes[0, 1]
    for layer_type in df['layer_type'].unique():
        layer_df = df[df['layer_type'] == layer_type]
        grouped = layer_df.groupby('epoch')['condition_number'].agg(['mean', 'std'])
        
        ax.plot(
            grouped.index,
            grouped['mean'],
            label=f'{layer_type} (mean)',
            linewidth=2
        )
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2
        )
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Condition Number', fontsize=10)
    ax.set_title('Mean ± Std Over Epochs', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. By projection type
    ax = axes[1, 0]
    proj_stats = df.groupby('projection')['condition_number'].agg(['mean', 'std', 'max'])
    proj_stats.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Projection Type', fontsize=10)
    ax.set_ylabel('Condition Number (log scale)', fontsize=10)
    ax.set_title('Statistics by Projection Type', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(['Mean', 'Std', 'Max'], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. By factor type
    ax = axes[1, 1]
    factor_stats = df.groupby('factor')['condition_number'].agg(['mean', 'std', 'max'])
    factor_stats.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Factor Type', fontsize=10)
    ax.set_ylabel('Condition Number (log scale)', fontsize=10)
    ax.set_title('Statistics by Factor Type', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(['Mean', 'Std', 'Max'], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = os.path.join(output_dir, 'condition_number_statistics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced condition number analysis for Shampoo preconditioners.'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory containing checkpoint .pth files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output plots (default: checkpoint_dir)'
    )
    parser.add_argument(
        '--memory-efficient',
        action='store_true',
        help='Use memory-efficient processing'
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Skip CSV export'
    )
    parser.add_argument(
        '--condition-threshold',
        type=float,
        default=1e6,
        help='Threshold for flagging high condition numbers (default: 1e6)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.checkpoint_dir
    
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("🔬 SHAMPOO CONDITION NUMBER ANALYZER (Enhanced Version)")
    print("="*80)
    print(f"📂 Checkpoint directory: {args.checkpoint_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🧮 Memory efficient mode: {args.memory_efficient}")
    print(f"📊 Export CSV: {not args.no_csv}")
    print(f"⚠️  Condition threshold: {args.condition_threshold:.0e}")
    print("="*80 + "\n")

    analyze_condition_numbers(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir,
        memory_efficient=args.memory_efficient,
        export_csv=not args.no_csv,
        condition_threshold=args.condition_threshold
    )


if __name__ == '__main__':
    main()
