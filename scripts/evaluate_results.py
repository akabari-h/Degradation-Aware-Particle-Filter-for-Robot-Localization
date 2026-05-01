#!/usr/bin/env python3
"""
Evaluate Experiment Results
============================
Computes ATE and RPE for DA-PF and AMCL across all degradation levels
using the 'evo' package, and generates comparison plots.

Usage:
  python3 evaluate_results.py

Expects files in ~/rsn_project/results/:
  level_0_ground_truth.txt, level_0_da_pf.txt, level_0_amcl.txt
  level_1_ground_truth.txt, ...
  (through level_4)

Install evo: pip install evo --break-system-packages
  (or in a venv)
"""
import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.expanduser('~/rsn_project/results')
LEVELS = [0, 1, 2, 3, 4]
LEVEL_NAMES = ['Clean', 'LiDAR degraded', 'Camera degraded', 'LiDAR+Camera degraded', 'All degraded']


def run_evo_ape(gt_file, est_file):
    """Run evo_ape and return the RMSE."""
    try:
        result = subprocess.run(
            ['evo_ape', 'tum', gt_file, est_file,
             '--align', '--correct_scale', '--no_warnings',
             '--plot_mode', 'xy'],
            capture_output=True, text=True, timeout=30
        )
        # Parse RMSE from output
        for line in result.stdout.split('\n'):
            if 'rmse' in line.lower():
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if 'rmse' in p.lower() and i + 1 < len(parts):
                        try:
                            return float(parts[i + 1])
                        except ValueError:
                            pass
            # Also try parsing the stats format
            if line.strip().startswith('rmse'):
                try:
                    return float(line.strip().split()[-1])
                except (ValueError, IndexError):
                    pass
        # Fallback: try JSON output
        result2 = subprocess.run(
            ['evo_ape', 'tum', gt_file, est_file,
             '--align', '--correct_scale', '--no_warnings',
             '--output', '/tmp/evo_result.json'],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists('/tmp/evo_result.json'):
            with open('/tmp/evo_result.json') as f:
                data = json.load(f)
                return data.get('rmse', None)
    except Exception as e:
        print(f'  Error running evo_ape: {e}')
    return None


def run_evo_rpe(gt_file, est_file):
    """Run evo_rpe and return the RMSE."""
    try:
        result2 = subprocess.run(
            ['evo_rpe', 'tum', gt_file, est_file,
             '--align', '--correct_scale', '--no_warnings',
             '--output', '/tmp/evo_rpe_result.json'],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists('/tmp/evo_rpe_result.json'):
            with open('/tmp/evo_rpe_result.json') as f:
                data = json.load(f)
                return data.get('rmse', None)
    except Exception as e:
        print(f'  Error running evo_rpe: {e}')
    return None


def compute_simple_ate(gt_file, est_file):
    """
    Fallback ATE computation if evo is not installed.
    Loads TUM files, matches by sequence order (not timestamp)
    when timestamps are in different time bases.
    """
    try:
        gt = np.loadtxt(gt_file)
        est = np.loadtxt(est_file)
    except Exception as e:
        print(f'  Error loading files: {e}')
        return None

    if len(gt) < 2 or len(est) < 2:
        print(f'  Not enough data points (GT: {len(gt)}, Est: {len(est)})')
        return None

    # Check if timestamps are in different bases
    gt_t0 = gt[0, 0]
    est_t0 = est[0, 0]
    time_diff = abs(gt_t0 - est_t0)

    if time_diff > 1000:
        # Different time bases — match by evenly sampling
        # Resample both to same number of points
        n_points = min(len(gt), len(est))
        gt_indices = np.linspace(0, len(gt) - 1, n_points, dtype=int)
        est_indices = np.linspace(0, len(est) - 1, n_points, dtype=int)

        gt_sampled = gt[gt_indices]
        est_sampled = est[est_indices]

        errors = []
        for i in range(n_points):
            dx = est_sampled[i, 1] - gt_sampled[i, 1]
            dy = est_sampled[i, 2] - gt_sampled[i, 2]
            errors.append(np.sqrt(dx**2 + dy**2))
    else:
        # Same time base — match by nearest timestamp
        errors = []
        for e_row in est:
            t_est = e_row[0]
            idx = np.argmin(np.abs(gt[:, 0] - t_est))
            dt = abs(gt[idx, 0] - t_est)
            if dt < 1.0:
                dx = e_row[1] - gt[idx, 1]
                dy = e_row[2] - gt[idx, 2]
                errors.append(np.sqrt(dx**2 + dy**2))

    if len(errors) < 2:
        print(f'  Not enough matched points ({len(errors)})')
        return None

    rmse = np.sqrt(np.mean(np.array(errors)**2))
    return rmse


def main():
    print('=' * 60)
    print('  DA-PF vs AMCL — Experiment Evaluation')
    print('=' * 60)
    print()

    # Check if evo is available
    evo_available = True
    try:
        subprocess.run(['evo_ape', '--help'], capture_output=True, timeout=5)
    except FileNotFoundError:
        evo_available = False
        print('WARNING: evo not installed. Using simple ATE computation.')
        print('Install with: pip install evo --break-system-packages')
        print()

    # Collect results
    da_pf_ate = []
    amcl_ate = []
    da_pf_rpe = []
    amcl_rpe = []

    for level in LEVELS:
        gt_file = os.path.join(RESULTS_DIR, f'level_{level}_ground_truth.txt')
        pf_file = os.path.join(RESULTS_DIR, f'level_{level}_da_pf.txt')
        amcl_file = os.path.join(RESULTS_DIR, f'level_{level}_amcl.txt')

        print(f'Level {level} ({LEVEL_NAMES[level]}):')

        # Check files exist
        missing = False
        for f, name in [(gt_file, 'GT'), (pf_file, 'DA-PF'), (amcl_file, 'AMCL')]:
            if not os.path.exists(f):
                print(f'  MISSING: {name} file ({f})')
                missing = True
            else:
                lines = sum(1 for _ in open(f))
                print(f'  {name}: {lines} poses recorded')

        if missing:
            da_pf_ate.append(None)
            amcl_ate.append(None)
            da_pf_rpe.append(None)
            amcl_rpe.append(None)
            print()
            continue

        # Compute ATE using simple method (handles timestamp mismatches)
        pf_rmse = compute_simple_ate(gt_file, pf_file)
        amcl_rmse = compute_simple_ate(gt_file, amcl_file)

        da_pf_ate.append(pf_rmse)
        amcl_ate.append(amcl_rmse)

        if pf_rmse is not None:
            print(f'  DA-PF ATE (RMSE): {pf_rmse:.4f} m')
        else:
            print(f'  DA-PF ATE: FAILED')

        if amcl_rmse is not None:
            print(f'  AMCL  ATE (RMSE): {amcl_rmse:.4f} m')
        else:
            print(f'  AMCL  ATE: FAILED')

        # Compute RPE (evo only)
        if evo_available:
            pf_rpe = run_evo_rpe(gt_file, pf_file)
            amcl_rpe_val = run_evo_rpe(gt_file, amcl_file)
            da_pf_rpe.append(pf_rpe)
            amcl_rpe.append(amcl_rpe_val)
        else:
            da_pf_rpe.append(None)
            amcl_rpe.append(None)

        print()

    # ==============================
    #   Print summary table
    # ==============================
    print('=' * 60)
    print('  SUMMARY TABLE — ATE RMSE (meters)')
    print('=' * 60)
    print(f'{"Level":<12} {"DA-PF":>10} {"AMCL":>10} {"Winner":>10}')
    print('-' * 42)

    for i, level in enumerate(LEVELS):
        pf_val = f'{da_pf_ate[i]:.4f}' if da_pf_ate[i] is not None else 'N/A'
        amcl_val = f'{amcl_ate[i]:.4f}' if amcl_ate[i] is not None else 'N/A'
        if da_pf_ate[i] is not None and amcl_ate[i] is not None:
            winner = 'DA-PF' if da_pf_ate[i] < amcl_ate[i] else 'AMCL'
        else:
            winner = '-'
        print(f'{LEVEL_NAMES[i]:<12} {pf_val:>10} {amcl_val:>10} {winner:>10}')

    # ==============================
    #   Generate plots
    # ==============================
    valid_levels = [i for i in range(len(LEVELS)) if da_pf_ate[i] is not None and amcl_ate[i] is not None]

    if len(valid_levels) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ATE plot
        ax1 = axes[0]
        x = [LEVELS[i] for i in valid_levels]
        pf_vals = [da_pf_ate[i] for i in valid_levels]
        amcl_vals = [amcl_ate[i] for i in valid_levels]

        ax1.plot(x, pf_vals, 'b-o', linewidth=2, markersize=8, label='DA-PF (ours)')
        ax1.plot(x, amcl_vals, 'r-s', linewidth=2, markersize=8, label='AMCL (baseline)')
        ax1.set_xlabel('Degradation Level', fontsize=12)
        ax1.set_ylabel('ATE RMSE (meters)', fontsize=12)
        ax1.set_title('Absolute Trajectory Error vs Degradation', fontsize=14)
        ax1.set_xticks(LEVELS)
        ax1.set_xticklabels(LEVEL_NAMES, fontsize=10)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Particle spread plot (if we recorded it)
        ax2 = axes[1]
        improvement = []
        for i in valid_levels:
            if amcl_ate[i] > 0:
                imp = ((amcl_ate[i] - da_pf_ate[i]) / amcl_ate[i]) * 100
                improvement.append(imp)
            else:
                improvement.append(0)

        colors = ['#1D9E75' if v > 0 else '#E24B4A' for v in improvement]
        ax2.bar([LEVELS[i] for i in valid_levels], improvement, color=colors, alpha=0.8)
        ax2.set_xlabel('Degradation Level', fontsize=12)
        ax2.set_ylabel('DA-PF Improvement (%)', fontsize=12)
        ax2.set_title('DA-PF Improvement Over AMCL', fontsize=14)
        ax2.set_xticks(LEVELS)
        ax2.set_xticklabels(LEVEL_NAMES, fontsize=10)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, 'comparison_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f'\nPlot saved: {plot_path}')
        plt.show()
    else:
        print('\nNot enough data for plots. Run more experiments.')

    print('\nDone!')


if __name__ == '__main__':
    main()