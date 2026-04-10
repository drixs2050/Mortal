#!/usr/bin/env python

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import (  # noqa: E402
    build_eval_command,
    build_stage_command,
    build_train_command,
    ensure_output_dirs,
    launcher_payload,
    make_campaign_summary,
    missing_input_paths,
    run_command,
    subprocess_env,
    validate_torch_visible_launch_gpus,
    utc_now_iso,
    write_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Launch a Step 6 BC campaign: train, then final full val/test eval.',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the Step 6 BC config TOML.',
    )
    parser.add_argument(
        '--torchrun-bin',
        default='torchrun',
        help='Torchrun executable to use for distributed training.',
    )
    parser.add_argument(
        '--python-bin',
        default=sys.executable,
        help='Python executable to use for final eval commands.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path, full_config, launch_settings, config_fingerprint_value = launcher_payload(args.config)

    missing = missing_input_paths(launch_settings)
    if missing:
        missing_desc = ', '.join(f'{name}={filename}' for name, filename in missing.items())
        raise FileNotFoundError(f'missing BC campaign input artifacts: {missing_desc}')

    ensure_output_dirs(launch_settings)

    env = subprocess_env(config_path)
    if launch_settings['nproc_per_node'] > 1:
        validate_torch_visible_launch_gpus(
            env=env,
            python_bin=args.python_bin,
            expected_count=launch_settings['nproc_per_node'],
            required_name_substring='A100',
        )
    stage_command = build_stage_command(
        config_path=config_path,
        full_config=full_config,
        python_bin=args.python_bin,
    )
    train_command = build_train_command(
        config_path=config_path,
        launch_settings=launch_settings,
        torchrun_bin=args.torchrun_bin,
    )
    checkpoint_path = launch_settings['best_state_file']
    val_command = build_eval_command(
        checkpoint=checkpoint_path,
        split='val',
        output_json=launch_settings['final_val_json'],
        eval_device=launch_settings['eval_device'],
        python_bin=args.python_bin,
    )
    test_command = build_eval_command(
        checkpoint=checkpoint_path,
        split='test',
        output_json=launch_settings['final_test_json'],
        eval_device=launch_settings['eval_device'],
        python_bin=args.python_bin,
    )
    commands = {
        **({'stage': stage_command} if len(stage_command) > 4 else {}),
        'train': train_command,
        'final_val': val_command,
        'final_test': test_command,
    }

    started_at = utc_now_iso()
    current_stage = 'stage' if 'stage' in commands else 'train'
    try:
        if 'stage' in commands:
            run_command(command=stage_command, env=env, stage=current_stage)
            current_stage = 'train'
        run_command(command=train_command, env=env, stage=current_stage)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'expected promoted best checkpoint after training, but it does not exist: {checkpoint_path}'
            )
        current_stage = 'final_val'
        run_command(command=val_command, env=env, stage=current_stage)
        current_stage = 'final_test'
        run_command(command=test_command, env=env, stage=current_stage)
    except KeyboardInterrupt as exc:
        finished_at = utc_now_iso()
        summary = make_campaign_summary(
            config_path=str(config_path),
            config_fingerprint_value=config_fingerprint_value,
            started_at=started_at,
            finished_at=finished_at,
            status='interrupted',
            checkpoint_paths={
                'state_file': str(launch_settings['state_file']),
                'best_state_file': str(launch_settings['best_state_file']),
            },
            report_paths={
                'final_val_json': str(launch_settings['final_val_json']),
                'final_test_json': str(launch_settings['final_test_json']),
            },
            commands=commands,
            failed_stage=current_stage,
            return_code=130,
            error='KeyboardInterrupt',
        )
        write_summary(launch_settings['campaign_summary_json'], summary)
        raise SystemExit(130) from exc
    except subprocess.CalledProcessError as exc:
        finished_at = utc_now_iso()
        summary = make_campaign_summary(
            config_path=str(config_path),
            config_fingerprint_value=config_fingerprint_value,
            started_at=started_at,
            finished_at=finished_at,
            status='failed',
            checkpoint_paths={
                'state_file': str(launch_settings['state_file']),
                'best_state_file': str(launch_settings['best_state_file']),
            },
            report_paths={
                'final_val_json': str(launch_settings['final_val_json']),
                'final_test_json': str(launch_settings['final_test_json']),
            },
            commands=commands,
            failed_stage=current_stage,
            return_code=exc.returncode,
            error=str(exc),
        )
        write_summary(launch_settings['campaign_summary_json'], summary)
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        finished_at = utc_now_iso()
        summary = make_campaign_summary(
            config_path=str(config_path),
            config_fingerprint_value=config_fingerprint_value,
            started_at=started_at,
            finished_at=finished_at,
            status='failed',
            checkpoint_paths={
                'state_file': str(launch_settings['state_file']),
                'best_state_file': str(launch_settings['best_state_file']),
            },
            report_paths={
                'final_val_json': str(launch_settings['final_val_json']),
                'final_test_json': str(launch_settings['final_test_json']),
            },
            commands=commands,
            failed_stage=current_stage,
            error=str(exc),
        )
        write_summary(launch_settings['campaign_summary_json'], summary)
        raise

    finished_at = utc_now_iso()
    summary = make_campaign_summary(
        config_path=str(config_path),
        config_fingerprint_value=config_fingerprint_value,
        started_at=started_at,
        finished_at=finished_at,
        status='completed',
        checkpoint_paths={
            'state_file': str(launch_settings['state_file']),
            'best_state_file': str(launch_settings['best_state_file']),
        },
        report_paths={
            'final_val_json': str(launch_settings['final_val_json']),
            'final_test_json': str(launch_settings['final_test_json']),
        },
        commands=commands,
    )
    write_summary(launch_settings['campaign_summary_json'], summary)


if __name__ == '__main__':
    main()
