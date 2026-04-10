import os
from os import path


def flatten_config_for_wandb(value, prefix: str = '', out: dict | None = None) -> dict:
    if out is None:
        out = {}

    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            flatten_config_for_wandb(item, child_prefix, out)
        return out

    if isinstance(value, (list, tuple)):
        out[prefix] = list(value)
        return out

    if isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
        return out

    out[prefix] = repr(value)
    return out


def default_wandb_run_name() -> str:
    config_file = os.environ.get('MORTAL_CFG', 'config.toml')
    return path.splitext(path.basename(config_file))[0]


def wandb_logging_disabled() -> bool:
    raw_disabled = str(os.environ.get('WANDB_DISABLED', '')).strip().lower()
    if raw_disabled in {'1', 'true', 'yes', 'on'}:
        return True
    return str(os.environ.get('WANDB_MODE', '')).strip().lower() == 'disabled'


def maybe_init_wandb_run(
    *,
    full_config: dict,
    wandb_cfg: dict,
    fallback_name: str,
    job_type: str,
    run_id: str = '',
    name_suffix: str = '',
):
    if not wandb_cfg.get('enabled', False):
        return None
    if wandb_logging_disabled():
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'Weights & Biases logging is enabled, but the `wandb` package is not installed.'
        ) from exc

    wandb_dir = wandb_cfg.get('dir') or ''
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        cache_dir = path.join(wandb_dir, 'cache')
        config_dir = path.join(wandb_dir, 'config')
        data_dir = path.join(wandb_dir, 'data')
        for dirname in (cache_dir, config_dir, data_dir):
            os.makedirs(dirname, exist_ok=True)
        os.environ.setdefault('WANDB_DIR', wandb_dir)
        os.environ.setdefault('WANDB_CACHE_DIR', cache_dir)
        os.environ.setdefault('WANDB_CONFIG_DIR', config_dir)
        os.environ.setdefault('WANDB_DATA_DIR', data_dir)

    entity = os.environ.get('WANDB_ENTITY') or wandb_cfg.get('entity') or None
    project = os.environ.get('WANDB_PROJECT') or wandb_cfg.get('project') or 'mortal'
    name = os.environ.get('WANDB_RUN_NAME') or wandb_cfg.get('name') or fallback_name
    if name_suffix:
        name = f'{name}{name_suffix}'

    init_kwargs = {
        'project': project,
        'entity': entity,
        'name': name,
        'group': wandb_cfg.get('group') or None,
        'job_type': wandb_cfg.get('job_type') or job_type,
        'tags': list(wandb_cfg.get('tags', [])) or None,
        'notes': wandb_cfg.get('notes') or None,
        'dir': wandb_cfg.get('dir') or None,
        'mode': wandb_cfg.get('mode') or None,
        'resume': wandb_cfg.get('resume') or 'allow',
        'id': run_id or wandb_cfg.get('run_id') or None,
        'config': flatten_config_for_wandb(full_config) if wandb_cfg.get('log_config', True) else None,
        'reinit': 'create_new',
    }

    run = wandb.init(**init_kwargs)
    return run
