import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from wandb_utils import default_wandb_run_name, flatten_config_for_wandb, wandb_logging_disabled  # noqa: E402


class WandbUtilsTest(unittest.TestCase):
    def test_flatten_config_for_wandb_preserves_nested_scalars(self):
        flat = flatten_config_for_wandb({
            'bc': {
                'control': {
                    'batch_size': 2048,
                    'device': 'cuda:0',
                },
                'wandb': {
                    'tags': ['step5', 'bc'],
                },
            },
        })
        self.assertEqual(flat['bc.control.batch_size'], 2048)
        self.assertEqual(flat['bc.control.device'], 'cuda:0')
        self.assertEqual(flat['bc.wandb.tags'], ['step5', 'bc'])

    def test_default_wandb_run_name_uses_config_basename(self):
        old = __import__('os').environ.get('MORTAL_CFG')
        try:
            __import__('os').environ['MORTAL_CFG'] = '/tmp/step5_bc_medium.toml'
            self.assertEqual(default_wandb_run_name(), 'step5_bc_medium')
        finally:
            if old is None:
                __import__('os').environ.pop('MORTAL_CFG', None)
            else:
                __import__('os').environ['MORTAL_CFG'] = old

    def test_wandb_logging_disabled_respects_env_switches(self):
        os = __import__('os')
        old_mode = os.environ.get('WANDB_MODE')
        old_disabled = os.environ.get('WANDB_DISABLED')
        try:
            os.environ['WANDB_MODE'] = 'disabled'
            os.environ.pop('WANDB_DISABLED', None)
            self.assertTrue(wandb_logging_disabled())
            os.environ.pop('WANDB_MODE', None)
            os.environ['WANDB_DISABLED'] = 'true'
            self.assertTrue(wandb_logging_disabled())
        finally:
            if old_mode is None:
                os.environ.pop('WANDB_MODE', None)
            else:
                os.environ['WANDB_MODE'] = old_mode
            if old_disabled is None:
                os.environ.pop('WANDB_DISABLED', None)
            else:
                os.environ['WANDB_DISABLED'] = old_disabled


if __name__ == '__main__':
    unittest.main()
