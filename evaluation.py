import os
import absl
import absl.flags

from edp.utilities.utils import (
  Timer,
  WandBLogger,
  define_flags_with_default,
)
from edp.diffusion.trainer import DiffusionTrainer
from edp.diffusion.dql import DiffusionQL


FLAGS_DEF = define_flags_with_default(
  algo="DiffQL",
  # algo="DiffusionQL",
  type="model-free",
  env="walker2d-medium-replay-v2",
  dataset='d4rl',
  rl_unplugged_task_class='control_suite',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256-256",
  qf_arch="256-256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=DiffusionQL.get_default_config(),
  n_epochs=2000,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  logging=WandBLogger.get_default_config(mcep=True),
  qf_layer_norm=False,
  policy_layer_norm=False,
  activation="mish",
  obs_norm=False,
  act_method='',
  sample_method='ddpm',
  policy_temp=1.0,
  norm_reward=False,
  pkl_path="model.pkl",
  project='OfflineRL_edp',
  prefix='mcep_',
)


if __name__ == '__main__':

  def main(argv):
    trainer = DiffusionTrainer(FLAGS_DEF)
    trainer.train_mcep()
    os._exit(os.EX_OK)

  absl.app.run(main)
