from rsl_rl import runners


class OnPolicyRunner(runners.OnPolicyRunner):
    """Thin compatibility wrapper around current rsl-rl OnPolicyRunner."""

    def __init__(self, env, agent_cfg, log_cfg, device="cpu"):
        # Use the official rsl-rl 5.x runner implementation.
        # Do not override learn(), because the old custom learn loop is incompatible
        # with the current IsaacLab / rsl-rl API.
        super().__init__(env, agent_cfg, log_cfg.run_log_dir, device)

        self.log_cfg = log_cfg
        self.no_log = getattr(log_cfg, "no_log", False)
        self.no_wandb = getattr(log_cfg, "no_wandb", True)