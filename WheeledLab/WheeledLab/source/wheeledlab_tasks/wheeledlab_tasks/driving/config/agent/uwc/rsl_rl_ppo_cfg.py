from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
# Training hyperparameters for PPO on UWC Driving task
@configclass
class UWCDrivePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128                 # 한 PPO 학습 마다 각 환경에서 모으는 transition 수
    max_iterations = 300                    # 총 학습 iteration(반복) 횟수
    save_interval = 50                      # 50 step마다 checkpoint 저장
    experiment_name = "ppo_uwc_drive"

    empirical_normalization = False         # 관측지 정규화 여부
    # 한 번의 주기에서 모은 경험(128회)을 한 set 모으고, 그 set을 학습시키고, 다시 경험 모으고 다시 학습시키고 반복을 300회
    # 50번 학습마다 checkpoint 저장하고 학습이 이상해져도 50번 전으로 돌아갈 수 있음

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                 # 얼마나 랜덤하게 행동할지 결정
        actor_hidden_dims=[128, 128],       # 행동 결정(actor) 신경망의 크기
        critic_hidden_dims=[128, 128],      # 행동 평가(critic) 신경망의 크기
        activation="elu",                   # 신경망 활성화 함수(ELU)  
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                # Critic
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )