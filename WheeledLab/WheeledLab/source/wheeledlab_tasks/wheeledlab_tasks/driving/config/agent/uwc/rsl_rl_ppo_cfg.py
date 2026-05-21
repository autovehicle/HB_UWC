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
        value_loss_coef=1.0,                # Critic 실수에서 얼마나 penalty를 줄지 결정 (해당 값이 높을수록 critic교육에 더 집중)
        use_clipped_value_loss=True,        # 한 번에 너무 크게 업데이트 되는 것을 방지
        clip_param=0.2,                     # 한 번에 너무 크게 업데이트 되는 것을 방지 (clip_param이 0.2면, 행동 확률이 20% 이상 바뀌는 것을 방지)
        entropy_coef=0.005,                 # 행동의 다양성 유지를 위해 entropy에 penalty 주는 parameter (값이 높을수록 행동의 다양성 유지에 더 집중)
        num_learning_epochs=5,              # epoch: 128회의 경험을 몇 번 복습할건지 결정
        num_mini_batches=4,                 # mini-batch: 128회의 경험을 몇 개의 작은 batch로 나눌건지 결정 (128회 경험을 4개로 나누면, 한 mini-batch는 32회 경험으로 구성)
        learning_rate=5.0e-4,               # learning rate: 신경망이 얼마나 빠르게 업데이트 되는지 결정 (값이 너무 높으면 학습이 불안정해지고, 너무 낮으면 학습이 느려짐)
        schedule="adaptive",                # learning rate를 adaptive하게 조절 (KL divergence가 desired_kl보다 크면 learning rate를 줄이고, 작으면 learning rate를 늘리는 방식)
        gamma=0.99,                         # 미래 보상에 대한 할인율 (값이 0.99면, 1초 후의 보상보다 100초 후의 보상이 100배 더 작게 평가됨)
        lam=0.95,                           # 이득 계산을 얼마나 더 멀리 볼지 결정 (값이 0.95면, 1초 후의 보상보다 20초 후의 보상이 100배 더 작게 평가됨)
        desired_kl=0.01,                    # learning rate를 adaptive하게 조절할 때 사용하는 parameter (KL divergence가 desired_kl보다 크면 learning rate를 줄이고, 작으면 learning rate를 늘리는 방식)
        max_grad_norm=1.0,                  # gradient clipping: 한 번에 너무 크게 업데이트 되는 것을 방지 (값이 1.0이면, gradient의 L2 norm이 1.0을 넘지 않도록 클리핑)
    )