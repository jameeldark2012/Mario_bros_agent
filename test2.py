
# %%
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.logger import configure
from mario_env import MarioEnv , wrap_mario_env
logger = configure("logs\\PPO_best_at2_test", ["stdout", "csv", "tensorboard"])


# %%
env = MarioEnv()
env = wrap_mario_env(env)


# %%
net_arch = [ 
            495,
            414,
            377,
            281,
            474
        ]

model_params = {
            'policy': 'CnnPolicy',
            'learning_rate': 0.00010124518870353412,
            'n_epochs': 5,
            'batch_size': 1918,
            'clip_range': 0.23544921413780545,
            'target_kl': 0.01375822651159254,
            'gamma': 0.9628033486044686,
            'gae_lambda': 0.9602843057997902,
            'vf_coef': 0.8334199629307834,
            'ent_coef': 0.00014560245019128837,
            'n_steps': 4097
            }

# %%
model = PPO( env=env ,  device="cuda" , verbose= 1    , **model_params  , policy_kwargs={ "net_arch" : net_arch} )
model = model.load("models/ppo_mario_3900000_steps" )
model.set_env(env , force_reset=False)

# %%


# %%
# mean_reward , _ = evaluate_policy(model ,env,n_eval_episodes= 5 , render=False  )

mean_reward = 0

for i in range(5):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action , _ = model.predict(obs)
        obs , reward , done , _ = env.step(action)
        mean_reward+=reward
    done = False
    obs = env.reset()
    
mean_reward/=5

print(mean_reward)
