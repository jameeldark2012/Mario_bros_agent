from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.logger import configure
logger = configure("logs\\", ["stdout", "csv", "tensorboard"])
from mario_env import MarioEnv , wrap_mario_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Create a checkpoint callback to save the model every X steps



def optimize_PPO(trail):
    n_layers = trail.suggest_int("n_layers", 5, 10)
    net_arch = [trail.suggest_int(f"neurons_hidden_layer_{i+1}", 512, 1024) for i in range(n_layers)]

    return{
        "policy":trail.suggest_categorical("policy", ["CnnPolicy"]),
        "learning_rate":trail.suggest_loguniform("learning_rate" , 0.00010124,  0.00010125 ),
        "n_epochs":trail.suggest_int("n_epochs" , 4 , 5),
        "batch_size":trail.suggest_int("batch_size" , 1918 , 1919),
        "clip_range":trail.suggest_loguniform("clip_range" , 0.235449 ,0.23545  ),
        "target_kl":trail.suggest_loguniform("target_kl" , 0.01375822 , 0.01375823),
        "gamma":trail.suggest_loguniform("gamma" , 0.921127, 0.99),
        "gae_lambda":trail.suggest_loguniform("gae_lambda" , 0.96028 , 0.96029 ),
        "vf_coef":trail.suggest_loguniform("vf_coef" , 0.833419 , 0.83342 ),
        "ent_coef":trail.suggest_loguniform("ent_coef" , 0.00014560 , 0.00014561),
        "n_steps":trail.suggest_int("n_steps" , 4097 , 4098),


    } , net_arch


TOTAL_TIME_STEPS = 10000000
N_ENVS = 5
N_TRIALS = 10

# import optuna
turn = 0

def optimize_agent(trail):
    global turn , TOTAL_TIME_STEPS
    turn+=1
    logger = configure("logs\\PPO_iter3_"+str(turn), ["stdout", "csv", "tensorboard"])
    
    env = MarioEnv()
    env = wrap_mario_env(env)
    model_params , net_arch = optimize_PPO(trail)

    model = PPO( env=env ,  device="cuda" , policy_kwargs={ "net_arch" : net_arch} , verbose= 1      , **model_params     )
    model.set_logger(logger)
    model.learn(total_timesteps= TOTAL_TIME_STEPS , log_interval=1 , reset_num_timesteps=True )
    
    mean_reward , _ = evaluate_policy(model ,env,n_eval_episodes= 50)
    print("Evaluated trail and mean reward is" , mean_reward)

    return mean_reward

if __name__ == '__main__':

        # study = optuna.create_study(direction='maximize')
        # study.optimize(optimize_agent, n_trials=N_TRIALS, n_jobs=N_ENVS ,show_progress_bar= True )
        
        # with open("best_params/params_PPO_iter3", "wb") as file:
        #     pickle.dump(study.best_params, file)

        # with open("best_params/oct_1", "wb") as file:
        #     pickle.dump(study, file)

        logger = configure("logs\\PPO_best_at2", ["stdout", "csv", "tensorboard"])
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/',
                                         name_prefix='ppo_mario')
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
        env = MarioEnv()
        env = wrap_mario_env(env)

        
        model = PPO( env=env ,  device="cuda" , verbose= 1    , **model_params  , policy_kwargs={ "net_arch" : net_arch} )
        model.set_logger(logger)
        model.learn(total_timesteps= TOTAL_TIME_STEPS , log_interval=1 , reset_num_timesteps=True   , callback=checkpoint_callback)
        model.save("PPO")
        mean_reward , _ = evaluate_policy(model ,env,n_eval_episodes= 50 , render=True  )
        # print("Evaluated trail and mean reward is" , mean_reward)
        # # model = model.load("ppo")
        # # model.set_logger(logger)

        # print()
        # for i in range (1):
        #      episode_reward = 0
        #      observation = env.reset()[0]
        #      done = False
             
        #      while not done:

        #         action , _ = model.predict(observation)
        #         print(action)
        #         print(env.table.iloc[int(action)])
        #         observation  ,reward, done, truncated, info = env.step(int(action))
        #         print(env.table.iloc[int(action)])
        #      episode_reward+=reward
        #      print("Ep reward:" , episode_reward)




