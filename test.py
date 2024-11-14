import gym
from DQAgent import Agent
from model import Model
import torch
from book_keeper import BookKeeper
import argparse 

def main():
    parser = argparse.ArgumentParser(description='Deep Q Learning')
    parser.add_argument('--config_file', type=str, default='config.json', help='file to save the configuration')
    parser.add_argument('--log_folder', type=str, default='logs', help='Folder to save the logs')
    parser.add_argument('--resume_run', type=str, default=None, help='Resume a previous run')
    parser.add_argument('--epochs', type=str, default=None, help='File to save/load the model')
    args = parser.parse_args()
    
    
    bookkeeper = BookKeeper(config_source_file= args.config_file)
    hyperparameters = bookkeeper.get_hyperparameters()
    model_file = bookkeeper.get_model_file()
    env = gym.make('CartPole-v0')

    model = Model(env)
    optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    agent = Agent(model,
                    optimizer=optimizer,
                  loss_function=loss_function,
                  memory_size=hyperparameters['memory_size'],
                  epsilon=hyperparameters['epsilon'],
                  epsilon_decay=hyperparameters['epsilon_decay'],
                  epsilon_min=hyperparameters['epsilon_min'],
                  replace_target_frequency=hyperparameters['replace_target_frequency'],
                  gamma=hyperparameters['gamma'],
                  batch_size=hyperparameters['batch_size'],
                  learning_function=hyperparameters['learning_function'],
                  save_model_frequency=hyperparameters['save_model_frequency'],
                  model_file=model_file)
    
    epochs =20
    for i in range(epochs):
        obs = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.learn()
            obs = next_obs
            
            info = {
                'reward':reward,
            }
            bookkeeper.store_step(info)
        episode_info ={
            'epsilon':agent.epsilon
        }
        bookkeeper.store_episode(info=episode_info,verbose=True)
        agent.reduce_epsilon()
    bookkeeper.plot_metrics()
            

if __name__ == '__main__':
    main()