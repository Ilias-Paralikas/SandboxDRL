import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning Parameters')

    parser.add_argument('--config_file', type=str, default='config.json', help='file to save the configuration')
    parser.add_argument('--memory_size', type=int, default=10000, help='Size of the replay memory')
    parser.add_argument('--epsilon', type=float, default=1, help='Initial value of epsilon for epsilon-greedy policy')
    parser.add_argument('--epsilon_decay', type=float, default=0.1, help='Decay rate of epsilon')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum value of epsilon')
    parser.add_argument('--replace_target_frequency', type=int, default=10, help='Frequency of replacing target network')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor for future rewards')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_function', type=str, default='simple', help='Learning function to use')
    parser.add_argument('--save_model_frequency', type=int, default=1, help='Frequency of saving the model')

    args = parser.parse_args()
    
    json_object = json.dumps(vars(args),indent=4) ### this saves the array in .json format)


    with open(args.config_file, 'w') as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()