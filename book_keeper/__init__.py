import os
import json
import shutil
import pickle
import matplotlib.pyplot as plt
class BookKeeper:
    
    def __init__(self,
                 config_source_file,
                 log_folder='logs',
                 resume_run=None):
        
        
        self.log_folder = log_folder        
        os.makedirs(self.log_folder, exist_ok=True)
        
        if resume_run:
            self.run_folder  = os.path.join(self.log_folder, resume_run)
        else:
            run_index = self.create_index_file()
            self.run_folder = os.path.join(self.log_folder,f'run_{str(run_index)}')
        
        
        self.config_file = os.path.join(self.run_folder, 'config.json')
        self.metrics_folder = os.path.join(self.run_folder,'metrics.pkl')
        
        if resume_run:
            with open(self.metrics_folder, 'rb') as f:
                self.metrics = pickle.load(f)
        else:
            os.makedirs(self.run_folder)
            shutil.copyfile(config_source_file, self.config_file)
                        
            self.metrics ={
                'step_metrics':{},
                'episode_metrics':{}
            }

        self.model_file = os.path.join(self.run_folder,'model.pth')

    def store_step(self,info):
        for key,value in info.items():
            self.add_to_dict(self.metrics['step_metrics'],key,value)
    def store_episode(self,info={},verbose=False):
        for key,value in info.items():
            self.add_to_dict(self.metrics['episode_metrics'],key,value)
        
        for key,value in self.metrics['step_metrics'].items():
            self.add_to_dict(self.metrics['episode_metrics'],f'{key}_accumulated',sum(value))
            self.metrics['step_metrics'][key] = []
        
        if verbose:
            for key in self.metrics['episode_metrics']:
                print(f'{key}: {self.metrics["episode_metrics"][key][-1]}')
    def get_hyperparameters(self):
        with open(self.config_file) as f:
            hyperparameters = json.load(f)
        return hyperparameters

    def create_index_file(self):
        index_filepath  =self.log_folder+'/index.txt'
        if not os.path.exists(index_filepath):
                with open(index_filepath, 'w') as file:
                    file.write('0')
                    run_index = 0
        else:
            with open(index_filepath, 'r') as file:
                run_index = int(file.read().strip())
            run_index += 1
            with open(index_filepath, 'w') as file:
                file.write(str(run_index))
        return run_index
    
    @staticmethod
    def add_to_dict(dictionary,key,value):
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]
            
    
    def get_model_file(self): 
        return self.model_file        
    
    
    def plot_metrics(self):
        for key,value in self.metrics['episode_metrics'].items():
            plt.plot(value,label=key)
            plt.savefig(os.path.join(self.run_folder,f'{key}.png'))
            plt.close()