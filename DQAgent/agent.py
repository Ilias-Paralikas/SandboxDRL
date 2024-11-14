import torch 
import torch.nn as nn
import copy
from .replay_memory import ReplayMemory
from .learning_functions import simple_learning_function, double_learning_function
class Agent():
    def __init__(self,
                 model,
                 optimizer,
                 loss_function,
                 memory_size=10000,
                 epsilon=1,
                 epsilon_decay=0.01,
                 epsilon_min=0.01,
                 replace_target_frequency=10,
                 gamma=0.9,
                 batch_size =32,
                 learning_function='simple',
                 device =None,
                 model_file=None,
                save_model_frequency=1):
        def handle_none_input(value,func):
            if value is None:
                return func()
            return value
        
        learning_function_choices = {
            'simple': simple_learning_function,
            'double': double_learning_function
        }
        self.learning_function = learning_function_choices[learning_function]
        
        self.device = handle_none_input(device,lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.Q_eval_network =model.to(device)
        self.action_space = self.Q_eval_network.get_action_space()
        self.Q_target_network = copy.deepcopy(model).to(device)
        self.replay_memory = ReplayMemory(memory_size)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min= epsilon_min
        self.replace_target_frequency = replace_target_frequency
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer =optimizer
        self.loss_function = loss_function
        self.model_file = model_file
        self.save_model_frequency = save_model_frequency
        
        
        self.learn_step_counter = 0
    def to_torch(self,value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype).to(self.device)

        
    def choose_action(self,state):
        if torch.rand(1).item() < self.epsilon:
            action =  torch.randint(0,self.action_space,(1,)).item()
        else:
            with torch.no_grad():
                self.Q_eval_network.eval()
                state = self.to_torch(state)
                action=  torch.argmax(self.Q_eval_network(state)).item()
        return action
    
    def store_transition(self,state,action,reward,next_state,done):
        self.replay_memory.push(state,action,reward,next_state,done)

    def save_model(self,model_file=None):
        if model_file is None:
            model_file = self.model_file
        torch.save(self.Q_eval_network.state_dict(), model_file)
        
    def load_model(self,model_file=None):
        if model_file is None:
            model_file = self.model_file
        self.Q_eval_network.load_state_dict(torch.load(model_file))
        self.Q_target_network.load_state_dict(torch.load(model_file))
            
    def update_target_network(self):
        self.Q_target_network.load_state_dict(self.Q_eval_network.state_dict())
    
    def learn(self):
    
        if len(self.replay_memory) < self.batch_size:
            return
        self.Q_eval_network.train()
        self.optimizer.zero_grad()
        state, action, reward, next_state, done = self.replay_memory.sample(self.batch_size)
        state = self.to_torch(state)
        action = self.to_torch(action,dtype=torch.int32)
        reward = self.to_torch(reward)
        next_state = self.to_torch(next_state)
        done = self.to_torch(done,dtype=torch.bool)
      
        self.learning_function(self.Q_eval_network,
                               self.Q_target_network
                                 ,self.optimizer,
                                    self.loss_function,
                                    state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    self.batch_size,
                                    self.gamma)
        
        
        self.learn_step_counter +=1
        if self.learn_step_counter % self.replace_target_frequency == 0:
            self.update_target_network()
            
        if self.learn_step_counter % self.save_model_frequency == 0:
            self.save_model()
            
    def get_epsilon(self):
        return self.epsilon
    def reduce_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)