import torch
import torch.nn as nn

def simple_learning_function(Q_eval_network,
           Q_target_network,
           optimizer,
           loss_function,
           state,
           action,
           reward,
           next_state,
           done,
           batch_size,
           gamma):
        q_eval = Q_eval_network(state)[torch.arange(batch_size),action]
        q_next = Q_target_network(next_state).max(dim=1)[0]
        q_next[done] = 0.0
        q_target = reward + gamma*q_next
        loss = loss_function(q_eval,q_target)
        loss.backward()
        optimizer.step()
        
def double_learning_function(Q_eval_network,
           Q_target_network,
           optimizer,
           loss_function,
           state,
           action,
           reward,
           next_state,
           done,
           batch_size,
           gamma):
    
        q_eval = Q_eval_network(state)[torch.arange(batch_size),action]
        q_next_eval = Q_eval_network(next_state)
        next_actions = q_next_eval.max(dim=1)[1]
        q_next_target = Q_target_network(next_state)
        
        q_next_eval = q_next_target[torch.arange(batch_size),action]
        
        q_next_eval[done] = 0.0
        q_target = reward + gamma*q_next_eval
        loss = loss_function(q_eval,q_target)
        loss.backward()
        optimizer.step()