import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        print('Getting batch...')
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        print('Got batch. Shapes:', 
              '\nstates:', states.shape if states is not None else None,
              '\nactions:', actions.shape if actions is not None else None,
              '\nrtg:', rtg.shape if rtg is not None else None)
        
        action_target = torch.clone(actions)
        print('Created action target')

        print('Running forward pass...')
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        print('Forward pass complete')

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        print('Reshaped predictions')

        # For trajectory prediction, we only care about action prediction loss
        loss = self.loss_fn(None, action_preds, None, None, action_target, None)
        print('Calculated loss:', loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        print('Completed optimization step')

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
