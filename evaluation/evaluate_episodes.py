import numpy as np
import torch
from typing import Tuple, List, Optional


def compute_trajectory_error(predicted: torch.Tensor, actual: torch.Tensor) -> Tuple[float, float, float]:
    """Compute trajectory prediction errors.
    
    Args:
        predicted: Predicted trajectories (N, T, 3) for xyz coordinates
        actual: Actual trajectories (N, T, 3) for xyz coordinates
        
    Returns:
        Tuple of (ADE, FDE, RMSE)
        - ADE: Average Displacement Error across all timesteps
        - FDE: Final Displacement Error (last timestep)
        - RMSE: Root Mean Square Error across all dimensions using PyTorch's MSELoss
    """
    # Average Displacement Error
    ade = torch.mean(torch.sqrt(torch.sum((predicted - actual) ** 2, dim=-1))).item()
    
    # Final Displacement Error (last timestep)
    fde = torch.mean(torch.sqrt(torch.sum((predicted[:, -1] - actual[:, -1]) ** 2, dim=-1))).item()
    
    # Root Mean Square Error using PyTorch's MSELoss
    mse_loss = torch.nn.MSELoss()
    rmse = torch.sqrt(mse_loss(predicted, actual)).item()
    
    return ade, fde, rmse


def evaluate_trajectory(
        scene_data: torch.Tensor,
        state_dim: int,
        act_dim: int,
        model: torch.nn.Module,
        max_seq_len: int = 50,
        device: str = 'cuda',
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,

) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """Evaluate trajectory prediction on a single scene.
    

    Uses our 50xNx10 format
    Evaluates a single scene
    Handles data normilzation
    Computes metrics for predictions and evaluation
    Runs models evavaltion (model.eval())
    """
    model.eval()
    model.to(device=device)

    # Move data to device
    scene_data = scene_data.to(device=device)
    if state_mean is not None:
        state_mean = torch.from_numpy(state_mean).to(dtype=torch.float32, device=device)
    if state_std is not None:
        state_std = torch.from_numpy(state_std).to(dtype=torch.float32, device=device)


        # ensure we are not updating models weights during evaluation
    with torch.no_grad():
        
        # creates tensor for our 50 timesteps
        # So DT can know our current temporal state
        timesteps = torch.arange(0, scene_data.shape[0], device=device)
        
        # Create causal attention mask where each position can only attend to previous positions
        # Shape: (1, sequence_length, sequence_length)
        # For frame 10, it can only see frames 0-9, and frames 10-49 are masked
        seq_length = scene_data.shape[0]
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=device))
        # Add batch dimension
        attention_mask = attention_mask.unsqueeze(0)
        print(f'\nDebug: Attention mask info:')
        print(f'  Shape: {attention_mask.shape} (batch, seq_len, seq_len)')
        print(f'  Example - Frame 10 can see: {attention_mask[0, 10, :11].sum().item()} previous frames')
        print(f'  Example - Frame 10 is masked from: {attention_mask[0, 10, 11:].sum().item()} future frames')

        # Normalize states if mean/std provided
        states = scene_data.clone().to(dtype=torch.float32)
        if state_mean is not None and state_std is not None:
            states = (states - state_mean) / state_std
        
        # Get dimensions
        num_timesteps, num_objects, feature_dim = states.shape
        
        # Calculate required padding to match training dimension of 36480
        required_objects = 36480 // feature_dim  # Calculate how many objects we need
        if num_objects < required_objects:
            padded_states = torch.zeros((num_timesteps, required_objects, feature_dim), dtype=torch.float32, device=device)
            padded_states[:, :num_objects, :] = states
            states = padded_states
        
        # Re shapes data to match DT input format
        # Ex: 50x753x10 --> 1x50x7530
        states = states.reshape(1, num_timesteps, -1) 
        

        # Get number of objects in the scene
        num_timesteps, num_objects, _ = scene_data.shape
        
        # Set reward of 1 for each prediction
        # RTG is sum of current + future rewards
        # For each timestep t, RTG is (num_timesteps - t)
        # Add +1 to timesteps since trainer expects extra timestep
        base_rtg = torch.arange(num_timesteps, -1, -1, dtype=torch.float32, device=device)
        # Expand RTG for each object: (timesteps+1, num_objects)
        rtg = base_rtg.unsqueeze(1).repeat(1, num_objects)
        # Add batch dimension: (1, timesteps+1, num_objects)
        rtg = rtg.unsqueeze(0)
        
        print(f'\nDebug: RTG info:')
        print(f'  Shape: {rtg.shape} (batch, timesteps+1, num_objects)')
        print(f'  Starts at {rtg[0,0,0]} for each object')
        print(f'  Ends at {rtg[0,-1,0]} for each object')
        print(f'  Number of objects with RTG: {num_objects}')

        print(f'\nDebug: Original state shape: {states.shape}')
        
        # Truncate or pad to match expected feature dimension
        target_features = model.embed_state.weight.shape[1]  # What the model expects
        current_features = states.shape[-1]
        
        if current_features != target_features:
            print(f'Warning: Input features ({current_features}) do not match model expected features ({target_features})')
            if current_features > target_features:
                # Truncate if we have too many features
                states = states[:, :, :target_features]
            else:
                # Pad with zeros if we have too few features
                padding = torch.zeros((1, num_timesteps, target_features - current_features), dtype=torch.float32, device=device)
                states = torch.cat([states, padding], dim=-1)
            
        print(f'Debug: Final state shape: {states.shape}')
        


        # Create tensors for actions with same shape as states
        actions = torch.zeros_like(states)
        
        # Forward pass through model
        # tells the model what postion in the sequence we are for evaluation
        timesteps = torch.arange(start=0, end=num_timesteps, step=1, device=device)
        timesteps = timesteps.unsqueeze(0).repeat(1, 1)
        


        # Create attention mask for padded sequences
        # set to all 1s becuase all tiesteps are valid
        # transfomer should look at all states
        attention_mask = torch.ones((1, num_timesteps), dtype=torch.float32, device=device)
        


        # Actual forward pass now that we have defined all our requirements
        # We only really care about state_preds
        # As DT handles actions, rewards, etc.
        state_preds, action_preds, reward_preds = model.forward(
            states=states,
            actions=actions,
            rewards=torch.zeros((1, scene_data.shape[0], 1), dtype=torch.float32, device=device),
            returns_to_go=rtg[:,:-1],
            timesteps=timesteps,
            attention_mask=attention_mask
        )


        """ 
            Collect and return the predictions 
            PLUS
            Debug print statements
        """

        # Get the actual number of objects from scene_data
        _, num_objects_actual, _ = scene_data.shape
        print(f'Debug: Actual number of objects: {num_objects_actual}')
        
        # Get predictions (next state positions)
        predictions = action_preds[0]  # Remove batch dimension
        print(f'Debug: Raw predictions shape: {predictions.shape}')
        
        # Calculate how many features per object we have in the predictions
        features_per_object = 10  # Same as input
        num_objects_pred = predictions.shape[-1] // features_per_object
        print(f'Debug: Number of objects in predictions: {num_objects_pred}')
        
        # Reshape predictions to match the number of objects we can handle
        predictions = predictions.reshape(-1, num_objects_pred, features_per_object)
        print(f'Debug: Reshaped predictions: {predictions.shape}')
        
        # Take only the first num_objects_actual objects
        predictions = predictions[:, :num_objects_actual, :]
        print(f'Debug: Truncated predictions: {predictions.shape}')
        
        # Extract only the position predictions (first 3 dimensions of each object)
        position_preds = predictions[:, :, :3]  # Shape: (T, N, 3)
        position_actual = scene_data[:, :, :3]  # Shape: (T, N, 3)
        
        print(f'Debug: Position predictions shape: {position_preds.shape}')
        print(f'Debug: Actual positions shape: {position_actual.shape}')
        
        # Calculate errors comparing to ground truth
        # Use the next timestep as ground truth
        print(f'\nDebug: Computing errors between predictions and ground truth')
        print(f'Predictions timesteps (excluding last): {position_preds[:-1].shape}')
        print(f'Ground truth timesteps (excluding first): {position_actual[1:].shape}')
        
        ade, fde, rmse = compute_trajectory_error(position_preds[:-1], position_actual[1:])
        print(f'\nDebug: Error metrics:')
        print(f'ADE: {ade:.4f}')
        print(f'FDE: {fde:.4f}')
        print(f'RMSE: {rmse:.4f}')
        

        
        # Return predictions and error metrics
        return predictions, (ade, fde, rmse)
    returns_to_go = returns_to_go.reshape(-1, 1).repeat(1, num_objects).reshape(-1, num_objects, 1)

    # Normalize states
    normalized_states = (states - state_mean) / state_std

    # Initialize predictions storage
    predictions = torch.zeros((max_seq_len, num_objects, 3), device=device)
    
    # Generate predictions for each timestep
    for t in range(max_seq_len - 1):

        # Get model's prediction for the next position
        timesteps = torch.full((1,), t, device=device, dtype=torch.long)
        
        action = model.get_action(
            normalized_states[:t+1],  # History of states up to t
            actions[:t+1],            # History of actions up to t
            returns_to_go[:t+1],      # Remaining returns
            timesteps,                # Current timestep
        )
        
        # Store prediction
        predictions[t+1] = action.reshape(num_objects, 3)

    # Compute evaluation metrics
    # Use actual positions from t+1 onwards as ground truth
    actual_trajectories = scene_data[1:, :, :3]  # xyz coordinates
    predicted_trajectories = predictions[1:]      # Remove first timestep (given)
    
    # Compute trajectory errors
    ade, fde, rmse = compute_trajectory_error(predicted_trajectories, actual_trajectories)

    return predictions, (ade, fde, rmse)


def evaluate_batch_trajectories(
        batch_data: torch.Tensor,
        state_dim: int,
        act_dim: int,
        model: torch.nn.Module,
        max_seq_len: int = 50,
        device: str = 'cuda',
        target_return: Optional[float] = None,
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float, float]]]:
    """Evaluate trajectory prediction on a batch of scenes.
    
    Args:
        batch_data: Tensor of shape (B, T, N, D) containing multiple scenes
            B: batch size
            T: number of timesteps
            N: number of objects
            D: feature dimension
        state_dim: Dimension of state features
        act_dim: Dimension of action features
        model: Decision Transformer model
        max_seq_len: Maximum sequence length
        device: Device to run evaluation on
        target_return: Optional target return (not used for trajectory prediction)
        state_mean: Optional state normalization mean
        state_std: Optional state normalization std
    
    Returns:
        Tuple of (all_predictions, all_errors)
        - all_predictions: List of predicted trajectories for each scene
        - all_errors: List of (ade, fde, rmse) tuples for each scene
    """


    # calling the eval model
    model.eval()
    model.to(device=device)

    # Initialize storage    
    batch_size = batch_data.shape[0]
    all_predictions = []
    all_errors = []
    

    # Extract scene from batch
  
    for scene_idx in range(batch_data.shape[0]):
        scene_data = batch_data[scene_idx]  # Each scene is 50xNx10
        
        # Skip scenes that are too short
        # Need at least 2 timesteps for prediction

        if scene_data.shape[0] < 2:  
            continue
            

         # Call our single-scene evaluator
         # Store each scene 
         # Maintain batch processing structure
        predictions, errors = evaluate_trajectory(
            scene_data=scene_data,
            state_dim=state_dim,
            act_dim=act_dim,
            model=model,
            max_seq_len=max_seq_len,
            device=device,
            target_return=target_return,
            state_mean=state_mean,
            state_std=state_std,
        )
        
        # Store results
        all_predictions.append(predictions)
        all_errors.append(errors)
    
    if not all_predictions:
        raise ValueError("No valid scenes found in batch_data")
        
    return all_predictions, all_errors
