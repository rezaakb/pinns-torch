from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact)
    exact_v = np.imag(exact)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)
    return {"u": exact_u, "v": exact_v, "h": exact_h}


def output_fn(outputs: Dict[str, torch.Tensor],
              x: torch.Tensor,
              t: torch.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["h"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    return outputs


def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):   
    """Define the partial differential equations (PDEs)."""
    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    v_x, v_t = pinnstorch.utils.gradient(outputs["v"], [x, t])

    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    v_xx = pinnstorch.utils.gradient(v_x, x)[0]

    outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
    outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

    return outputs


def perform_mc_predictions(cfg, model, save_dir):
    """Perform MC-Dropout predictions and save results"""
    
    print("Starting MC-Dropout predictions...")
    
    # Create prediction coordinates
    n_x = cfg.spatial_domain.shape[0]
    n_t = cfg.time_domain.t_points
    
    x = torch.linspace(cfg.mesh.lb[0], cfg.mesh.ub[0], n_x).reshape(-1, 1)
    t = torch.linspace(cfg.mesh.lb[1], cfg.mesh.ub[1], n_t).reshape(-1, 1)
    
    # Create meshgrid for all combinations
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    spatial_coords = [X.reshape(-1, 1)]
    time_coords = T.reshape(-1, 1)
    
    # Move to device
    device = next(model.parameters()).device
    spatial_coords = [coord.to(device) for coord in spatial_coords]
    time_coords = time_coords.to(device)
    
    # Perform MC sampling
    num_samples = cfg.mc_dropout.num_mc_samples
    mc_results = model.mc_predict(spatial_coords, time_coords, num_samples)
    
    # Manually compute h statistics from u and v
    u_samples = mc_results['u_samples']  # (num_samples, batch_size, 1)
    v_samples = mc_results['v_samples']
    
    # Compute h for each sample
    h_samples = torch.sqrt(u_samples ** 2 + v_samples ** 2)
    
    # Compute h statistics
    h_mean = torch.mean(h_samples, dim=0)
    h_std = torch.std(h_samples, dim=0)
    
    # Save predictions
    predictions = {
        'u_mean': mc_results['u_mean'].cpu().numpy(),
        'u_std': mc_results['u_std'].cpu().numpy(),
        'v_mean': mc_results['v_mean'].cpu().numpy(),
        'v_std': mc_results['v_std'].cpu().numpy(),
        'h_mean': h_mean.cpu().numpy(),       
        'h_std': h_std.cpu().numpy()           
    }
    
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    for key, value in predictions.items():
        np.save(pred_dir / f"{key}.npy", value)
    
    print(f"MC-Dropout predictions saved to {pred_dir}")
    
    # Log uncertainty statistics to wandb
    if wandb.run:
        for component in ['u', 'v', 'h']:
            mean_uncertainty = np.mean(predictions[f'{component}_std'])
            max_uncertainty = np.max(predictions[f'{component}_std'])
            wandb.log({
                f"mc_dropout_mean_uncertainty_{component}": mean_uncertainty,
                f"mc_dropout_max_uncertainty_{component}": max_uncertainty
            })

def train_mc_dropout(cfg: DictConfig):
    """Train model with MC-Dropout and perform uncertainty quantification"""
    
    print("Starting MC-Dropout training...")
    
    # Train single model with dropout
    metric_dict, object_dict = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )
    
    print("MC-Dropout training completed. Starting uncertainty analysis...")
    
    # Get trained model and trainer
    model = object_dict.get("model")
    trainer = object_dict.get("trainer")
    
    # Save model
    save_dir = Path(cfg.paths.output_dir) / "mc_dropout_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if trainer and hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            torch.save(torch.load(best_path, weights_only=False), save_dir / "best_model.ckpt")
    
    # Perform MC predictions if model has mc_predict method
    if model and hasattr(model.net, 'mc_predict'):
        perform_mc_predictions(cfg, model.net, save_dir)
    else:
        print("Warning: Model does not support MC-Dropout predictions")
    
    return metric_dict.get("error", None)


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    pinnstorch.utils.extras(cfg)
    
    # Check for ensemble mode
    if cfg.get("ensemble", {}).get("enable", False):
        # Ensemble training (your existing code)
        ensemble_results = []
        save_dir = Path(cfg.paths.output_dir) / "ensemble_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(cfg.ensemble.n_models):
            print(f"Training ensemble model {i+1}/{cfg.ensemble.n_models}")
            
            # Set seed
            seed = cfg.ensemble.base_seeds[i]
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Update config for this model
            model_cfg = cfg.copy()
            model_cfg.seed = seed
            model_cfg.tags = cfg.get("tags", []) + [f"ensemble_model_{i}"]
            
            # Create separate output directory for each model
            base_output_dir = Path(cfg.paths.output_dir)
            model_output_dir = base_output_dir.parent / f"{base_output_dir.name}_model_{i}"
            model_cfg.paths.output_dir = str(model_output_dir)
            model_cfg.task_name = f"{cfg.task_name}_model_{i}"
            
            # Initialize wandb for this model
            if cfg.get("logger", {}).get("wandb"):
                wandb.init(
                    project=cfg.logger.wandb.get("project", "schrodinger_pinn"),
                    group="ensemble_baseline",
                    name=f"model_{i}",
                    tags=model_cfg.tags,
                    reinit=True
                )
            
            # Train model
            metric_dict, object_dict = pinnstorch.train(
                model_cfg, 
                read_data_fn=read_data_fn, 
                pde_fn=pde_fn, 
                output_fn=output_fn
            )
            
            # Save model
            model_dir = save_dir / f"model_{i}"
            model_dir.mkdir(exist_ok=True)
            
            # Get trainer from object_dict
            trainer = object_dict.get("trainer")
            if trainer and hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                torch.save(
                    torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False), 
                    model_dir / "best_model.ckpt"
                )
            
            ensemble_results.append(metric_dict)
            
            if wandb.run:
                wandb.finish()
        
        # Ensemble aggregation logic
        if cfg.get("logger", {}).get("wandb"):
            wandb.init(
                project=cfg.logger.wandb.get("project", "schrodinger_pinn"),
                group="ensemble_baseline", 
                name="ensemble_aggregation",
                tags=cfg.get("tags", []) + ["ensemble_aggregation"],
                reinit=True
            )
        
        metric_values = []
        for result in ensemble_results:
            if "error" in result:
                metric_values.append(result["error"])
        
        if metric_values:
            ensemble_mean = np.mean(metric_values)
            ensemble_std = np.std(metric_values)
            
            if wandb.run:
                wandb.log({
                    "ensemble_mean_error": ensemble_mean,
                    "ensemble_std_error": ensemble_std,
                    "ensemble_size": len(metric_values)
                })
                wandb.finish()
            
            return ensemble_mean
        
        return None
    
    # Check for MC-Dropout mode
    elif cfg.get("mc_dropout", {}).get("enable", False):
        # Initialize wandb for MC-Dropout
        if cfg.get("logger", {}).get("wandb"):
            wandb.init(
                project=cfg.logger.wandb.get("project", "schrodinger_pinn"),
                group="mc_dropout_baseline",
                name="mc_dropout_training",
                tags=cfg.get("tags", []) + ["mc_dropout"],
                reinit=True
            )
        
        result = train_mc_dropout(cfg)
        
        if wandb.run:
            wandb.finish()
            
        return result
        
    else:
        # Original single model training
        metric_dict, _ = pinnstorch.train(
            cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
        )
        
        metric_value = pinnstorch.utils.get_metric_value(
            metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
        )
        
        return metric_value


if __name__ == "__main__":
    main()