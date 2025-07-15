"""Experiment tracking system for adaptive conformal prediction."""

from datetime import datetime
import os
from pathlib import Path


class ExperimentTracker:
    def __init__(self):
        self.log_file = Path(__file__).parent / "experiment_log.md"
        self.current_experiment = 1
        
    def log_experiment_start(self, approach_name, hypothesis):
        """Log the start of a new experiment."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n## Experiment {self.current_experiment}: {approach_name}\n")
            f.write(f"**Hypothesis**: {hypothesis}\n")
            f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_training_progress(self, epoch, metrics):
        """Log training progress at specific epochs."""
        with open(self.log_file, 'a') as f:
            f.write(f"### Epoch {epoch} Results:\n")
            f.write(f"- Coverage: {metrics.get('coverage', 0):.1%}\n")
            f.write(f"- MPIW: {metrics.get('mpiw', 0):.1f} pixels\n")
            f.write(f"- Width STD: {metrics.get('width_std', 0):.2f}\n")
            if 'loss' in metrics:
                f.write(f"- Loss: {metrics['loss']:.4f}\n")
            f.write("\n")
    
    def log_results(self, metrics, observations, next_steps):
        """Log final results of an experiment."""
        with open(self.log_file, 'a') as f:
            f.write(f"### Final Results\n")
            f.write(f"- Coverage: {metrics['coverage']:.1%}\n")
            f.write(f"- MPIW: {metrics['mpiw']:.1f} pixels\n")
            f.write(f"- Width STD: {metrics['width_std']:.2f}\n")
            
            if 'size_stratified' in metrics and metrics['size_stratified']:
                f.write(f"\n**Size Stratified Results:**\n")
                for size, data in metrics['size_stratified'].items():
                    f.write(f"- {size.capitalize()}: {data.get('coverage', 0):.1%} coverage, "
                           f"{data.get('mpiw', 0):.1f} MPIW\n")
            
            f.write(f"\n**Observations**: {observations}\n")
            f.write(f"**Next Steps**: {next_steps}\n")
            f.write(f"**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
        
        self.current_experiment += 1
    
    def save_checkpoint(self, model, epoch, metrics, experiment_name):
        """Save model checkpoint."""
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoint_path = checkpoint_dir / f"{experiment_name}_epoch{epoch}.pt"
        
        import torch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        return checkpoint_path