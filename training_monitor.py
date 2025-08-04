#!/usr/bin/env python3
"""
Training Progress Monitor for Smart Amharic LLM
Follows troubleshooting guidelines for monitoring training progress
"""

import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class TrainingMonitor:
    """Monitor training progress following troubleshooting guidelines"""
    
    def __init__(self, output_dir: str = "training_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training metrics storage
        self.training_history = {
            "losses": [],
            "learning_rates": [],
            "epochs": [],
            "eval_losses": [],
            "perplexities": [],
            "timestamps": []
        }
        
        logger.info(f"📊 Training monitor initialized. Plots will be saved to {self.output_dir}")
        
    def load_training_logs(self, log_file: Optional[str] = None) -> bool:
        """Load training logs from file or search for existing logs"""
        if log_file is None:
            # Search for trainer_state.json files
            possible_paths = [
                "models/amharic-gpt2-local/trainer_state.json",
                "checkpoint-*/trainer_state.json",
                "training_logs.json"
            ]
            
            for pattern in possible_paths:
                files = list(Path(".").glob(pattern))
                if files:
                    log_file = str(files[0])
                    break
                    
        if log_file and Path(log_file).exists():
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract training history from trainer_state.json format
                if 'log_history' in data:
                    self._parse_trainer_state(data)
                    logger.info(f"✅ Loaded training logs from {log_file}")
                    return True
                    
            except Exception as e:
                logger.error(f"❌ Error loading training logs: {e}")
                
        logger.warning("⚠️  No training logs found. Using synthetic data for demonstration.")
        self._generate_synthetic_data()
        return False
        
    def _parse_trainer_state(self, trainer_state: Dict):
        """Parse trainer_state.json format"""
        log_history = trainer_state.get('log_history', [])
        
        for entry in log_history:
            if 'train_loss' in entry:
                self.training_history["losses"].append(entry['train_loss'])
                self.training_history["epochs"].append(entry.get('epoch', 0))
                self.training_history["learning_rates"].append(entry.get('learning_rate', 0))
                
            if 'eval_loss' in entry:
                self.training_history["eval_losses"].append(entry['eval_loss'])
                
        # Calculate perplexities from losses
        self.training_history["perplexities"] = [np.exp(loss) for loss in self.training_history["losses"]]
        
    def _generate_synthetic_data(self):
        """Generate synthetic training data for demonstration"""
        logger.info("📈 Generating synthetic training data for demonstration...")
        
        # Simulate realistic training curve
        epochs = np.linspace(0, 10, 100)
        base_loss = 4.0
        
        # Exponential decay with some noise
        losses = base_loss * np.exp(-epochs/3) + 0.5 + 0.1 * np.random.normal(0, 1, len(epochs))
        losses = np.maximum(losses, 0.5)  # Minimum loss threshold
        
        # Learning rate schedule (cosine decay)
        initial_lr = 5e-5
        learning_rates = initial_lr * (1 + np.cos(np.pi * epochs / 10)) / 2
        
        # Evaluation losses (slightly higher than training)
        eval_losses = losses + 0.1 + 0.05 * np.random.normal(0, 1, len(losses))
        
        self.training_history = {
            "losses": losses.tolist(),
            "learning_rates": learning_rates.tolist(),
            "epochs": epochs.tolist(),
            "eval_losses": eval_losses.tolist(),
            "perplexities": np.exp(losses).tolist(),
            "timestamps": [datetime.now().isoformat() for _ in epochs]
        }
        
    def plot_training_loss(self, save_plot: bool = True) -> str:
        """Plot training loss curves - following guidelines"""
        plt.figure(figsize=(12, 8))
        
        # Main loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history["epochs"], self.training_history["losses"], 
                label='Training Loss', linewidth=2, color='#e74c3c')
        
        if self.training_history["eval_losses"]:
            plt.plot(self.training_history["epochs"], self.training_history["eval_losses"], 
                    label='Validation Loss', linewidth=2, color='#3498db')
                    
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('🇪🇹 Smart Amharic LLM - Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity plot
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history["epochs"], self.training_history["perplexities"], 
                linewidth=2, color='#9b59b6')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Model Perplexity')
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history["epochs"], self.training_history["learning_rates"], 
                linewidth=2, color='#f39c12')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Loss improvement rate
        plt.subplot(2, 2, 4)
        if len(self.training_history["losses"]) > 1:
            loss_diff = np.diff(self.training_history["losses"])
            plt.plot(self.training_history["epochs"][1:], loss_diff, 
                    linewidth=2, color='#27ae60')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.title('Loss Improvement Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"training_loss_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training loss plot saved to {plot_file}")
            
        plt.show()
        return str(plot_file) if save_plot else ""
        
    def plot_training_metrics_dashboard(self, save_plot: bool = True) -> str:
        """Create comprehensive training dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🇪🇹 Smart Amharic LLM - Training Dashboard', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(self.training_history["epochs"], self.training_history["losses"], 
                       label='Training', linewidth=2, color='#e74c3c')
        if self.training_history["eval_losses"]:
            axes[0, 0].plot(self.training_history["epochs"], self.training_history["eval_losses"], 
                           label='Validation', linewidth=2, color='#3498db')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perplexity
        axes[0, 1].plot(self.training_history["epochs"], self.training_history["perplexities"], 
                       linewidth=2, color='#9b59b6')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Model Perplexity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 2].plot(self.training_history["epochs"], self.training_history["learning_rates"], 
                       linewidth=2, color='#f39c12')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[1, 0].hist(self.training_history["losses"], bins=30, alpha=0.7, color='#e74c3c')
        axes[1, 0].set_xlabel('Loss Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Loss Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training progress (smoothed)
        if len(self.training_history["losses"]) > 10:
            window_size = min(10, len(self.training_history["losses"]) // 5)
            smoothed_loss = np.convolve(self.training_history["losses"], 
                                      np.ones(window_size)/window_size, mode='valid')
            smoothed_epochs = self.training_history["epochs"][:len(smoothed_loss)]
            
            axes[1, 1].plot(smoothed_epochs, smoothed_loss, linewidth=3, color='#27ae60')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].set_title('Training Progress (Smoothed)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training statistics
        stats_text = f"""
        📊 Training Statistics:
        
        • Total Epochs: {len(self.training_history['epochs'])}
        • Final Loss: {self.training_history['losses'][-1]:.4f}
        • Best Loss: {min(self.training_history['losses']):.4f}
        • Final Perplexity: {self.training_history['perplexities'][-1]:.2f}
        • Best Perplexity: {min(self.training_history['perplexities']):.2f}
        
        🎯 Performance Indicators:
        • Loss Reduction: {((self.training_history['losses'][0] - self.training_history['losses'][-1]) / self.training_history['losses'][0] * 100):.1f}%
        • Convergence: {'✅ Good' if self.training_history['losses'][-1] < 2.0 else '⚠️ Needs Improvement'}
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"training_dashboard_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training dashboard saved to {plot_file}")
            
        plt.show()
        return str(plot_file) if save_plot else ""
        
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "training_summary": {
                "total_epochs": len(self.training_history['epochs']),
                "final_loss": self.training_history['losses'][-1] if self.training_history['losses'] else 0,
                "best_loss": min(self.training_history['losses']) if self.training_history['losses'] else 0,
                "final_perplexity": self.training_history['perplexities'][-1] if self.training_history['perplexities'] else 0,
                "best_perplexity": min(self.training_history['perplexities']) if self.training_history['perplexities'] else 0
            },
            "performance_metrics": {
                "loss_reduction_percentage": 0,
                "convergence_status": "unknown",
                "training_stability": "unknown"
            },
            "recommendations": []
        }
        
        if self.training_history['losses']:
            initial_loss = self.training_history['losses'][0]
            final_loss = self.training_history['losses'][-1]
            loss_reduction = ((initial_loss - final_loss) / initial_loss * 100)
            
            report["performance_metrics"]["loss_reduction_percentage"] = loss_reduction
            
            # Convergence analysis
            if final_loss < 1.5:
                report["performance_metrics"]["convergence_status"] = "excellent"
            elif final_loss < 2.5:
                report["performance_metrics"]["convergence_status"] = "good"
            else:
                report["performance_metrics"]["convergence_status"] = "needs_improvement"
                
            # Training stability
            if len(self.training_history['losses']) > 10:
                recent_losses = self.training_history['losses'][-10:]
                loss_variance = np.var(recent_losses)
                if loss_variance < 0.01:
                    report["performance_metrics"]["training_stability"] = "stable"
                elif loss_variance < 0.05:
                    report["performance_metrics"]["training_stability"] = "moderate"
                else:
                    report["performance_metrics"]["training_stability"] = "unstable"
                    
            # Generate recommendations
            if loss_reduction < 20:
                report["recommendations"].append("Consider increasing training epochs or adjusting learning rate")
            if final_loss > 2.5:
                report["recommendations"].append("Model may benefit from more training data or different architecture")
            if report["performance_metrics"]["training_stability"] == "unstable":
                report["recommendations"].append("Consider reducing learning rate or adding gradient clipping")
                
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"training_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📋 Training report saved to {report_file}")
        return str(report_file)
        
    def print_training_summary(self):
        """Print training summary to console"""
        print("\n" + "="*60)
        print("🇪🇹 SMART AMHARIC LLM - TRAINING SUMMARY")
        print("="*60)
        
        if not self.training_history['losses']:
            print("❌ No training data available")
            return
            
        print(f"\n📊 TRAINING METRICS:")
        print(f"   • Total Epochs: {len(self.training_history['epochs'])}")
        print(f"   • Final Loss: {self.training_history['losses'][-1]:.4f}")
        print(f"   • Best Loss: {min(self.training_history['losses']):.4f}")
        print(f"   • Final Perplexity: {self.training_history['perplexities'][-1]:.2f}")
        print(f"   • Best Perplexity: {min(self.training_history['perplexities']):.2f}")
        
        initial_loss = self.training_history['losses'][0]
        final_loss = self.training_history['losses'][-1]
        loss_reduction = ((initial_loss - final_loss) / initial_loss * 100)
        
        print(f"\n🎯 PERFORMANCE:")
        print(f"   • Loss Reduction: {loss_reduction:.1f}%")
        
        if final_loss < 1.5:
            print("   • Status: ✅ Excellent convergence")
        elif final_loss < 2.5:
            print("   • Status: ✅ Good convergence")
        else:
            print("   • Status: ⚠️ Needs improvement")
            
        print("\n" + "="*60)
        
def main():
    """Main monitoring function"""
    logger.info("🇪🇹 Starting Smart Amharic LLM Training Monitor")
    
    monitor = TrainingMonitor()
    
    # Load training logs
    monitor.load_training_logs()
    
    # Generate plots
    monitor.plot_training_loss()
    monitor.plot_training_metrics_dashboard()
    
    # Generate report
    monitor.generate_training_report()
    
    # Print summary
    monitor.print_training_summary()
    
    logger.info("✅ Training monitoring completed!")
    logger.info("📁 Check training_plots/ folder for visualizations")
    
if __name__ == "__main__":
    main()