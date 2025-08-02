#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for Amharic H-Net model results.

This script provides functions for visualizing model performance,
training progress, and comparison between different models.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


class ResultVisualizer:
    """Visualizer for model results."""
    
    def __init__(self, 
                 output_dir: Optional[Union[str, Path]] = None,
                 dpi: int = 300,
                 fig_format: str = 'png'):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
            fig_format: Format for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi
        self.fig_format = fig_format
        
        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Filename without extension
        """
        if self.output_dir:
            filepath = self.output_dir / f"{filename}.{self.fig_format}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
    
    def plot_training_loss(self, 
                          log_file: Union[str, Path],
                          save: bool = True) -> plt.Figure:
        """Plot training loss.
        
        Args:
            log_file: Path to training log file
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read log file
        log_file = Path(log_file)
        if not log_file.exists():
            logger.error(f"Log file {log_file} does not exist")
            return None
        
        # Parse log file
        epochs = []
        train_losses = []
        val_losses = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Epoch' in line and 'Train Loss' in line:
                        parts = line.strip().split()
                        epoch_idx = parts.index('Epoch')
                        train_loss_idx = parts.index('Loss:') if 'Loss:' in parts else parts.index('Train')
                        
                        try:
                            epoch = int(parts[epoch_idx + 1].rstrip(':,'))
                            train_loss = float(parts[train_loss_idx + 1].rstrip(','))
                            
                            epochs.append(epoch)
                            train_losses.append(train_loss)
                        except (ValueError, IndexError):
                            continue
                    
                    if 'Validation Loss' in line:
                        parts = line.strip().split()
                        val_loss_idx = parts.index('Loss:') if 'Loss:' in parts else parts.index('Validation')
                        
                        try:
                            val_loss = float(parts[val_loss_idx + 1].rstrip(','))
                            val_losses.append(val_loss)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training loss
        if train_losses:
            ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        
        # Plot validation loss
        if val_losses and len(val_losses) == len(epochs):
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        
        # Set x-axis to integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'training_loss')
        
        return fig
    
    def plot_learning_rate(self, 
                          log_file: Union[str, Path],
                          save: bool = True) -> plt.Figure:
        """Plot learning rate.
        
        Args:
            log_file: Path to training log file
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read log file
        log_file = Path(log_file)
        if not log_file.exists():
            logger.error(f"Log file {log_file} does not exist")
            return None
        
        # Parse log file
        steps = []
        learning_rates = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Learning Rate' in line:
                        parts = line.strip().split()
                        lr_idx = parts.index('Rate:')
                        step_idx = parts.index('Step:')
                        
                        try:
                            step = int(parts[step_idx + 1].rstrip(','))
                            lr = float(parts[lr_idx + 1].rstrip(','))
                            
                            steps.append(step)
                            learning_rates.append(lr)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot learning rate
        if learning_rates:
            ax.plot(steps, learning_rates, 'g-')
        
        # Set labels and title
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'learning_rate')
        
        return fig
    
    def plot_evaluation_metrics(self, 
                               eval_file: Union[str, Path],
                               metrics: Optional[List[str]] = None,
                               save: bool = True) -> plt.Figure:
        """Plot evaluation metrics.
        
        Args:
            eval_file: Path to evaluation results file
            metrics: List of metrics to plot
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read evaluation file
        eval_file = Path(eval_file)
        if not eval_file.exists():
            logger.error(f"Evaluation file {eval_file} does not exist")
            return None
        
        # Load evaluation results
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation results from {eval_file}: {e}")
            return None
        
        # Extract metrics
        if not metrics:
            metrics = ['grammar_score', 'coherence_score', 'repetition_score', 'cultural_relevance_score', 'overall_score']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            if any(metric in result for result in results):
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.error(f"No metrics found in evaluation results")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        data = {}
        for metric in available_metrics:
            data[metric] = [result.get(metric, 0) for result in results]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Plot metrics
        df.boxplot(ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'evaluation_metrics')
        
        return fig
    
    def plot_model_comparison(self, 
                             comparison_file: Union[str, Path],
                             metrics: Optional[List[str]] = None,
                             save: bool = True) -> plt.Figure:
        """Plot model comparison.
        
        Args:
            comparison_file: Path to model comparison file
            metrics: List of metrics to plot
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read comparison file
        comparison_file = Path(comparison_file)
        if not comparison_file.exists():
            logger.error(f"Comparison file {comparison_file} does not exist")
            return None
        
        # Load comparison results
        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading comparison results from {comparison_file}: {e}")
            return None
        
        # Extract metrics
        if not metrics:
            metrics = ['grammar_score', 'coherence_score', 'repetition_score', 'cultural_relevance_score', 'overall_score']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            if any(metric in model_results for model_name, model_results in results.items()):
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.error(f"No metrics found in comparison results")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        model_names = list(results.keys())
        data = []
        
        for metric in available_metrics:
            metric_data = []
            for model_name in model_names:
                metric_data.append(results[model_name].get(metric, 0))
            data.append(metric_data)
        
        # Set bar width
        bar_width = 0.8 / len(available_metrics)
        
        # Set positions
        positions = np.arange(len(model_names))
        
        # Plot bars
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics) / 2 + 0.5) * bar_width
            ax.bar(positions + offset, data[i], bar_width, label=metric)
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        
        # Set x-axis ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'model_comparison')
        
        return fig
    
    def plot_inference_time(self, 
                           comparison_file: Union[str, Path],
                           save: bool = True) -> plt.Figure:
        """Plot inference time comparison.
        
        Args:
            comparison_file: Path to model comparison file
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read comparison file
        comparison_file = Path(comparison_file)
        if not comparison_file.exists():
            logger.error(f"Comparison file {comparison_file} does not exist")
            return None
        
        # Load comparison results
        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading comparison results from {comparison_file}: {e}")
            return None
        
        # Extract inference times
        model_names = []
        inference_times = []
        
        for model_name, model_results in results.items():
            if 'inference_time' in model_results:
                model_names.append(model_name)
                inference_times.append(model_results['inference_time'])
        
        if not inference_times:
            logger.error(f"No inference times found in comparison results")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        ax.bar(model_names, inference_times)
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Comparison')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'inference_time')
        
        return fig
    
    def plot_model_size(self, 
                       comparison_file: Union[str, Path],
                       save: bool = True) -> plt.Figure:
        """Plot model size comparison.
        
        Args:
            comparison_file: Path to model comparison file
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read comparison file
        comparison_file = Path(comparison_file)
        if not comparison_file.exists():
            logger.error(f"Comparison file {comparison_file} does not exist")
            return None
        
        # Load comparison results
        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading comparison results from {comparison_file}: {e}")
            return None
        
        # Extract model sizes
        model_names = []
        model_sizes = []
        
        for model_name, model_results in results.items():
            if 'model_size' in model_results:
                model_names.append(model_name)
                model_sizes.append(model_results['model_size'] / (1024 * 1024))  # Convert to MB
        
        if not model_sizes:
            logger.error(f"No model sizes found in comparison results")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        ax.bar(model_names, model_sizes)
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size Comparison')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'model_size')
        
        return fig
    
    def plot_perplexity(self, 
                       eval_file: Union[str, Path],
                       save: bool = True) -> plt.Figure:
        """Plot perplexity.
        
        Args:
            eval_file: Path to evaluation results file
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Read evaluation file
        eval_file = Path(eval_file)
        if not eval_file.exists():
            logger.error(f"Evaluation file {eval_file} does not exist")
            return None
        
        # Load evaluation results
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation results from {eval_file}: {e}")
            return None
        
        # Extract perplexity
        perplexities = []
        prompts = []
        
        for result in results:
            if 'perplexity' in result and 'prompt' in result:
                perplexities.append(result['perplexity'])
                prompts.append(result['prompt'])
        
        if not perplexities:
            logger.error(f"No perplexity found in evaluation results")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        ax.bar(range(len(prompts)), perplexities)
        
        # Set labels and title
        ax.set_xlabel('Prompt')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity by Prompt')
        
        # Set x-axis ticks
        ax.set_xticks(range(len(prompts)))
        ax.set_xticklabels([f"Prompt {i+1}" for i in range(len(prompts))], rotation=45)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        if save:
            self._save_figure(fig, 'perplexity')
        
        return fig
    
    def plot_all(self, 
                log_file: Optional[Union[str, Path]] = None,
                eval_file: Optional[Union[str, Path]] = None,
                comparison_file: Optional[Union[str, Path]] = None) -> None:
        """Plot all available visualizations.
        
        Args:
            log_file: Path to training log file
            eval_file: Path to evaluation results file
            comparison_file: Path to model comparison file
        """
        if log_file:
            self.plot_training_loss(log_file)
            self.plot_learning_rate(log_file)
        
        if eval_file:
            self.plot_evaluation_metrics(eval_file)
            self.plot_perplexity(eval_file)
        
        if comparison_file:
            self.plot_model_comparison(comparison_file)
            self.plot_inference_time(comparison_file)
            self.plot_model_size(comparison_file)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize model results")
    
    # Input arguments
    parser.add_argument("--log_file", type=str, help="Path to training log file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation results file")
    parser.add_argument("--comparison_file", type=str, help="Path to model comparison file")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--fig_format", type=str, default="png", choices=["png", "jpg", "svg", "pdf"], help="Format for saved figures")
    
    # Visualization arguments
    parser.add_argument("--plot_all", action="store_true", help="Plot all available visualizations")
    parser.add_argument("--plot_training_loss", action="store_true", help="Plot training loss")
    parser.add_argument("--plot_learning_rate", action="store_true", help="Plot learning rate")
    parser.add_argument("--plot_evaluation_metrics", action="store_true", help="Plot evaluation metrics")
    parser.add_argument("--plot_model_comparison", action="store_true", help="Plot model comparison")
    parser.add_argument("--plot_inference_time", action="store_true", help="Plot inference time comparison")
    parser.add_argument("--plot_model_size", action="store_true", help="Plot model size comparison")
    parser.add_argument("--plot_perplexity", action="store_true", help="Plot perplexity")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultVisualizer(
        output_dir=args.output_dir,
        dpi=args.dpi,
        fig_format=args.fig_format
    )
    
    # Plot visualizations
    if args.plot_all:
        visualizer.plot_all(
            log_file=args.log_file,
            eval_file=args.eval_file,
            comparison_file=args.comparison_file
        )
    else:
        if args.plot_training_loss and args.log_file:
            visualizer.plot_training_loss(args.log_file)
        
        if args.plot_learning_rate and args.log_file:
            visualizer.plot_learning_rate(args.log_file)
        
        if args.plot_evaluation_metrics and args.eval_file:
            visualizer.plot_evaluation_metrics(args.eval_file)
        
        if args.plot_model_comparison and args.comparison_file:
            visualizer.plot_model_comparison(args.comparison_file)
        
        if args.plot_inference_time and args.comparison_file:
            visualizer.plot_inference_time(args.comparison_file)
        
        if args.plot_model_size and args.comparison_file:
            visualizer.plot_model_size(args.comparison_file)
        
        if args.plot_perplexity and args.eval_file:
            visualizer.plot_perplexity(args.eval_file)


if __name__ == "__main__":
    main()