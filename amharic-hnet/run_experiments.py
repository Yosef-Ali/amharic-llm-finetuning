#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment runner for Amharic language model improvements.

This script runs experiments with different model configurations and
compares their performance to identify the best improvements.
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import model and training components
from improved_model import HNetTransformer
from hybrid_tokenizer import HybridAmharicTokenizer
from improved_training import AmharicDataset, ImprovedTrainer, collate_fn
from improved_generation import ImprovedAmharicGenerator
from linguistic_quality_metrics import AmharicLinguisticEvaluator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for Amharic language model experiments."""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 tokenizer_path: Optional[str] = None,
                 base_model_path: Optional[str] = None,
                 test_prompts_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize experiment runner.
        
        Args:
            data_dir: Directory containing data files
            output_dir: Directory to save experiment results
            tokenizer_path: Path to pretrained tokenizer (optional)
            base_model_path: Path to base model for comparison (optional)
            test_prompts_path: Path to test prompts for generation (optional)
            device: Device to use
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path
        self.base_model_path = base_model_path
        self.test_prompts_path = test_prompts_path
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load or create tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = HybridAmharicTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.info("Creating new tokenizer")
            self.tokenizer = HybridAmharicTokenizer(use_bpe=False)  # Character-level tokenizer
            
            # Train tokenizer if data directory exists
            if os.path.exists(data_dir):
                logger.info(f"Training tokenizer on data from {data_dir}")
                self.tokenizer.train_bpe_tokenizer(data_dir)
                
                # Save tokenizer
                tokenizer_dir = os.path.join(output_dir, "tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                self.tokenizer.save(tokenizer_dir)
                logger.info(f"Tokenizer saved to {tokenizer_dir}")
                self.tokenizer_path = tokenizer_dir
        
        # Load test prompts if available
        self.test_prompts = self._load_test_prompts(test_prompts_path)
        
        # Initialize experiment results
        self.experiment_results = {}
    
    def _load_test_prompts(self, test_prompts_path: Optional[str]) -> List[str]:
        """Load test prompts for generation.
        
        Args:
            test_prompts_path: Path to test prompts file
            
        Returns:
            List of test prompts
        """
        default_prompts = [
            "ኢትዮጵያ",
            "አዲስ አበባ",
            "የአማርኛ ቋንቋ",
            "ባህላዊ ምግብ",
            "የኢትዮጵያ ታሪክ"
        ]
        
        if not test_prompts_path or not os.path.exists(test_prompts_path):
            logger.info(f"Test prompts file not found. Using default prompts.")
            return default_prompts
        
        try:
            with open(test_prompts_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            if not prompts:
                logger.warning(f"No prompts found in {test_prompts_path}. Using default prompts.")
                return default_prompts
            
            logger.info(f"Loaded {len(prompts)} test prompts from {test_prompts_path}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading test prompts from {test_prompts_path}: {e}")
            return default_prompts
    
    def run_experiments(self, experiments_config: List[Dict[str, Any]]):
        """Run experiments with different model configurations.
        
        Args:
            experiments_config: List of experiment configurations
        """
        logger.info(f"Running {len(experiments_config)} experiments")
        
        for i, config in enumerate(experiments_config):
            experiment_name = config.get('name', f"experiment_{i+1}")
            logger.info(f"\n{'='*50}\nRunning experiment: {experiment_name}\n{'='*50}")
            
            # Create experiment directory
            experiment_dir = os.path.join(self.output_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Save experiment configuration
            config_path = os.path.join(experiment_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Run experiment
            try:
                result = self._run_single_experiment(config, experiment_dir)
                self.experiment_results[experiment_name] = result
                
                # Save experiment result
                result_path = os.path.join(experiment_dir, "result.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=4)
                
                logger.info(f"Experiment {experiment_name} completed successfully")
            except Exception as e:
                logger.error(f"Error running experiment {experiment_name}: {e}")
                self.experiment_results[experiment_name] = {"error": str(e)}
        
        # Compare experiment results
        self._compare_experiments()
    
    def _run_single_experiment(self, config: Dict[str, Any], experiment_dir: str) -> Dict[str, Any]:
        """Run a single experiment.
        
        Args:
            config: Experiment configuration
            experiment_dir: Directory to save experiment results
            
        Returns:
            Experiment results
        """
        # Extract configuration
        model_config = config.get('model_config', {})
        training_config = config.get('training_config', {})
        generation_config = config.get('generation_config', {})
        
        # Prepare datasets
        max_length = training_config.get('max_length', 512)
        stride = training_config.get('stride', 256)
        val_split = training_config.get('val_split', 0.1)
        use_augmentation = training_config.get('use_augmentation', False)
        
        train_dataset, val_dataset = self._prepare_datasets(
            max_length=max_length,
            stride=stride,
            val_split=val_split,
            use_augmentation=use_augmentation
        )
        
        # Create model
        model = HNetTransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=model_config.get('d_model', 512),
            n_layers=model_config.get('n_layers', 6),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 2048),
            dropout=model_config.get('dropout', 0.1),
            pad_idx=self.tokenizer.get_pad_token_id(),
            use_decoder=model_config.get('use_decoder', False)
        )
        
        # Create trainer
        trainer = ImprovedTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=training_config.get('batch_size', 16),
            learning_rate=training_config.get('learning_rate', 5e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 1000),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            num_epochs=training_config.get('num_epochs', 10),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            use_mixed_precision=training_config.get('use_mixed_precision', True),
            use_cosine_scheduler=training_config.get('use_cosine_scheduler', True),
            checkpoint_dir=experiment_dir,
            log_interval=training_config.get('log_interval', 100),
            device=self.device
        )
        
        # Train model
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Load best model
        best_model_path = os.path.join(experiment_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model = HNetTransformer.from_pretrained(best_model_path, device=self.device)
        
        # Evaluate model
        val_loss = None
        if val_dataset:
            val_loss = self._evaluate_model(model, val_dataset)
        
        # Generate text
        generation_results = self._generate_text(model, generation_config)
        
        # Calculate metrics
        metrics = self._calculate_metrics(generation_results)
        
        # Return experiment results
        return {
            "training_time": training_time,
            "val_loss": val_loss,
            "generation_results": generation_results,
            "metrics": metrics
        }
    
    def _prepare_datasets(self, 
                         max_length: int = 512,
                         stride: int = 256,
                         val_split: float = 0.1,
                         use_augmentation: bool = False) -> Tuple[AmharicDataset, AmharicDataset]:
        """Prepare training and validation datasets.
        
        Args:
            max_length: Maximum sequence length
            stride: Stride for sliding window
            val_split: Fraction of data to use for validation
            use_augmentation: Whether to use data augmentation
            
        Returns:
            Tuple of training and validation datasets
        """
        # Find all text and JSON files in the data directory
        file_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.txt', '.json')):
                    file_paths.append(os.path.join(root, file))
        
        # Shuffle files
        import random
        random.shuffle(file_paths)
        
        # Split into training and validation sets
        val_size = int(len(file_paths) * val_split)
        train_files = file_paths[val_size:]
        val_files = file_paths[:val_size]
        
        logger.info(f"Found {len(file_paths)} files: {len(train_files)} for training, {len(val_files)} for validation")
        
        # Create datasets
        train_dataset = AmharicDataset(
            file_paths=train_files,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=stride,
            use_augmentation=use_augmentation
        )
        
        val_dataset = AmharicDataset(
            file_paths=val_files,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=stride,
            use_augmentation=False  # No augmentation for validation
        )
        
        return train_dataset, val_dataset
    
    def _evaluate_model(self, model: HNetTransformer, val_dataset: AmharicDataset) -> float:
        """Evaluate model on validation dataset.
        
        Args:
            model: Model to evaluate
            val_dataset: Validation dataset
            
        Returns:
            Validation loss
        """
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=model.pad_idx)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Get batch
                input_ids, target_ids = batch
                batch_size = input_ids.size(0)
                
                # Move tensors to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                outputs = model(input_ids)
                
                # Calculate loss
                outputs = outputs.contiguous().view(-1, model.vocab_size)
                targets = target_ids.contiguous().view(-1)
                loss = criterion(outputs, targets)
                
                # Accumulate loss
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return avg_loss
    
    def _generate_text(self, model: HNetTransformer, generation_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate text using the model.
        
        Args:
            model: Model to use for generation
            generation_config: Generation configuration
            
        Returns:
            Dictionary of generated texts for each prompt
        """
        # Create generator
        generator = ImprovedAmharicGenerator(
            model=model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Extract generation parameters
        max_length = generation_config.get('max_length', 100)
        temperature = generation_config.get('temperature', 0.7)
        top_p = generation_config.get('top_p', 0.9)
        repetition_penalty = generation_config.get('repetition_penalty', 1.2)
        use_templates = generation_config.get('use_templates', False)
        
        # Generate text for each prompt
        results = {}
        for prompt in self.test_prompts:
            generated_text = generator.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_template=use_templates
            )
            
            results[prompt] = generated_text
        
        return results
    
    def _calculate_metrics(self, generation_results: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate metrics for generated text.
        
        Args:
            generation_results: Dictionary of generated texts for each prompt
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Create evaluator
            evaluator = AmharicLinguisticEvaluator()
            
            # Calculate metrics
            metrics = {
                'grammar_score': 0.0,
                'coherence_score': 0.0,
                'repetition_score': 0.0,
                'cultural_relevance_score': 0.0,
                'overall_score': 0.0
            }
            
            # Evaluate each generated text
            for prompt, generated_text in generation_results.items():
                # Skip empty generations
                if not generated_text:
                    continue
                
                # Evaluate text
                scores = evaluator.evaluate_text(generated_text)
                
                # Accumulate scores
                for key in metrics.keys():
                    metrics[key] += scores.get(key, 0.0)
            
            # Calculate average scores
            num_generations = len(generation_results)
            if num_generations > 0:
                for key in metrics.keys():
                    metrics[key] /= num_generations
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'error': str(e)
            }
    
    def _compare_experiments(self):
        """Compare experiment results and generate report."""
        if not self.experiment_results:
            logger.warning("No experiment results to compare")
            return
        
        # Create report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_experiments": len(self.experiment_results),
            "experiments": self.experiment_results,
            "comparison": {}
        }
        
        # Compare validation loss
        val_losses = {}
        for name, result in self.experiment_results.items():
            if isinstance(result, dict) and 'val_loss' in result and result['val_loss'] is not None:
                val_losses[name] = result['val_loss']
        
        if val_losses:
            best_val_loss = min(val_losses.items(), key=lambda x: x[1])
            report["comparison"]["best_val_loss"] = {
                "experiment": best_val_loss[0],
                "value": best_val_loss[1]
            }
        
        # Compare training time
        training_times = {}
        for name, result in self.experiment_results.items():
            if isinstance(result, dict) and 'training_time' in result:
                training_times[name] = result['training_time']
        
        if training_times:
            fastest_training = min(training_times.items(), key=lambda x: x[1])
            report["comparison"]["fastest_training"] = {
                "experiment": fastest_training[0],
                "value": fastest_training[1]
            }
        
        # Compare metrics
        metrics_by_experiment = {}
        for name, result in self.experiment_results.items():
            if isinstance(result, dict) and 'metrics' in result and isinstance(result['metrics'], dict):
                metrics_by_experiment[name] = result['metrics']
        
        if metrics_by_experiment:
            # Find best experiment for each metric
            for metric in ['grammar_score', 'coherence_score', 'repetition_score', 'cultural_relevance_score', 'overall_score']:
                metric_values = {}
                for name, metrics in metrics_by_experiment.items():
                    if metric in metrics:
                        metric_values[name] = metrics[metric]
                
                if metric_values:
                    best_metric = max(metric_values.items(), key=lambda x: x[1])
                    report["comparison"][f"best_{metric}"] = {
                        "experiment": best_metric[0],
                        "value": best_metric[1]
                    }
        
        # Determine overall best experiment
        if 'best_overall_score' in report["comparison"]:
            report["comparison"]["best_experiment"] = report["comparison"]["best_overall_score"]["experiment"]
        elif 'best_val_loss' in report["comparison"]:
            report["comparison"]["best_experiment"] = report["comparison"]["best_val_loss"]["experiment"]
        
        # Save report
        report_path = os.path.join(self.output_dir, "experiment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Experiment comparison report saved to {report_path}")
        
        # Log summary
        logger.info("\n" + "="*50)
        logger.info("Experiment Comparison Summary")
        logger.info("="*50)
        
        if 'best_experiment' in report["comparison"]:
            logger.info(f"Best experiment: {report['comparison']['best_experiment']}")
        
        if 'best_val_loss' in report["comparison"]:
            logger.info(f"Best validation loss: {report['comparison']['best_val_loss']['value']:.4f} ({report['comparison']['best_val_loss']['experiment']})")
        
        if 'best_overall_score' in report["comparison"]:
            logger.info(f"Best overall score: {report['comparison']['best_overall_score']['value']:.4f} ({report['comparison']['best_overall_score']['experiment']})")


def get_default_experiments() -> List[Dict[str, Any]]:
    """Get default experiment configurations.
    
    Returns:
        List of experiment configurations
    """
    return [
        {
            "name": "baseline",
            "description": "Baseline model with default parameters",
            "model_config": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "use_decoder": False
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "num_epochs": 5,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": True,
                "use_cosine_scheduler": False,
                "use_augmentation": False,
                "max_length": 512,
                "stride": 256,
                "val_split": 0.1
            },
            "generation_config": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "use_templates": False
            }
        },
        {
            "name": "improved_architecture",
            "description": "Improved model architecture with decoder layers",
            "model_config": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "use_decoder": True
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "num_epochs": 5,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": True,
                "use_cosine_scheduler": False,
                "use_augmentation": False,
                "max_length": 512,
                "stride": 256,
                "val_split": 0.1
            },
            "generation_config": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "use_templates": False
            }
        },
        {
            "name": "improved_training",
            "description": "Improved training techniques with cosine scheduler and data augmentation",
            "model_config": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "use_decoder": False
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "num_epochs": 5,
                "gradient_accumulation_steps": 2,
                "use_mixed_precision": True,
                "use_cosine_scheduler": True,
                "use_augmentation": True,
                "max_length": 512,
                "stride": 256,
                "val_split": 0.1
            },
            "generation_config": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "use_templates": False
            }
        },
        {
            "name": "improved_generation",
            "description": "Improved generation techniques with nucleus sampling and templates",
            "model_config": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "use_decoder": False
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "num_epochs": 5,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": True,
                "use_cosine_scheduler": False,
                "use_augmentation": False,
                "max_length": 512,
                "stride": 256,
                "val_split": 0.1
            },
            "generation_config": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.5,
                "use_templates": True
            }
        },
        {
            "name": "full_improvements",
            "description": "All improvements combined",
            "model_config": {
                "d_model": 512,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "use_decoder": True
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "num_epochs": 5,
                "gradient_accumulation_steps": 2,
                "use_mixed_precision": True,
                "use_cosine_scheduler": True,
                "use_augmentation": True,
                "max_length": 512,
                "stride": 256,
                "val_split": 0.1
            },
            "generation_config": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.5,
                "use_templates": True
            }
        }
    ]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Amharic language model experiments")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Directory to save experiment results")
    parser.add_argument("--tokenizer_path", type=str, help="Path to pretrained tokenizer")
    parser.add_argument("--base_model_path", type=str, help="Path to base model for comparison")
    parser.add_argument("--test_prompts_path", type=str, help="Path to test prompts for generation")
    parser.add_argument("--config_path", type=str, help="Path to experiment configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        base_model_path=args.base_model_path,
        test_prompts_path=args.test_prompts_path,
        device=args.device
    )
    
    # Load experiment configurations
    if args.config_path and os.path.exists(args.config_path):
        logger.info(f"Loading experiment configurations from {args.config_path}")
        with open(args.config_path, 'r') as f:
            experiments_config = json.load(f)
    else:
        logger.info("Using default experiment configurations")
        experiments_config = get_default_experiments()
    
    # Run experiments
    runner.run_experiments(experiments_config)


if __name__ == "__main__":
    main()