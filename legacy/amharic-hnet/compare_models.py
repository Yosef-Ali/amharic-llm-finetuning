#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison script for original and improved Amharic language models.

This script compares the performance of the original HNet model with the
improved version using various metrics and generation examples.
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import original model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_train import EnhancedHNet, AmharicHNetDataset
from generate import generate_text as original_generate_text

# Import improved model components
from improved_model import HNetTransformer
from hybrid_tokenizer import HybridAmharicTokenizer
from improved_training import AmharicDataset, collate_fn
from improved_generation import ImprovedAmharicGenerator
from linguistic_quality_metrics import AmharicLinguisticEvaluator
from model_optimization import ModelOptimizer, benchmark_models

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Comparator for original and improved Amharic language models."""
    
    def __init__(self, 
                 original_model_path: str,
                 improved_model_path: str,
                 original_tokenizer_path: str,
                 improved_tokenizer_path: str,
                 test_prompts_path: Optional[str] = None,
                 test_data_dir: Optional[str] = None,
                 output_dir: str = "./comparison_results",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model comparator.
        
        Args:
            original_model_path: Path to the original model
            improved_model_path: Path to the improved model
            original_tokenizer_path: Path to the original tokenizer
            improved_tokenizer_path: Path to the improved tokenizer
            test_prompts_path: Path to test prompts for generation (optional)
            test_data_dir: Directory containing test data files (optional)
            output_dir: Directory to save comparison results
            device: Device to use
        """
        self.original_model_path = original_model_path
        self.improved_model_path = improved_model_path
        self.original_tokenizer_path = original_tokenizer_path
        self.improved_tokenizer_path = improved_tokenizer_path
        self.test_prompts_path = test_prompts_path
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test prompts
        self.test_prompts = self._load_test_prompts(test_prompts_path)
        
        # Load models and tokenizers
        self._load_models_and_tokenizers()
        
        # Create evaluator
        self.linguistic_evaluator = AmharicLinguisticEvaluator()
    
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
    
    def _load_models_and_tokenizers(self):
        """Load models and tokenizers."""
        # Load original model and tokenizer
        logger.info(f"Loading original model from {self.original_model_path}")
        try:
            # Load original model
            self.original_model = EnhancedHNet.load_from_checkpoint(self.original_model_path)
            self.original_model.to(self.device)
            self.original_model.eval()
            
            # Load original tokenizer
            # Note: The original tokenizer is typically loaded within the generate_text function
            # We'll keep a reference to the path for use with the original generation function
            logger.info(f"Original model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading original model: {e}")
            self.original_model = None
        
        # Load improved model and tokenizer
        logger.info(f"Loading improved model from {self.improved_model_path}")
        try:
            # Load improved tokenizer
            self.improved_tokenizer = HybridAmharicTokenizer.from_pretrained(self.improved_tokenizer_path)
            
            # Load improved model
            self.improved_model = HNetTransformer.from_pretrained(self.improved_model_path, device=self.device)
            self.improved_model.eval()
            
            # Create generator
            self.improved_generator = ImprovedAmharicGenerator(
                model=self.improved_model,
                tokenizer=self.improved_tokenizer,
                device=self.device
            )
            
            logger.info(f"Improved model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading improved model: {e}")
            self.improved_model = None
            self.improved_tokenizer = None
            self.improved_generator = None
    
    def compare(self):
        """Compare the models."""
        logger.info("Comparing original and improved models")
        
        # Check if models are loaded
        if self.original_model is None or self.improved_model is None:
            logger.error("One or both models failed to load. Cannot compare.")
            return
        
        # Compare generation quality
        generation_comparison = self.compare_generation()
        
        # Compare model size and inference speed
        performance_comparison = self.compare_performance()
        
        # Compare model optimization
        optimization_comparison = self.compare_optimization()
        
        # Prepare comparison results
        comparison_results = {
            "original_model_path": self.original_model_path,
            "improved_model_path": self.improved_model_path,
            "generation_comparison": generation_comparison,
            "performance_comparison": performance_comparison,
            "optimization_comparison": optimization_comparison
        }
        
        # Save comparison results
        results_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to {results_path}")
        
        # Generate comparison charts
        self.generate_comparison_charts(comparison_results)
        
        # Print summary
        self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def compare_generation(self) -> Dict[str, Any]:
        """Compare text generation quality.
        
        Returns:
            Dictionary of generation comparison results
        """
        logger.info("Comparing text generation quality...")
        
        # Generate text for each prompt
        generation_results = {}
        original_texts = []
        improved_texts = []
        
        for prompt in tqdm(self.test_prompts, desc="Generating text"):
            # Generate text with original model
            try:
                start_time = time.time()
                original_text = original_generate_text(
                    prompt=prompt,
                    model_path=self.original_model_path,
                    max_length=100
                )
                original_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Error generating text with original model: {e}")
                original_text = ""
                original_time = 0
            
            original_texts.append(original_text)
            
            # Generate text with improved model
            try:
                start_time = time.time()
                improved_text = self.improved_generator.generate(
                    prompt=prompt,
                    max_length=100,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.5,
                    use_template=True
                )
                improved_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Error generating text with improved model: {e}")
                improved_text = ""
                improved_time = 0
            
            improved_texts.append(improved_text)
            
            # Evaluate linguistic quality
            original_scores = self.linguistic_evaluator.evaluate_text(original_text) if original_text else {}
            improved_scores = self.linguistic_evaluator.evaluate_text(improved_text) if improved_text else {}
            
            # Add to generation results
            generation_results[prompt] = {
                "original": {
                    "text": original_text,
                    "generation_time": original_time,
                    "linguistic_scores": original_scores
                },
                "improved": {
                    "text": improved_text,
                    "generation_time": improved_time,
                    "linguistic_scores": improved_scores
                }
            }
        
        # Calculate average linguistic scores
        original_avg_scores = self._calculate_average_scores([result["original"]["linguistic_scores"] 
                                                         for result in generation_results.values()])
        improved_avg_scores = self._calculate_average_scores([result["improved"]["linguistic_scores"] 
                                                         for result in generation_results.values()])
        
        # Calculate average generation time
        original_avg_time = sum([result["original"]["generation_time"] for result in generation_results.values()]) / len(generation_results)
        improved_avg_time = sum([result["improved"]["generation_time"] for result in generation_results.values()]) / len(generation_results)
        
        return {
            "prompts": self.test_prompts,
            "generation_results": generation_results,
            "original_avg_scores": original_avg_scores,
            "improved_avg_scores": improved_avg_scores,
            "original_avg_generation_time": original_avg_time,
            "improved_avg_generation_time": improved_avg_time
        }
    
    def compare_performance(self) -> Dict[str, Any]:
        """Compare model size and inference speed.
        
        Returns:
            Dictionary of performance comparison results
        """
        logger.info("Comparing model size and inference speed...")
        
        # Get model size
        original_size = self._get_model_size(self.original_model)
        improved_size = self._get_model_size(self.improved_model)
        
        # Measure inference speed
        original_inference_time = self._measure_inference_speed(self.original_model)
        improved_inference_time = self._measure_inference_speed(self.improved_model)
        
        return {
            "original_model_size": original_size,
            "improved_model_size": improved_size,
            "original_inference_time": original_inference_time,
            "improved_inference_time": improved_inference_time
        }
    
    def compare_optimization(self) -> Dict[str, Any]:
        """Compare model optimization.
        
        Returns:
            Dictionary of optimization comparison results
        """
        logger.info("Comparing model optimization...")
        
        # Only optimize the improved model
        if self.improved_model is None:
            return {}
        
        try:
            # Create model optimizer
            optimizer = ModelOptimizer(self.improved_model, self.improved_tokenizer)
            
            # Apply dynamic quantization
            quantized_model = optimizer.apply_dynamic_quantization()
            
            # Benchmark models
            benchmark_results = benchmark_models(
                models={
                    "original": self.improved_model,
                    "quantized": quantized_model
                },
                tokenizer=self.improved_tokenizer,
                input_texts=self.test_prompts,
                device=self.device
            )
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error comparing optimization: {e}")
            return {}
    
    def generate_comparison_charts(self, comparison_results: Dict[str, Any]):
        """Generate comparison charts.
        
        Args:
            comparison_results: Comparison results
        """
        logger.info("Generating comparison charts...")
        
        # Create charts directory
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate linguistic scores chart
        self._generate_linguistic_scores_chart(comparison_results, charts_dir)
        
        # Generate generation time chart
        self._generate_generation_time_chart(comparison_results, charts_dir)
        
        # Generate model size chart
        self._generate_model_size_chart(comparison_results, charts_dir)
        
        # Generate inference time chart
        self._generate_inference_time_chart(comparison_results, charts_dir)
    
    def _generate_linguistic_scores_chart(self, comparison_results: Dict[str, Any], charts_dir: str):
        """Generate linguistic scores chart.
        
        Args:
            comparison_results: Comparison results
            charts_dir: Directory to save charts
        """
        try:
            # Get scores
            original_scores = comparison_results["generation_comparison"]["original_avg_scores"]
            improved_scores = comparison_results["generation_comparison"]["improved_avg_scores"]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Set metrics and values
            metrics = list(original_scores.keys())
            original_values = [original_scores[metric] for metric in metrics]
            improved_values = [improved_scores[metric] for metric in metrics]
            
            # Set positions
            x = np.arange(len(metrics))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, original_values, width, label='Original Model')
            plt.bar(x + width/2, improved_values, width, label='Improved Model')
            
            # Add labels and title
            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('Linguistic Quality Scores Comparison')
            plt.xticks(x, [metric.replace('_score', '').capitalize() for metric in metrics])
            plt.ylim(0, 1.0)
            plt.legend()
            
            # Add value labels
            for i, v in enumerate(original_values):
                plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
            
            for i, v in enumerate(improved_values):
                plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'linguistic_scores.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error generating linguistic scores chart: {e}")
    
    def _generate_generation_time_chart(self, comparison_results: Dict[str, Any], charts_dir: str):
        """Generate generation time chart.
        
        Args:
            comparison_results: Comparison results
            charts_dir: Directory to save charts
        """
        try:
            # Get generation times
            original_time = comparison_results["generation_comparison"]["original_avg_generation_time"]
            improved_time = comparison_results["generation_comparison"]["improved_avg_generation_time"]
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Create bars
            plt.bar(['Original Model', 'Improved Model'], [original_time, improved_time])
            
            # Add labels and title
            plt.xlabel('Model')
            plt.ylabel('Average Generation Time (seconds)')
            plt.title('Text Generation Time Comparison')
            
            # Add value labels
            plt.text(0, original_time + 0.1, f'{original_time:.2f}s', ha='center')
            plt.text(1, improved_time + 0.1, f'{improved_time:.2f}s', ha='center')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'generation_time.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error generating generation time chart: {e}")
    
    def _generate_model_size_chart(self, comparison_results: Dict[str, Any], charts_dir: str):
        """Generate model size chart.
        
        Args:
            comparison_results: Comparison results
            charts_dir: Directory to save charts
        """
        try:
            # Get model sizes
            original_size = comparison_results["performance_comparison"]["original_model_size"]
            improved_size = comparison_results["performance_comparison"]["improved_model_size"]
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Create bars
            plt.bar(['Original Model', 'Improved Model'], [original_size, improved_size])
            
            # Add labels and title
            plt.xlabel('Model')
            plt.ylabel('Model Size (MB)')
            plt.title('Model Size Comparison')
            
            # Add value labels
            plt.text(0, original_size + 1, f'{original_size:.2f} MB', ha='center')
            plt.text(1, improved_size + 1, f'{improved_size:.2f} MB', ha='center')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'model_size.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error generating model size chart: {e}")
    
    def _generate_inference_time_chart(self, comparison_results: Dict[str, Any], charts_dir: str):
        """Generate inference time chart.
        
        Args:
            comparison_results: Comparison results
            charts_dir: Directory to save charts
        """
        try:
            # Get inference times
            original_time = comparison_results["performance_comparison"]["original_inference_time"]
            improved_time = comparison_results["performance_comparison"]["improved_inference_time"]
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
            # Create bars
            plt.bar(['Original Model', 'Improved Model'], [original_time, improved_time])
            
            # Add labels and title
            plt.xlabel('Model')
            plt.ylabel('Inference Time (ms)')
            plt.title('Inference Time Comparison')
            
            # Add value labels
            plt.text(0, original_time + 1, f'{original_time:.2f} ms', ha='center')
            plt.text(1, improved_time + 1, f'{improved_time:.2f} ms', ha='center')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'inference_time.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error generating inference time chart: {e}")
    
    def _get_model_size(self, model) -> float:
        """Get model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        try:
            # Get model state dict
            state_dict = model.state_dict()
            
            # Calculate size
            model_size_bytes = sum(param.nelement() * param.element_size() for param in state_dict.values())
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            return model_size_mb
        except Exception as e:
            logger.error(f"Error getting model size: {e}")
            return 0.0
    
    def _measure_inference_speed(self, model) -> float:
        """Measure inference speed in milliseconds.
        
        Args:
            model: PyTorch model
            
        Returns:
            Average inference time in milliseconds
        """
        try:
            # Create dummy input
            batch_size = 1
            seq_length = 32
            
            if isinstance(model, EnhancedHNet):
                # Original model
                vocab_size = model.vocab_size
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
            else:
                # Improved model
                vocab_size = model.vocab_size
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
            
            # Warm-up
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Measure inference time
            num_runs = 20
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            end_time = time.time()
            
            # Calculate average inference time in milliseconds
            avg_inference_time = (end_time - start_time) * 1000 / num_runs
            
            return avg_inference_time
        except Exception as e:
            logger.error(f"Error measuring inference speed: {e}")
            return 0.0
    
    def _calculate_average_scores(self, scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average scores.
        
        Args:
            scores_list: List of score dictionaries
            
        Returns:
            Dictionary of average scores
        """
        # Initialize average scores
        avg_scores = {}
        
        # Check if scores list is empty
        if not scores_list:
            return avg_scores
        
        # Get all metrics
        all_metrics = set()
        for scores in scores_list:
            all_metrics.update(scores.keys())
        
        # Calculate average for each metric
        for metric in all_metrics:
            values = [scores.get(metric, 0.0) for scores in scores_list if metric in scores]
            avg_scores[metric] = sum(values) / len(values) if values else 0.0
        
        return avg_scores
    
    def _print_comparison_summary(self, comparison_results: Dict[str, Any]):
        """Print comparison summary.
        
        Args:
            comparison_results: Comparison results
        """
        logger.info("\n" + "="*50)
        logger.info("Comparison Summary")
        logger.info("="*50)
        
        # Print linguistic scores comparison
        logger.info("\nLinguistic Quality Scores:")
        original_scores = comparison_results["generation_comparison"]["original_avg_scores"]
        improved_scores = comparison_results["generation_comparison"]["improved_avg_scores"]
        
        for metric in original_scores.keys():
            original_value = original_scores.get(metric, 0.0)
            improved_value = improved_scores.get(metric, 0.0)
            difference = improved_value - original_value
            percentage = (difference / original_value * 100) if original_value > 0 else 0
            
            logger.info(f"  {metric.replace('_score', '').capitalize()}:")
            logger.info(f"    Original: {original_value:.4f}")
            logger.info(f"    Improved: {improved_value:.4f}")
            logger.info(f"    Difference: {difference:.4f} ({percentage:+.2f}%)")
        
        # Print generation time comparison
        logger.info("\nGeneration Time:")
        original_time = comparison_results["generation_comparison"]["original_avg_generation_time"]
        improved_time = comparison_results["generation_comparison"]["improved_avg_generation_time"]
        time_difference = improved_time - original_time
        time_percentage = (time_difference / original_time * 100) if original_time > 0 else 0
        
        logger.info(f"  Original: {original_time:.4f} seconds")
        logger.info(f"  Improved: {improved_time:.4f} seconds")
        logger.info(f"  Difference: {time_difference:.4f} seconds ({time_percentage:+.2f}%)")
        
        # Print model size comparison
        logger.info("\nModel Size:")
        original_size = comparison_results["performance_comparison"]["original_model_size"]
        improved_size = comparison_results["performance_comparison"]["improved_model_size"]
        size_difference = improved_size - original_size
        size_percentage = (size_difference / original_size * 100) if original_size > 0 else 0
        
        logger.info(f"  Original: {original_size:.4f} MB")
        logger.info(f"  Improved: {improved_size:.4f} MB")
        logger.info(f"  Difference: {size_difference:.4f} MB ({size_percentage:+.2f}%)")
        
        # Print inference time comparison
        logger.info("\nInference Time:")
        original_inference = comparison_results["performance_comparison"]["original_inference_time"]
        improved_inference = comparison_results["performance_comparison"]["improved_inference_time"]
        inference_difference = improved_inference - original_inference
        inference_percentage = (inference_difference / original_inference * 100) if original_inference > 0 else 0
        
        logger.info(f"  Original: {original_inference:.4f} ms")
        logger.info(f"  Improved: {improved_inference:.4f} ms")
        logger.info(f"  Difference: {inference_difference:.4f} ms ({inference_percentage:+.2f}%)")
        
        # Print generation examples
        logger.info("\nGeneration Examples:")
        
        # Select a random prompt
        import random
        prompt = random.choice(comparison_results["generation_comparison"]["prompts"])
        
        logger.info(f"\nPrompt: {prompt}")
        
        # Print generated texts
        original_text = comparison_results["generation_comparison"]["generation_results"][prompt]["original"]["text"]
        improved_text = comparison_results["generation_comparison"]["generation_results"][prompt]["improved"]["text"]
        
        logger.info(f"\nOriginal model generation:")
        logger.info(f"{original_text[:200]}..." if len(original_text) > 200 else original_text)
        
        logger.info(f"\nImproved model generation:")
        logger.info(f"{improved_text[:200]}..." if len(improved_text) > 200 else improved_text)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare original and improved Amharic language models")
    
    # Model arguments
    parser.add_argument("--original_model_path", type=str, required=True, help="Path to the original model")
    parser.add_argument("--improved_model_path", type=str, required=True, help="Path to the improved model")
    parser.add_argument("--original_tokenizer_path", type=str, required=True, help="Path to the original tokenizer")
    parser.add_argument("--improved_tokenizer_path", type=str, required=True, help="Path to the improved tokenizer")
    
    # Data arguments
    parser.add_argument("--test_prompts_path", type=str, help="Path to test prompts for generation")
    parser.add_argument("--test_data_dir", type=str, help="Directory containing test data files")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./comparison_results", help="Directory to save comparison results")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ModelComparator(
        original_model_path=args.original_model_path,
        improved_model_path=args.improved_model_path,
        original_tokenizer_path=args.original_tokenizer_path,
        improved_tokenizer_path=args.improved_tokenizer_path,
        test_prompts_path=args.test_prompts_path,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Compare models
    comparator.compare()


if __name__ == "__main__":
    main()