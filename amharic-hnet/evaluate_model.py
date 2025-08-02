#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for Amharic language models.

This script evaluates the performance of Amharic language models using
linguistic quality metrics and perplexity.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import model and evaluation components
from improved_model import HNetTransformer
from hybrid_tokenizer import HybridAmharicTokenizer
from improved_training import AmharicDataset, collate_fn
from improved_generation import ImprovedAmharicGenerator
from linguistic_quality_metrics import AmharicLinguisticEvaluator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for Amharic language models."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: str,
                 test_data_dir: Optional[str] = None,
                 test_prompts_path: Optional[str] = None,
                 reference_texts_path: Optional[str] = None,
                 output_dir: str = "./evaluation_results",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model evaluator.
        
        Args:
            model_path: Path to the model to evaluate
            tokenizer_path: Path to the tokenizer
            test_data_dir: Directory containing test data files (optional)
            test_prompts_path: Path to test prompts for generation (optional)
            reference_texts_path: Path to reference texts for comparison (optional)
            output_dir: Directory to save evaluation results
            device: Device to use
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.test_data_dir = test_data_dir
        self.test_prompts_path = test_prompts_path
        self.reference_texts_path = reference_texts_path
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = HybridAmharicTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = HNetTransformer.from_pretrained(model_path, device=device)
        
        # Create generator
        self.generator = ImprovedAmharicGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        
        # Create evaluator
        self.linguistic_evaluator = AmharicLinguisticEvaluator()
        
        # Load test prompts
        self.test_prompts = self._load_test_prompts(test_prompts_path)
        
        # Load reference texts
        self.reference_texts = self._load_reference_texts(reference_texts_path)
    
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
    
    def _load_reference_texts(self, reference_texts_path: Optional[str]) -> Dict[str, str]:
        """Load reference texts for comparison.
        
        Args:
            reference_texts_path: Path to reference texts file
            
        Returns:
            Dictionary mapping prompts to reference texts
        """
        if not reference_texts_path or not os.path.exists(reference_texts_path):
            logger.info(f"Reference texts file not found. No reference texts will be used.")
            return {}
        
        try:
            with open(reference_texts_path, 'r', encoding='utf-8') as f:
                reference_texts = json.load(f)
            
            if not isinstance(reference_texts, dict):
                logger.warning(f"Reference texts file must contain a JSON object. No reference texts will be used.")
                return {}
            
            logger.info(f"Loaded {len(reference_texts)} reference texts from {reference_texts_path}")
            return reference_texts
        except Exception as e:
            logger.error(f"Error loading reference texts from {reference_texts_path}: {e}")
            return {}
    
    def evaluate(self):
        """Evaluate the model."""
        logger.info(f"Evaluating model from {self.model_path}")
        
        # Evaluate perplexity if test data is available
        perplexity = None
        if self.test_data_dir and os.path.exists(self.test_data_dir):
            perplexity = self.evaluate_perplexity()
        
        # Generate text and evaluate linguistic quality
        generation_results, linguistic_metrics = self.evaluate_generation()
        
        # Prepare evaluation results
        evaluation_results = {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "perplexity": perplexity,
            "linguistic_metrics": linguistic_metrics,
            "generation_results": generation_results
        }
        
        # Save evaluation results
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def evaluate_perplexity(self) -> float:
        """Evaluate model perplexity on test data.
        
        Returns:
            Perplexity score
        """
        logger.info("Evaluating perplexity...")
        
        # Prepare test dataset
        test_dataset = self._prepare_test_dataset()
        
        # Create data loader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Evaluate perplexity
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Calculating perplexity"):
                # Get batch
                input_ids, target_ids = batch
                
                # Move tensors to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                outputs = outputs.contiguous().view(-1, self.model.vocab_size)
                targets = target_ids.contiguous().view(-1)
                
                # Create mask for non-padding tokens
                mask = (targets != self.model.pad_idx).float()
                
                # Calculate cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    outputs, targets, reduction='none', ignore_index=self.model.pad_idx
                )
                
                # Apply mask and sum
                masked_loss = loss * mask
                total_loss += masked_loss.sum().item()
                total_tokens += mask.sum().item()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        return perplexity
    
    def evaluate_generation(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Evaluate text generation quality.
        
        Returns:
            Tuple of generation results and linguistic metrics
        """
        logger.info("Evaluating text generation...")
        
        # Generate text for each prompt
        generation_results = {}
        all_generated_texts = []
        
        for prompt in tqdm(self.test_prompts, desc="Generating text"):
            # Generate text with different settings
            results = {}
            
            # Standard generation
            standard_text = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                use_template=False
            )
            results["standard"] = {
                "text": standard_text,
                "settings": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "use_template": False
                }
            }
            all_generated_texts.append(standard_text)
            
            # Nucleus sampling
            nucleus_text = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                use_template=False
            )
            results["nucleus"] = {
                "text": nucleus_text,
                "settings": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "repetition_penalty": 1.2,
                    "use_template": False
                }
            }
            all_generated_texts.append(nucleus_text)
            
            # Enhanced repetition penalty
            rep_penalty_text = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                use_template=False
            )
            results["rep_penalty"] = {
                "text": rep_penalty_text,
                "settings": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.5,
                    "use_template": False
                }
            }
            all_generated_texts.append(rep_penalty_text)
            
            # Template-based generation
            template_text = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                use_template=True
            )
            results["template"] = {
                "text": template_text,
                "settings": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "use_template": True
                }
            }
            all_generated_texts.append(template_text)
            
            # Combined improvements
            combined_text = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.5,
                use_template=True
            )
            results["combined"] = {
                "text": combined_text,
                "settings": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "repetition_penalty": 1.5,
                    "use_template": True
                }
            }
            all_generated_texts.append(combined_text)
            
            # Add reference text if available
            if prompt in self.reference_texts:
                results["reference"] = {
                    "text": self.reference_texts[prompt]
                }
            
            # Add to generation results
            generation_results[prompt] = results
        
        # Evaluate linguistic quality
        linguistic_metrics = self._evaluate_linguistic_quality(all_generated_texts)
        
        return generation_results, linguistic_metrics
    
    def _prepare_test_dataset(self) -> AmharicDataset:
        """Prepare test dataset.
        
        Returns:
            Test dataset
        """
        # Find all text and JSON files in the test data directory
        file_paths = []
        for root, _, files in os.walk(self.test_data_dir):
            for file in files:
                if file.endswith(('.txt', '.json')):
                    file_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(file_paths)} test files in {self.test_data_dir}")
        
        # Create dataset
        test_dataset = AmharicDataset(
            file_paths=file_paths,
            tokenizer=self.tokenizer,
            max_length=512,
            stride=256,
            use_augmentation=False
        )
        
        return test_dataset
    
    def _evaluate_linguistic_quality(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate linguistic quality of generated texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary of linguistic quality metrics
        """
        # Initialize metrics
        metrics = {
            'grammar_score': 0.0,
            'coherence_score': 0.0,
            'repetition_score': 0.0,
            'cultural_relevance_score': 0.0,
            'overall_score': 0.0
        }
        
        # Evaluate each text
        num_texts = 0
        for text in texts:
            # Skip empty texts
            if not text:
                continue
            
            # Evaluate text
            scores = self.linguistic_evaluator.evaluate_text(text)
            
            # Accumulate scores
            for key in metrics.keys():
                metrics[key] += scores.get(key, 0.0)
            
            num_texts += 1
        
        # Calculate average scores
        if num_texts > 0:
            for key in metrics.keys():
                metrics[key] /= num_texts
        
        return metrics
    
    def _print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print evaluation summary.
        
        Args:
            evaluation_results: Evaluation results
        """
        logger.info("\n" + "="*50)
        logger.info("Evaluation Summary")
        logger.info("="*50)
        
        # Print perplexity
        if evaluation_results["perplexity"] is not None:
            logger.info(f"Perplexity: {evaluation_results['perplexity']:.4f}")
        
        # Print linguistic metrics
        if evaluation_results["linguistic_metrics"]:
            logger.info("\nLinguistic Metrics:")
            for metric, value in evaluation_results["linguistic_metrics"].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Print generation examples
        if evaluation_results["generation_results"]:
            logger.info("\nGeneration Examples:")
            
            # Select a random prompt
            import random
            prompt = random.choice(list(evaluation_results["generation_results"].keys()))
            
            logger.info(f"\nPrompt: {prompt}")
            
            # Print generated texts
            for method, result in evaluation_results["generation_results"][prompt].items():
                if method != "reference":
                    logger.info(f"\n{method.capitalize()} generation:")
                    logger.info(f"{result['text'][:200]}..." if len(result['text']) > 200 else result['text'])
            
            # Print reference text if available
            if "reference" in evaluation_results["generation_results"][prompt]:
                logger.info(f"\nReference text:")
                reference_text = evaluation_results["generation_results"][prompt]["reference"]["text"]
                logger.info(f"{reference_text[:200]}..." if len(reference_text) > 200 else reference_text)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate Amharic language model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    
    # Data arguments
    parser.add_argument("--test_data_dir", type=str, help="Directory containing test data files")
    parser.add_argument("--test_prompts_path", type=str, help="Path to test prompts for generation")
    parser.add_argument("--reference_texts_path", type=str, help="Path to reference texts for comparison")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        test_data_dir=args.test_data_dir,
        test_prompts_path=args.test_prompts_path,
        reference_texts_path=args.reference_texts_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Evaluate model
    evaluator.evaluate()


if __name__ == "__main__":
    main()