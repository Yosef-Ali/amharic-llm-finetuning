#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for Amharic H-Net model.

This script provides comprehensive tests for all components of the Amharic H-Net model,
including model architecture, training, generation, evaluation, and optimization.
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TestAmharicPreprocessor(unittest.TestCase):
    """Test Amharic text preprocessor."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from amharic_preprocessor import AmharicTextPreprocessor
            self.preprocessor = AmharicTextPreprocessor()
        except ImportError:
            self.skipTest("AmharicTextPreprocessor not available")
    
    def test_remove_non_amharic(self):
        """Test removing non-Amharic characters."""
        text = "ሰላም123Hello"
        expected = "ሰላም"
        result = self.preprocessor.clean_text(text, remove_non_amharic=True)
        self.assertEqual(result, expected)
    
    def test_normalize_spaces(self):
        """Test normalizing spaces."""
        text = "ሰላም  ኢትዮጵያ\t\nአዲስ አበባ"
        expected = "ሰላም ኢትዮጵያ አዲስ አበባ"
        result = self.preprocessor.clean_text(text, normalize_spaces=True)
        self.assertEqual(result, expected)
    
    def test_normalize_punctuation(self):
        """Test normalizing punctuation."""
        text = "ሰላም፡፡ ኢትዮጵያ፥"
        result = self.preprocessor.clean_text(text, normalize_punctuation=True)
        self.assertNotEqual(result, text)  # Punctuation should be normalized
    
    def test_remove_urls(self):
        """Test removing URLs."""
        text = "ሰላም https://example.com ኢትዮጵያ"
        expected = "ሰላም  ኢትዮጵያ"
        result = self.preprocessor.clean_text(text, remove_urls=True)
        self.assertEqual(result, expected)
    
    def test_remove_emails(self):
        """Test removing emails."""
        text = "ሰላም user@example.com ኢትዮጵያ"
        expected = "ሰላም  ኢትዮጵያ"
        result = self.preprocessor.clean_text(text, remove_emails=True)
        self.assertEqual(result, expected)
    
    def test_remove_numbers(self):
        """Test removing numbers."""
        text = "ሰላም 123 ኢትዮጵያ"
        expected = "ሰላም  ኢትዮጵያ"
        result = self.preprocessor.clean_text(text, remove_numbers=True)
        self.assertEqual(result, expected)
    
    def test_filter_by_length(self):
        """Test filtering by length."""
        text = "ሰላም"
        result = self.preprocessor.clean_text(text, min_length=10)
        self.assertEqual(result, "")  # Text should be filtered out


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from data_augmentation import AmharicDataAugmenter
            self.augmenter = AmharicDataAugmenter()
        except ImportError:
            self.skipTest("AmharicDataAugmenter not available")
    
    def test_character_swap(self):
        """Test character swapping."""
        text = "ሰላም ኢትዮጵያ"
        result = self.augmenter.character_swap(text)
        self.assertNotEqual(result, text)  # Text should be augmented
    
    def test_word_dropout(self):
        """Test word dropout."""
        text = "ሰላም ኢትዮጵያ አዲስ አበባ"
        result = self.augmenter.word_dropout(text)
        self.assertNotEqual(result, text)  # Text should be augmented
    
    def test_word_swap(self):
        """Test word swapping."""
        text = "ሰላም ኢትዮጵያ አዲስ አበባ"
        result = self.augmenter.word_swap(text)
        self.assertNotEqual(result, text)  # Text should be augmented
    
    def test_synonym_replacement(self):
        """Test synonym replacement."""
        text = "ሰላም ኢትዮጵያ"
        result = self.augmenter.synonym_replacement(text)
        self.assertEqual(result, text)  # No synonyms available, should return original text


class TestAmharicDataset(unittest.TestCase):
    """Test Amharic dataset."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from improved_training import AmharicDataset
            from transformers import AutoTokenizer
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Create test data
            self.test_data = ["ሰላም ኢትዮጵያ", "አዲስ አበባ"]
            self.test_file = os.path.join(self.temp_dir, "test_data.txt")
            with open(self.test_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.test_data))
            
            # Create tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create dataset
            self.dataset = AmharicDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                max_length=10,
                sliding_window=True,
                window_size=5,
                window_stride=2,
                data_augmentation=False,
            )
        except ImportError:
            self.skipTest("AmharicDataset not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), len(self.test_data))
    
    def test_dataset_getitem(self):
        """Test dataset getitem."""
        item = self.dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
    
    def test_sliding_window(self):
        """Test sliding window."""
        # Create dataset with sliding window
        dataset = type(self.dataset)(
            file_path=self.test_file,
            tokenizer=self.tokenizer,
            max_length=10,
            sliding_window=True,
            window_size=5,
            window_stride=2,
            data_augmentation=False,
        )
        
        # Dataset length should be greater than original data length
        self.assertGreaterEqual(len(dataset), len(self.test_data))
    
    def test_data_augmentation(self):
        """Test data augmentation."""
        # Create dataset with data augmentation
        dataset = type(self.dataset)(
            file_path=self.test_file,
            tokenizer=self.tokenizer,
            max_length=10,
            sliding_window=False,
            data_augmentation=True,
            augmentation_prob=1.0,
        )
        
        # Dataset length should be greater than original data length
        self.assertGreaterEqual(len(dataset), len(self.test_data))


class TestHNetTransformer(unittest.TestCase):
    """Test HNetTransformer model."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from amharic_hnet.model import HNetTransformer, HNetConfig
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            self.model = HNetTransformer(self.config)
        except ImportError:
            self.skipTest("HNetTransformer not available")
    
    def test_model_forward(self):
        """Test model forward pass."""
        # Create dummy input
        batch_size = 2
        sequence_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, sequence_length))
        attention_mask = torch.ones((batch_size, sequence_length))
        
        # Run forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Check outputs
        self.assertIn("logits", outputs)
        self.assertEqual(outputs.logits.shape, (batch_size, sequence_length, self.config.vocab_size))
    
    def test_model_generate(self):
        """Test model generation."""
        # Create dummy input
        batch_size = 2
        sequence_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, sequence_length))
        attention_mask = torch.ones((batch_size, sequence_length))
        
        # Run generation
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=sequence_length + 5,
            num_beams=2,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        
        # Check outputs
        self.assertEqual(generated_ids.shape[0], batch_size)
        self.assertGreater(generated_ids.shape[1], sequence_length)
    
    def test_model_save_load(self):
        """Test model save and load."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            self.model.save_pretrained(temp_dir)
            
            # Load model
            loaded_model = HNetTransformer.from_pretrained(temp_dir)
            
            # Check model
            self.assertEqual(loaded_model.config.vocab_size, self.model.config.vocab_size)
            self.assertEqual(loaded_model.config.hidden_size, self.model.config.hidden_size)
            self.assertEqual(loaded_model.config.num_hidden_layers, self.model.config.num_hidden_layers)
            self.assertEqual(loaded_model.config.num_attention_heads, self.model.config.num_attention_heads)


class TestImprovedTrainer(unittest.TestCase):
    """Test ImprovedTrainer."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from improved_training import ImprovedTrainer
            from amharic_hnet.model import HNetTransformer, HNetConfig
            from transformers import AutoTokenizer
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            self.model = HNetTransformer(self.config)
            
            # Create tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Create test data
            self.test_data = ["ሰላም ኢትዮጵያ", "አዲስ አበባ"]
            self.test_file = os.path.join(self.temp_dir, "test_data.txt")
            with open(self.test_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.test_data))
            
            # Create trainer
            self.trainer = ImprovedTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_file=self.test_file,
                val_file=self.test_file,
                output_dir=self.temp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=5e-5,
                weight_decay=0.01,
                max_grad_norm=1.0,
                warmup_steps=0,
                logging_steps=10,
                save_steps=50,
                eval_steps=50,
                save_total_limit=2,
                fp16=False,
                max_length=10,
                sliding_window=False,
                data_augmentation=False,
            )
        except ImportError:
            self.skipTest("ImprovedTrainer not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_train(self):
        """Test trainer train."""
        # Train model
        self.trainer.train(max_steps=2)
        
        # Check if model is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "pytorch_model.bin")))
    
    def test_trainer_evaluate(self):
        """Test trainer evaluate."""
        # Evaluate model
        metrics = self.trainer.evaluate()
        
        # Check metrics
        self.assertIn("eval_loss", metrics)
        self.assertIn("eval_perplexity", metrics)


class TestAmharicGenerator(unittest.TestCase):
    """Test AmharicGenerator."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from amharic_hnet.generator import ImprovedAmharicGenerator
            from amharic_hnet.model import HNetTransformer, HNetConfig
            from transformers import AutoTokenizer
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            self.model = HNetTransformer(self.config)
            
            # Create tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create generator
            self.generator = ImprovedAmharicGenerator(self.model, self.tokenizer)
        except ImportError:
            self.skipTest("ImprovedAmharicGenerator not available")
    
    def test_generator_generate(self):
        """Test generator generate."""
        # Generate text
        prompt = "ሰላም"
        generated_texts = self.generator.generate(
            prompt,
            max_length=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=2,
        )
        
        # Check generated texts
        self.assertEqual(len(generated_texts), 2)
        self.assertIsInstance(generated_texts[0], str)
        self.assertIsInstance(generated_texts[1], str)
    
    def test_generator_batch_generate(self):
        """Test generator batch generate."""
        # Generate text
        prompts = ["ሰላም", "አዲስ"]
        generated_texts = self.generator.batch_generate(
            prompts,
            max_length=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
        )
        
        # Check generated texts
        self.assertEqual(len(generated_texts), len(prompts))
        self.assertIsInstance(generated_texts[0], str)
        self.assertIsInstance(generated_texts[1], str)


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from evaluate_model import ModelEvaluator
            from amharic_hnet.model import HNetTransformer, HNetConfig
            from transformers import AutoTokenizer
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            self.model = HNetTransformer(self.config)
            
            # Create tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Save model and tokenizer
            self.model.save_pretrained(self.temp_dir)
            self.tokenizer.save_pretrained(self.temp_dir)
            
            # Create test data
            self.test_prompts = ["ሰላም", "አዲስ"]
            self.test_prompts_file = os.path.join(self.temp_dir, "test_prompts.txt")
            with open(self.test_prompts_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.test_prompts))
            
            self.test_references = ["ሰላም ኢትዮጵያ", "አዲስ አበባ"]
            self.test_references_file = os.path.join(self.temp_dir, "test_references.txt")
            with open(self.test_references_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.test_references))
            
            # Create evaluator
            self.evaluator = ModelEvaluator(
                model_path=self.temp_dir,
                output_dir=self.temp_dir,
                prompts_file=self.test_prompts_file,
                references_file=self.test_references_file,
                max_length=20,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        except ImportError:
            self.skipTest("ModelEvaluator not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_evaluator_evaluate(self):
        """Test evaluator evaluate."""
        # Evaluate model
        results = self.evaluator.evaluate()
        
        # Check results
        self.assertEqual(len(results), len(self.test_prompts))
        self.assertIn("prompt", results[0])
        self.assertIn("generated_text", results[0])
        self.assertIn("reference_text", results[0])
        self.assertIn("perplexity", results[0])
        self.assertIn("grammar_score", results[0])
        self.assertIn("coherence_score", results[0])
        self.assertIn("repetition_score", results[0])
        self.assertIn("cultural_relevance_score", results[0])
        self.assertIn("overall_score", results[0])


class TestModelComparator(unittest.TestCase):
    """Test ModelComparator."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from compare_models import ModelComparator
            from amharic_hnet.model import HNetTransformer, HNetConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Create original model directory
            self.original_model_dir = os.path.join(self.temp_dir, "original")
            os.makedirs(self.original_model_dir, exist_ok=True)
            
            # Create improved model directory
            self.improved_model_dir = os.path.join(self.temp_dir, "improved")
            os.makedirs(self.improved_model_dir, exist_ok=True)
            
            # Create original model
            self.original_model = AutoModelForCausalLM.from_config(self.config)
            self.original_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create improved model
            self.improved_model = HNetTransformer(self.config)
            self.improved_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Save models and tokenizers
            self.original_model.save_pretrained(self.original_model_dir)
            self.original_tokenizer.save_pretrained(self.original_model_dir)
            self.improved_model.save_pretrained(self.improved_model_dir)
            self.improved_tokenizer.save_pretrained(self.improved_model_dir)
            
            # Create test data
            self.test_prompts = ["ሰላም", "አዲስ"]
            self.test_prompts_file = os.path.join(self.temp_dir, "test_prompts.txt")
            with open(self.test_prompts_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.test_prompts))
            
            # Create comparator
            self.comparator = ModelComparator(
                original_model_path=self.original_model_dir,
                improved_model_path=self.improved_model_dir,
                output_dir=self.temp_dir,
                prompts_file=self.test_prompts_file,
                max_length=20,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        except ImportError:
            self.skipTest("ModelComparator not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_comparator_compare(self):
        """Test comparator compare."""
        # Compare models
        results = self.comparator.compare()
        
        # Check results
        self.assertIn("original", results)
        self.assertIn("improved", results)
        self.assertIn("grammar_score", results["original"])
        self.assertIn("coherence_score", results["original"])
        self.assertIn("repetition_score", results["original"])
        self.assertIn("cultural_relevance_score", results["original"])
        self.assertIn("overall_score", results["original"])
        self.assertIn("model_size", results["original"])
        self.assertIn("inference_time", results["original"])
        self.assertIn("grammar_score", results["improved"])
        self.assertIn("coherence_score", results["improved"])
        self.assertIn("repetition_score", results["improved"])
        self.assertIn("cultural_relevance_score", results["improved"])
        self.assertIn("overall_score", results["improved"])
        self.assertIn("model_size", results["improved"])
        self.assertIn("inference_time", results["improved"])


class TestModelOptimizer(unittest.TestCase):
    """Test ModelOptimizer."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from optimize_model import ModelOptimizer
            from amharic_hnet.model import HNetTransformer, HNetConfig
            from transformers import AutoTokenizer
            
            # Create model configuration
            self.config = HNetConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            
            # Create model
            self.model = HNetTransformer(self.config)
            
            # Create tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Save model and tokenizer
            self.model.save_pretrained(self.temp_dir)
            self.tokenizer.save_pretrained(self.temp_dir)
            
            # Create optimizer
            self.optimizer = ModelOptimizer(
                model_path=self.temp_dir,
                output_dir=os.path.join(self.temp_dir, "optimized"),
                device="cpu",
            )
        except ImportError:
            self.skipTest("ModelOptimizer not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_optimizer_optimize_for_inference(self):
        """Test optimizer optimize for inference."""
        # Optimize model for inference
        self.optimizer.optimize_for_inference()
        
        # Check if optimized model is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "optimized", "optimized_model", "pytorch_model.bin")))
    
    def test_optimizer_convert_to_onnx(self):
        """Test optimizer convert to ONNX."""
        try:
            import onnx
            
            # Convert model to ONNX
            onnx_path = self.optimizer.convert_to_onnx()
            
            # Check if ONNX model is saved
            self.assertTrue(os.path.exists(onnx_path))
        except ImportError:
            self.skipTest("ONNX not available")


class TestResultVisualizer(unittest.TestCase):
    """Test ResultVisualizer."""
    
    def setUp(self):
        """Set up test case."""
        try:
            from visualize_results import ResultVisualizer
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Create test log file
            self.test_log_file = os.path.join(self.temp_dir, "test_log.txt")
            with open(self.test_log_file, "w", encoding="utf-8") as f:
                f.write("Epoch 1 Train Loss: 2.5\n")
                f.write("Validation Loss: 2.3\n")
                f.write("Epoch 2 Train Loss: 2.0\n")
                f.write("Validation Loss: 1.9\n")
                f.write("Learning Rate: 5e-5, Step: 10\n")
                f.write("Learning Rate: 4e-5, Step: 20\n")
            
            # Create test evaluation file
            self.test_eval_file = os.path.join(self.temp_dir, "test_eval.json")
            with open(self.test_eval_file, "w", encoding="utf-8") as f:
                f.write(json.dumps([
                    {
                        "prompt": "ሰላም",
                        "generated_text": "ሰላም ኢትዮጵያ",
                        "reference_text": "ሰላም ኢትዮጵያ",
                        "perplexity": 2.0,
                        "grammar_score": 0.8,
                        "coherence_score": 0.7,
                        "repetition_score": 0.9,
                        "cultural_relevance_score": 0.8,
                        "overall_score": 0.8,
                    },
                    {
                        "prompt": "አዲስ",
                        "generated_text": "አዲስ አበባ",
                        "reference_text": "አዲስ አበባ",
                        "perplexity": 1.8,
                        "grammar_score": 0.9,
                        "coherence_score": 0.8,
                        "repetition_score": 0.9,
                        "cultural_relevance_score": 0.9,
                        "overall_score": 0.9,
                    },
                ]))
            
            # Create test comparison file
            self.test_comparison_file = os.path.join(self.temp_dir, "test_comparison.json")
            with open(self.test_comparison_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "original": {
                        "grammar_score": 0.7,
                        "coherence_score": 0.6,
                        "repetition_score": 0.8,
                        "cultural_relevance_score": 0.7,
                        "overall_score": 0.7,
                        "model_size": 100000,
                        "inference_time": 50.0,
                    },
                    "improved": {
                        "grammar_score": 0.8,
                        "coherence_score": 0.7,
                        "repetition_score": 0.9,
                        "cultural_relevance_score": 0.8,
                        "overall_score": 0.8,
                        "model_size": 120000,
                        "inference_time": 45.0,
                    },
                }))
            
            # Create visualizer
            self.visualizer = ResultVisualizer(
                output_dir=self.temp_dir,
                dpi=100,
                fig_format="png",
            )
        except ImportError:
            self.skipTest("ResultVisualizer not available")
    
    def tearDown(self):
        """Tear down test case."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_visualizer_plot_training_loss(self):
        """Test visualizer plot training loss."""
        # Plot training loss
        fig = self.visualizer.plot_training_loss(self.test_log_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "training_loss.png")))
    
    def test_visualizer_plot_learning_rate(self):
        """Test visualizer plot learning rate."""
        # Plot learning rate
        fig = self.visualizer.plot_learning_rate(self.test_log_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "learning_rate.png")))
    
    def test_visualizer_plot_evaluation_metrics(self):
        """Test visualizer plot evaluation metrics."""
        # Plot evaluation metrics
        fig = self.visualizer.plot_evaluation_metrics(self.test_eval_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "evaluation_metrics.png")))
    
    def test_visualizer_plot_model_comparison(self):
        """Test visualizer plot model comparison."""
        # Plot model comparison
        fig = self.visualizer.plot_model_comparison(self.test_comparison_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "model_comparison.png")))
    
    def test_visualizer_plot_inference_time(self):
        """Test visualizer plot inference time."""
        # Plot inference time
        fig = self.visualizer.plot_inference_time(self.test_comparison_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "inference_time.png")))
    
    def test_visualizer_plot_model_size(self):
        """Test visualizer plot model size."""
        # Plot model size
        fig = self.visualizer.plot_model_size(self.test_comparison_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "model_size.png")))
    
    def test_visualizer_plot_perplexity(self):
        """Test visualizer plot perplexity."""
        # Plot perplexity
        fig = self.visualizer.plot_perplexity(self.test_eval_file)
        
        # Check if figure is created
        self.assertIsNotNone(fig)
        
        # Check if figure is saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "perplexity.png")))


def main():
    """Main function."""
    # Run tests
    unittest.main()


if __name__ == "__main__":
    main()