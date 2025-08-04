import torch
import os
import sys
import time
import argparse
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing modules
from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer


class ModelOptimizer:
    """Class for optimizing Amharic language models for inference."""
    
    def __init__(self, model_path, tokenizer_path, output_dir=None):
        """Initialize the optimizer.
        
        Args:
            model_path: Path to the model to optimize
            tokenizer_path: Path to the tokenizer
            output_dir: Directory to save optimized models
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        if output_dir is None:
            # Default to a subdirectory of the model directory
            self.output_dir = os.path.join(os.path.dirname(model_path), 'optimized')
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the model and tokenizer
        print(f"Loading model from {model_path}...")
        self.model = EnhancedHNet.from_pretrained(model_path)
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = EnhancedAmharicTokenizer.from_pretrained(tokenizer_path)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def quantize_dynamic(self):
        """Apply dynamic quantization to the model.
        
        Returns:
            The quantized model
        """
        print("Applying dynamic quantization...")
        
        # Quantize the model to int8
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8  # the target dtype for quantized weights
        )
        
        # Save the quantized model
        quantized_model_path = os.path.join(self.output_dir, 'model_quantized_dynamic.pt')
        torch.save(quantized_model.state_dict(), quantized_model_path)
        print(f"Saved dynamically quantized model to {quantized_model_path}")
        
        return quantized_model
    
    def quantize_static(self, calibration_data=None, calibration_size=100):
        """Apply static quantization to the model.
        
        Args:
            calibration_data: Data for calibration, or None to use default
            calibration_size: Number of samples to use for calibration
            
        Returns:
            The quantized model
        """
        print("Applying static quantization...")
        
        # Create a copy of the model for quantization
        qmodel = type(self.model)(
            vocab_size=self.model.embedding.num_embeddings,
            embed_dim=self.model.embedding.embedding_dim
        )
        qmodel.load_state_dict(self.model.state_dict())
        qmodel.eval()  # Set to evaluation mode
        
        # Set the backend based on platform
        if torch.backends.quantized.supported_engines:
            if 'qnnpack' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'qnnpack'  # Better for ARM-based processors
            elif 'fbgemm' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'fbgemm'   # Better for x86 processors
        
        # Get the appropriate quantization configuration
        backend = torch.backends.quantized.engine
        qconfig = torch.quantization.get_default_qconfig(backend)
        qmodel.qconfig = qconfig
        
        # Define quantization-ready model
        # Specify which modules to not quantize (typically embeddings and layer norms)
        quantized_modules = {nn.Linear, nn.LSTM}
        
        # Fuse modules where possible for better quantization results
        # This is a simplified approach - in a real scenario, you'd identify specific modules to fuse
        print("Preparing model for quantization...")
        
        # Prepare the model for quantization
        torch.quantization.prepare(qmodel, inplace=True)
        
        # Calibrate with sample data
        print("Calibrating with sample data...")
        if calibration_data is None:
            # Create representative data for calibration
            calibration_samples = []
            for _ in range(calibration_size):
                # Generate random sequences of different lengths
                seq_len = np.random.randint(10, 100)
                sample_input = torch.randint(0, self.model.embedding.num_embeddings, (1, seq_len))
                calibration_samples.append(sample_input)
            
            # Run calibration
            with torch.no_grad():
                for sample in tqdm(calibration_samples, desc="Calibrating"):
                    qmodel(sample)  # Run inference to collect statistics
        else:
            # Use provided calibration data
            with torch.no_grad():
                for data in tqdm(calibration_data, desc="Calibrating"):
                    qmodel(data)
        
        # Convert to quantized model
        print("Converting to quantized model...")
        torch.quantization.convert(qmodel, inplace=True)
        
        # Save the quantized model
        quantized_model_path = os.path.join(self.output_dir, 'model_quantized_static.pt')
        torch.save({
            'model_state_dict': qmodel.state_dict(),
            'model_config': {
                'vocab_size': qmodel.embedding.num_embeddings,
                'embed_dim': qmodel.embedding.embedding_dim,
                'quantized': True,
                'backend': backend
            }
        }, quantized_model_path)
        print(f"Saved statically quantized model to {quantized_model_path}")
        
        return qmodel
    
    def optimize_for_inference(self):
        """Optimize the model for inference without quantization.
        
        Returns:
            The optimized model
        """
        print("Optimizing model for inference...")
        
        # Create a copy of the model
        optimized_model = type(self.model)(
            vocab_size=self.model.embedding.num_embeddings,
            embed_dim=self.model.embedding.embedding_dim
        )
        optimized_model.load_state_dict(self.model.state_dict())
        
        # Set to evaluation mode
        optimized_model.eval()
        
        # Freeze the model weights
        for param in optimized_model.parameters():
            param.requires_grad = False
        
        # Save the optimized model
        optimized_model_path = os.path.join(self.output_dir, 'model_optimized.pt')
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'model_config': {
                'vocab_size': optimized_model.embedding.num_embeddings,
                'embed_dim': optimized_model.embedding.embedding_dim,
            }
        }, optimized_model_path)
        print(f"Saved inference-optimized model to {optimized_model_path}")
        
        return optimized_model
    
    def quantization_aware_training(self, train_loader=None, num_epochs=3, learning_rate=1e-5):
        """Perform quantization-aware training (QAT) to improve quantization accuracy.
        
        Args:
            train_loader: DataLoader with training data
            num_epochs: Number of epochs for fine-tuning
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            The QAT-trained model ready for quantization
        """
        print("Starting quantization-aware training...")
        
        if train_loader is None:
            print("Warning: No training data provided for QAT. Skipping training.")
            return self.model
        
        # Create a copy of the model for QAT
        qat_model = type(self.model)(
            vocab_size=self.model.embedding.num_embeddings,
            embed_dim=self.model.embedding.embedding_dim
        )
        qat_model.load_state_dict(self.model.state_dict())
        
        # Set the backend based on platform
        if torch.backends.quantized.supported_engines:
            if 'qnnpack' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'qnnpack'
            elif 'fbgemm' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'fbgemm'
        
        # Get the appropriate quantization configuration for QAT
        backend = torch.backends.quantized.engine
        qconfig = torch.quantization.get_default_qat_qconfig(backend)
        qat_model.qconfig = qconfig
        
        # Prepare the model for QAT
        torch.quantization.prepare_qat(qat_model, inplace=True)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        device = next(self.model.parameters()).device
        qat_model = qat_model.to(device)
        qat_model.train()
        
        print(f"Training QAT model for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs, _ = qat_model(inputs)
                # Reshape outputs for loss calculation
                outputs = outputs.view(-1, qat_model.vocab_size)
                targets = targets.view(-1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Convert the trained model to a quantized model
        qat_model.eval()
        qat_model = qat_model.cpu()
        torch.quantization.convert(qat_model, inplace=True)
        
        # Save the QAT model
        qat_model_path = os.path.join(self.output_dir, 'model_qat.pt')
        torch.save({
            'model_state_dict': qat_model.state_dict(),
            'model_config': {
                'vocab_size': qat_model.embedding.num_embeddings,
                'embed_dim': qat_model.embedding.embedding_dim,
                'quantized': True,
                'qat': True,
                'backend': backend
            }
        }, qat_model_path)
        print(f"Saved QAT model to {qat_model_path}")
        
        return qat_model
        
    def export_to_torchscript(self):
        """Export the model to TorchScript for faster inference.
        
        Returns:
            Path to the saved TorchScript model
        """
        print("Exporting model to TorchScript...")
        
        # Create a copy of the model
        model_for_export = type(self.model)(
            vocab_size=self.model.embedding.num_embeddings,
            embed_dim=self.model.embedding.embedding_dim
        )
        model_for_export.load_state_dict(self.model.state_dict())
        model_for_export.eval()
        
        # Trace the model with example input
        example_input = torch.randint(0, 1000, (1, 50))  # Batch size 1, sequence length 50
        traced_model = torch.jit.trace(model_for_export, example_input)
        
        # Save the traced model
        traced_model_path = os.path.join(self.output_dir, 'model_torchscript.pt')
        traced_model.save(traced_model_path)
        print(f"Saved TorchScript model to {traced_model_path}")
        
        return traced_model_path
    
    def benchmark_models(self, input_text="ኢትዮጵያ", sequence_length=50, num_runs=10, generate_text=False):
        """Benchmark different model optimizations.
        
        Args:
            input_text: Text to use for benchmarking
            sequence_length: Length of sequence to generate
            num_runs: Number of runs for averaging
            generate_text: Whether to benchmark text generation (slower) or just inference
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking models with input '{input_text}' for {num_runs} runs...")
        results = {}
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids])
        
        # Function to benchmark a model
        def benchmark_model(model, model_name, device='cpu'):
            model = model.to(device)
            model.eval()
            
            # Move input to device
            tensor_on_device = input_tensor.to(device)
            
            with torch.no_grad():
                # Warmup
                print(f"Warming up {model_name}...")
                for _ in range(3):
                    if generate_text and hasattr(model, 'generate'):
                        _ = model.generate(self.tokenizer, prompt=input_text, max_length=sequence_length, device=device)
                    else:
                        _ = model(tensor_on_device)
                
                # Benchmark
                print(f"Benchmarking {model_name}...")
                start_time = time.time()
                for i in range(num_runs):
                    if i % 5 == 0:
                        print(f"  Run {i+1}/{num_runs}")
                    if generate_text and hasattr(model, 'generate'):
                        _ = model.generate(self.tokenizer, prompt=input_text, max_length=sequence_length, device=device)
                    else:
                        _ = model(tensor_on_device)
                        
                run_time = (time.time() - start_time) / num_runs
            
            # Move model back to CPU to free GPU memory
            model = model.to('cpu')
            
            return run_time
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Benchmark original model
        print("\nBenchmarking original model...")
        orig_time = benchmark_model(self.model, "Original Model", device)
        
        results['original'] = {
            'avg_time': orig_time,
            'tokens_per_second': 1.0 / orig_time if orig_time > 0 else 0,
            'speedup': 1.0
        }
        print(f"Original model: {orig_time:.6f} seconds per inference")
        
        # Try to load and benchmark dynamically quantized model
        dyn_quant_path = os.path.join(self.output_dir, 'model_quantized_dynamic.pt')
        if os.path.exists(dyn_quant_path):
            try:
                print("\nBenchmarking dynamically quantized model...")
                # Load the quantized model
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                quantized_model.load_state_dict(torch.load(dyn_quant_path))
                
                quant_time = benchmark_model(quantized_model, "Dynamic Quantized Model", 'cpu')  # Quantized models run on CPU
                
                results['dynamic_quantized'] = {
                    'avg_time': quant_time,
                    'tokens_per_second': 1.0 / quant_time if quant_time > 0 else 0,
                    'speedup': orig_time / quant_time if quant_time > 0 else 0
                }
                print(f"Dynamic quantized model: {quant_time:.6f} seconds per inference")
                print(f"Speedup: {orig_time / quant_time:.2f}x")
            except Exception as e:
                print(f"Error benchmarking dynamic quantized model: {e}")
        
        # Try to load and benchmark static quantized model
        static_quant_path = os.path.join(self.output_dir, 'model_quantized_static.pt')
        if os.path.exists(static_quant_path):
            try:
                print("\nBenchmarking statically quantized model...")
                # Load the static quantized model
                static_model = type(self.model)(
                    vocab_size=self.model.embedding.num_embeddings,
                    embed_dim=self.model.embedding.embedding_dim
                )
                
                # Load the state dict
                checkpoint = torch.load(static_quant_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    static_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    static_model.load_state_dict(checkpoint)
                
                static_time = benchmark_model(static_model, "Static Quantized Model", 'cpu')  # Quantized models run on CPU
                
                results['static_quantized'] = {
                    'avg_time': static_time,
                    'tokens_per_second': 1.0 / static_time if static_time > 0 else 0,
                    'speedup': orig_time / static_time if static_time > 0 else 0
                }
                print(f"Static quantized model: {static_time:.6f} seconds per inference")
                print(f"Speedup: {orig_time / static_time:.2f}x")
            except Exception as e:
                print(f"Error benchmarking static quantized model: {e}")
        
        # Try to load and benchmark QAT model
        qat_path = os.path.join(self.output_dir, 'model_qat.pt')
        if os.path.exists(qat_path):
            try:
                print("\nBenchmarking QAT model...")
                # Load the QAT model
                qat_model = type(self.model)(
                    vocab_size=self.model.embedding.num_embeddings,
                    embed_dim=self.model.embedding.embedding_dim
                )
                
                # Load the state dict
                checkpoint = torch.load(qat_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    qat_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    qat_model.load_state_dict(checkpoint)
                
                qat_time = benchmark_model(qat_model, "QAT Model", 'cpu')  # Quantized models run on CPU
                
                results['qat'] = {
                    'avg_time': qat_time,
                    'tokens_per_second': 1.0 / qat_time if qat_time > 0 else 0,
                    'speedup': orig_time / qat_time if qat_time > 0 else 0
                }
                print(f"QAT model: {qat_time:.6f} seconds per inference")
                print(f"Speedup: {orig_time / qat_time:.2f}x")
            except Exception as e:
                print(f"Error benchmarking QAT model: {e}")
        
        # Try to load and benchmark TorchScript model
        script_path = os.path.join(self.output_dir, 'model_torchscript.pt')
        if os.path.exists(script_path):
            try:
                print("\nBenchmarking TorchScript model...")
                # Load the TorchScript model
                scripted_model = torch.jit.load(script_path)
                
                script_time = benchmark_model(scripted_model, "TorchScript Model", device)
                
                results['torchscript'] = {
                    'avg_time': script_time,
                    'tokens_per_second': 1.0 / script_time if script_time > 0 else 0,
                    'speedup': orig_time / script_time if script_time > 0 else 0
                }
                print(f"TorchScript model: {script_time:.6f} seconds per inference")
                print(f"Speedup: {orig_time / script_time:.2f}x")
            except Exception as e:
                print(f"Error benchmarking TorchScript model: {e}")
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"{'Model Type':<25} {'Avg Time (s)':<15} {'Tokens/s':<15} {'Speedup':<10} {'Memory (MB)':<15}")
        print("-" * 80)
        
        # Get memory usage for original model
        original_memory = 0
        try:
            original_memory = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        except Exception:
            pass
        
        for model_type, metrics in results.items():
            speedup = metrics.get('speedup', 1.0)
            memory = metrics.get('memory_mb', original_memory if model_type == 'original' else 'N/A')
            print(f"{model_type:<25} {metrics['avg_time']:<15.6f} "
                  f"{metrics['tokens_per_second']:<15.2f} "
                  f"{speedup:<10.2f}x "
                  f"{memory}")
        
        # Plot results if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            model_names = list(results.keys())
            inference_times = [results[model]['avg_time'] for model in model_names]
            speedups = [results[model]['speedup'] for model in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Inference time plot
            ax1.bar(model_names, inference_times)
            ax1.set_title('Inference Time (lower is better)')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_ylim(bottom=0)
            
            # Speedup plot
            ax2.bar(model_names, speedups)
            ax2.set_title('Speedup vs Original (higher is better)')
            ax2.set_ylabel('Speedup (x)')
            ax2.set_ylim(bottom=0)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'benchmark_results.png')
            plt.savefig(plot_path)
            print(f"\nBenchmark plot saved to {plot_path}")
            plt.close()
        except ImportError:
            print("\nMatplotlib not available, skipping benchmark plot generation")
        except Exception as e:
            print(f"\nError generating benchmark plot: {e}")
        
        return results


def main():
    """Main function to optimize models."""
    parser = argparse.ArgumentParser(description='Optimize Amharic language models')
    
    # Model and tokenizer paths
    parser.add_argument('--model', type=str, default='models/enhanced_hnet/model.pt',
                        help='Path to the model')
    parser.add_argument('--tokenizer', type=str, default='models/enhanced_hnet/tokenizer.json',
                        help='Path to the tokenizer')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save optimized models')
    
    # Optimization methods
    optimization_group = parser.add_argument_group('Optimization Methods')
    optimization_group.add_argument('--dynamic-quant', action='store_true',
                        help='Apply dynamic quantization')
    optimization_group.add_argument('--static-quant', action='store_true',
                        help='Apply static quantization')
    optimization_group.add_argument('--qat', action='store_true',
                        help='Apply quantization-aware training (requires training data)')
    optimization_group.add_argument('--optimize-inference', action='store_true',
                        help='Optimize for inference')
    optimization_group.add_argument('--export-torchscript', action='store_true',
                        help='Export to TorchScript')
    optimization_group.add_argument('--all', action='store_true',
                        help='Apply all optimization methods (except QAT)')
    
    # QAT specific options
    qat_group = parser.add_argument_group('QAT Options')
    qat_group.add_argument('--qat-epochs', type=int, default=3,
                        help='Number of epochs for QAT')
    qat_group.add_argument('--qat-lr', type=float, default=1e-5,
                        help='Learning rate for QAT')
    qat_group.add_argument('--train-data', type=str, default=None,
                        help='Path to training data for QAT (required for QAT)')
    
    # Static quantization options
    static_group = parser.add_argument_group('Static Quantization Options')
    static_group.add_argument('--calibration-size', type=int, default=100,
                        help='Number of samples for calibration')
    
    # Benchmarking options
    benchmark_group = parser.add_argument_group('Benchmarking Options')
    benchmark_group.add_argument('--benchmark', action='store_true',
                        help='Benchmark optimized models')
    benchmark_group.add_argument('--benchmark-text', type=str, default="ኢትዮጵያ",
                        help='Text to use for benchmarking')
    benchmark_group.add_argument('--benchmark-runs', type=int, default=10,
                        help='Number of runs for benchmarking')
    benchmark_group.add_argument('--benchmark-generation', action='store_true',
                        help='Benchmark text generation (slower) instead of just inference')
    
    args = parser.parse_args()
    
    # Ensure model path is absolute
    if not os.path.isabs(args.model):
        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    
    # Ensure tokenizer path is absolute
    if not os.path.isabs(args.tokenizer):
        args.tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tokenizer)
    
    # Create optimizer
    optimizer = ModelOptimizer(args.model, args.tokenizer, args.output_dir)
    
    # Check if any optimization method was selected
    optimization_selected = args.dynamic_quant or args.static_quant or args.qat or \
                           args.optimize_inference or args.export_torchscript or args.all
    
    # Apply optimizations
    if args.all:
        print("Applying all optimization methods (except QAT)...")
        optimizer.optimize_for_inference()
        optimizer.quantize_dynamic()
        optimizer.quantize_static(calibration_size=args.calibration_size)
        optimizer.export_to_torchscript()
    else:
        if args.dynamic_quant:
            optimizer.quantize_dynamic()
        
        if args.static_quant:
            optimizer.quantize_static(calibration_size=args.calibration_size)
        
        if args.optimize_inference:
            optimizer.optimize_for_inference()
        
        if args.export_torchscript:
            optimizer.export_to_torchscript()
        
        if args.qat:
            if args.train_data is None:
                print("Error: --train-data is required for QAT")
                return
            
            # Load training data
            print(f"Loading training data from {args.train_data}...")
            try:
                import torch
                from torch.utils.data import DataLoader, TensorDataset
                import os
                import json
                
                # Try to import the AmharicHNetDataset class if available
                try:
                    # First try to import from the current project
                    from enhanced_train import AmharicHNetDataset, EnhancedAmharicTokenizer
                    
                    # Load the tokenizer from the model path
                    tokenizer = EnhancedAmharicTokenizer.from_file(args.tokenizer)
                    
                    # Create dataset using the AmharicHNetDataset class
                    dataset = AmharicHNetDataset(
                        data_file=args.train_data,
                        tokenizer=tokenizer,
                        seq_length=128,  # Reasonable default, can be made configurable
                        stride=64        # Reasonable default, can be made configurable
                    )
                    
                    # Create DataLoader
                    train_loader = DataLoader(
                        dataset,
                        batch_size=16,  # Can be made configurable
                        shuffle=True
                    )
                    
                    print(f"Successfully loaded training data with {len(dataset)} samples")
                    
                except ImportError:
                    # If AmharicHNetDataset is not available, use a simple text loader
                    print("AmharicHNetDataset not found, using simple text loader...")
                    
                    # Load the tokenizer
                    with open(args.tokenizer, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    
                    # Simple function to load text data
                    def load_text_data(file_path, max_samples=1000):
                        texts = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if i >= max_samples:
                                    break
                                line = line.strip()
                                if line:  # Skip empty lines
                                    texts.append(line)
                        return texts
                    
                    # Simple tokenize function
                    def tokenize_text(text, vocab, max_length=128):
                        # Simple character-level tokenization
                        tokens = [vocab.get(char, 0) for char in text[:max_length]]
                        # Pad if necessary
                        if len(tokens) < max_length:
                            tokens = tokens + [0] * (max_length - len(tokens))
                        return tokens
                    
                    # Load and tokenize data
                    texts = load_text_data(args.train_data)
                    vocab = {token: idx for idx, token in enumerate(tokenizer_data.get('vocab', []))} 
                    
                    # Create tensors
                    input_ids = torch.tensor([tokenize_text(text, vocab) for text in texts], dtype=torch.long)
                    # For language modeling, targets are the same as inputs but shifted
                    target_ids = torch.tensor([tokenize_text(text[1:] + ' ', vocab) for text in texts], dtype=torch.long)
                    
                    # Create dataset and dataloader
                    dataset = TensorDataset(input_ids, target_ids)
                    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
                    
                    print(f"Successfully loaded {len(texts)} text samples for training")
                
                # Run QAT
                optimizer.quantization_aware_training(
                    train_loader=train_loader,
                    num_epochs=args.qat_epochs,
                    learning_rate=args.qat_lr
                )
                
            except Exception as e:
                print(f"Error loading training data: {e}")
                print("Skipping QAT.")
    
    # If no specific optimization was requested, show help
    if not optimization_selected and not args.benchmark:
        parser.print_help()
        print("\nNo optimization method selected. Please specify at least one optimization method.")
        return
    
    # Benchmark if requested
    if args.benchmark:
        optimizer.benchmark_models(
            input_text=args.benchmark_text,
            num_runs=args.benchmark_runs,
            generate_text=args.benchmark_generation
        )


if __name__ == "__main__":
    main()