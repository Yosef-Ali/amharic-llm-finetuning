#!/bin/bash

# Run the entire Amharic H-Net pipeline from training to evaluation

# Default parameters
DATA_DIR="../collected_articles"
OUTPUT_DIR="./output"
MODEL_NAME="amharic_hnet_improved"
BATCH_SIZE=16
NUM_EPOCHS=10
LEARNING_RATE=5e-5
MAX_LENGTH=512
USE_MIXED_PRECISION=true
GRADIENT_ACCUMULATION_STEPS=4
TEST_PROMPTS="./test_prompts.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --no_mixed_precision)
      USE_MIXED_PRECISION=false
      shift
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --test_prompts)
      TEST_PROMPTS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_dir DIR                  Data directory (default: ../collected_articles)"
      echo "  --output_dir DIR                Output directory (default: ./output)"
      echo "  --model_name NAME               Model name (default: amharic_hnet_improved)"
      echo "  --batch_size SIZE               Batch size (default: 16)"
      echo "  --num_epochs EPOCHS             Number of epochs (default: 10)"
      echo "  --learning_rate RATE            Learning rate (default: 5e-5)"
      echo "  --max_length LENGTH             Maximum sequence length (default: 512)"
      echo "  --no_mixed_precision           Disable mixed precision training"
      echo "  --gradient_accumulation_steps N Number of gradient accumulation steps (default: 4)"
      echo "  --test_prompts FILE             Test prompts file (default: ./test_prompts.txt)"
      echo "  --help                          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
do

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create test prompts file if it doesn't exist
if [ ! -f "$TEST_PROMPTS" ]; then
  echo "Creating default test prompts file..."
  cat > "$TEST_PROMPTS" << EOF
ኢትዮጵያ
አዲስ አበባ
የአማርኛ ቋንቋ
ባህላዊ ምግብ
የኢትዮጵያ ታሪክ
የኢትዮጵያ ባህል
የኢትዮጵያ ሙዚቃ
የኢትዮጵያ ስፖርት
የኢትዮጵያ ፖለቲካ
የኢትዮጵያ ኢኮኖሚ
EOF
fi

# Set mixed precision flag
if [ "$USE_MIXED_PRECISION" = true ]; then
  MIXED_PRECISION_FLAG="--use_mixed_precision"
else
  MIXED_PRECISION_FLAG=""
fi

# Step 1: Train the tokenizer and model
echo "=== Step 1: Training tokenizer and model ==="
python improved_training.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR/$MODEL_NAME" \
  --train_tokenizer \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  $MIXED_PRECISION_FLAG \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"

# Check if training was successful
if [ $? -ne 0 ]; then
  echo "Error: Training failed"
  exit 1
fi

# Step 2: Optimize the model
echo "\n=== Step 2: Optimizing model ==="
python model_optimization.py \
  --model_path "$OUTPUT_DIR/$MODEL_NAME" \
  --tokenizer_path "$OUTPUT_DIR/$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR/${MODEL_NAME}_optimized" \
  --quantization_type dynamic \
  --export_format torchscript

# Step 3: Evaluate the model
echo "\n=== Step 3: Evaluating model ==="
python evaluate_model.py \
  --model_path "$OUTPUT_DIR/$MODEL_NAME" \
  --tokenizer_path "$OUTPUT_DIR/$MODEL_NAME" \
  --test_data_dir "$DATA_DIR" \
  --test_prompts_path "$TEST_PROMPTS" \
  --output_dir "$OUTPUT_DIR/${MODEL_NAME}_evaluation"

# Step 4: Generate sample texts
echo "\n=== Step 4: Generating sample texts ==="
mkdir -p "$OUTPUT_DIR/${MODEL_NAME}_samples"

while IFS= read -r prompt; do
  echo "Generating text for prompt: $prompt"
  python improved_generation.py \
    --model_path "$OUTPUT_DIR/$MODEL_NAME" \
    --tokenizer_path "$OUTPUT_DIR/$MODEL_NAME" \
    --prompt "$prompt" \
    --max_length 200 \
    --temperature 0.7 \
    --top_p 0.95 \
    --repetition_penalty 1.5 \
    --use_template \
    --output_file "$OUTPUT_DIR/${MODEL_NAME}_samples/$(echo $prompt | tr ' ' '_').txt"
done < "$TEST_PROMPTS"

echo "\n=== Pipeline completed successfully ==="
echo "Model and tokenizer saved to: $OUTPUT_DIR/$MODEL_NAME"
echo "Optimized model saved to: $OUTPUT_DIR/${MODEL_NAME}_optimized"
echo "Evaluation results saved to: $OUTPUT_DIR/${MODEL_NAME}_evaluation"
echo "Sample texts saved to: $OUTPUT_DIR/${MODEL_NAME}_samples"