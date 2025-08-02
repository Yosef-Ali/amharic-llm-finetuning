# Amharic H-Net Project Summary & Debugging Notes

This document summarizes the key issues encountered and resolutions applied during the development and debugging of the Amharic H-Net project, as well as the current state of each component.

## Project Goal
To build a 300M parameter Amharic H-Net model, optimized for M2 8GB hardware, leveraging transfer learning, and ensuring robust cultural safety and linguistic accuracy.

---

## Component-wise Changes & Current Status:

### 1. `config.py`
- **Original Issue:** Hardcoded `vocab_size=5000` which caused a mismatch with the actual tokenizer vocabulary. `d_model`, `n_layers`, `n_heads` were too small for the target model size. `validate_every` was too high for small datasets.
- **Changes Made:**
    - Removed `vocab_size` (now dynamically determined by tokenizer).
    - Increased `d_model` to `1024`.
    - Increased `n_layers` to `24`.
    - Increased `n_heads` to `16`.
    - Adjusted `batch_size` to `4`.
    - Added `gradient_accumulation_steps = 8`.
    - Added `use_fp16 = True` for mixed precision.
    - Set `validate_every = 1` to ensure model saving during early training.
- **Current Status:** Configured for a larger model architecture and memory-efficient training.

### 2. `dataset.py`
- **Original Issue:** None directly, but its output was double-shifted by `hnet.py`.
- **Changes Made:** None.
- **Current Status:** Correctly prepares input/target pairs for next-token prediction.

### 3. `hybrid_tokenizer.py` (formerly `tokenizer.py`)
- **Original Issue:** Limited, hardcoded vocabulary led to `[UNK]` tokens in generated output. `self.punctuation` was accessed before definition in `__init__`.
- **Changes Made:**
    - Reordered `__init__` to define `self.punctuation` before calling `_build_vocab_from_corpus`.
    - Modified `_build_vocab_from_corpus` to:
        - Include a much larger base list of common Amharic words and numbers.
        - Dynamically build the rest of the vocabulary from `amharic_corpus.txt`.
    - Added a print statement to show the final vocabulary size.
    - Renamed file from `tokenizer.py` to `hybrid_tokenizer.py` to better reflect its functionality.
- **Current Status:** Dynamically builds a more comprehensive vocabulary from the corpus and a base list, reducing `[UNK]` occurrences.

### 4. `hnet.py`
- **Original Issue:** "Double-shifting" of logits and labels in the `forward` method, causing incorrect loss calculation and preventing learning.
- **Changes Made:**
    - Removed the redundant `shift_logits` and `shift_labels` logic. Loss is now calculated directly on `logits` and `labels`.
- **Current Status:** Correctly calculates loss and implements the H-Net architecture.

### 5. `validator.py`
- **Original Issue:** `KeyError: 'taboo_associations'` when cultural rules lacked this key. `IndexError: string index out of range` in `validate_spacing` when punctuation was at the end of a string.
- **Changes Made:**
    - Added checks for the existence of `"taboo_associations"` key before accessing it in `validate_cultural_safety` and `apply_guardrails`.
    - Added a boundary check (`text.index(punct) + 1 < len(text)`) in `validate_spacing` to prevent `IndexError`.
- **Current Status:** More robust cultural validation and spacing checks.

### 6. `train.py`
- **Original Issue:** Model instantiation used `AmharicConfig.vocab_size` (which was incorrect). Did not utilize `gradient_accumulation_steps` or `use_fp16`.
- **Changes Made:**
    - Modified `HNetAmharic` instantiation to use `vocab_size=len(tokenizer.vocab)`.
    - Integrated `GradScaler` and `autocast` for mixed precision (`use_fp16`).
    - Implemented gradient accumulation logic using `AmharicConfig.gradient_accumulation_steps`.
- **Current Status:** Configured for large-scale training with memory optimizations.

### 7. `generate.py`
- **Original Issue:** Missing `import os`. Model instantiation used `AmharicConfig.vocab_size` (which was incorrect). Failed to reliably locate the `amharic_hnet_best.pt` file.
- **Changes Made:**
    - Added `import os`.
    - Modified `HNetAmharic` instantiation to use `vocab_size=len(tokenizer.vocab)`.
    - Updated `load_model` to construct an absolute path to the model directory for robust loading.
- **Current Status:** Correctly loads and uses the trained model for generation.

### 8. `corpus_collector.py`
- **Original Issue:** Very slow data collection due to inefficient fetching (one article at a time) and synchronous, computationally expensive validation checks. Aborted prematurely.
- **Changes Made:**
    - Switched to batch fetching of article titles and content using Wikipedia API.
    - Implemented `multiprocessing` to parallelize the `_is_amharic`, `_validate_cultural_safety`, and saving steps.
    - Added more granular logging and error handling.
    - Re-enabled full `_is_amharic` and `_validate_cultural_safety` checks.
    - Set `num_articles` target to 1000 for actual collection.
- **Current Status:** Designed for efficient and parallel data collection.

### 9. `linguistic_analyzer.py`
- **Original Issue:** None directly, but it's a placeholder.
- **Changes Made:** None.
- **Current Status:** Basic structure for morphological analysis and cultural term protection. Requires further development for full functionality.

### 10. `requirements.txt`
- **Original Issue:** Incompatible `torch` and `numpy` versions for the environment. Missing `beautifulsoup4` and `requests`.
- **Changes Made:**
    - Updated `torch` to `2.6.0`.
    - Removed strict version for `numpy` to allow `pip` to choose a compatible version.
    - Installed `beautifulsoup4` and `requests`.
    - Upgraded `pip`, `setuptools`, and `wheel` in the `venv`.
- **Current Status:** All necessary dependencies are installed and compatible.

---

## Overall Project Status & Next Steps:

The project's core components are now stable and correctly configured for the ambitious goal of a 300M parameter Amharic H-Net.

**The most critical next step is to successfully complete the data collection phase.**

**Action for User:**
1.  **Run `venv/bin/python corpus_collector.py`** from the `amharic-hnet` directory. This will attempt to collect 1000 articles using parallel processing.
2.  **Monitor the output** for any errors or signs of premature termination. If it still aborts, the new logging should provide more specific clues.

Once a substantial corpus is collected in `data/raw`, the next steps would be:
- Run `linguistic_analyzer.py` to process the raw data into `data/processed`.
- Retrain the H-Net model using `train.py` on the larger, processed dataset.
- Evaluate the model's performance and generation quality.

This `note.md` file captures the journey and the current state, providing a clear handover for further development.
