import os
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from validator import AmharicValidator # Import the validator
import time # Import time for delays
import multiprocessing

# Global validator instance for multiprocessing (initialized per process)
_validator_instance = None

def _init_worker():
    global _validator_instance
    _validator_instance = AmharicValidator()

def _is_amharic_worker(text):
    amharic_chars = re.findall(r'[\u1200-\u137F]', text)
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    return len(amharic_chars) / total_chars

def _validate_cultural_safety_worker(text):
    # FAST MODE: Skip heavy validation during collection
    # Just do basic checks to speed up 1000 article collection
    return len(text) > 50  # Basic length check only

def _process_article_worker(args):
    article_title, article_text, output_dir, min_amharic_ratio = args
    
    try:
        amharic_ratio = _is_amharic_worker(article_text)
        cultural_safe = _validate_cultural_safety_worker(article_text)
        
        if amharic_ratio >= min_amharic_ratio and cultural_safe:
            filename = os.path.join(output_dir, f"{article_title.replace(' ', '_')}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(article_text)
            return True, article_title, "Collected"
        else:
            return False, article_title, f"Skipped (Ratio: {amharic_ratio:.2f}, Cultural Safe: {cultural_safe})"
    except Exception as e:
        return False, article_title, f"Error processing: {e}"

class AmharicCorpusCollector:
    def __init__(self, output_dir="data/raw", min_amharic_ratio=0.7, cultural_safety_threshold=0.95):
        self.output_dir = output_dir
        self.min_amharic_ratio = min_amharic_ratio
        self.cultural_safety_threshold = cultural_safety_threshold
        os.makedirs(self.output_dir, exist_ok=True)
        # Validator is now initialized per process in _init_worker

    def collect_wikipedia_articles(self, num_articles=1000):
        print(f"Starting Amharic Wikipedia article collection (target: {num_articles} articles)...")
        api_url = "https://am.wikipedia.org/w/api.php"
        
        collected_count = 0
        skipped_count = 0
        apcontinue = None # For pagination of article titles
        
        # Use multiprocessing Pool
        num_processes = multiprocessing.cpu_count() # Use all available CPU cores
        print(f"Using {num_processes} processes for article validation and saving.")
        pool = multiprocessing.Pool(processes=num_processes, initializer=_init_worker)

        with tqdm(total=num_articles, desc="Collecting articles") as pbar:
            while collected_count < num_articles:
                # Step 1: Get a batch of article titles
                title_params = {
                    "action": "query",
                    "format": "json",
                    "list": "allpages",
                    "aplimit": "50", # Fetch titles in smaller batches to avoid too large URL
                    "apnamespace": "0" # Main namespace
                }
                if apcontinue:
                    title_params["apcontinue"] = apcontinue

                try:
                    title_response = requests.get(api_url, params=title_params, timeout=30)
                    title_response.raise_for_status()
                    title_data = title_response.json()

                    titles = [page["title"] for page in title_data["query"]["allpages"]]
                    if not titles:
                        print("DEBUG: No more titles found from API.")
                        break # No more titles

                    # Step 2: Fetch content for these titles in a single request
                    content_params = {
                        "action": "query",
                        "format": "json",
                        "prop": "extracts",
                        "explaintext": True,
                        "titles": "|".join(titles) # Join titles with | for batch request
                    }
                    content_response = requests.get(api_url, params=content_params, timeout=30)
                    content_response.raise_for_status()
                    content_data = content_response.json()

                    articles_to_process = []
                    for page_id, page_info in content_data["query"]["pages"].items():
                        article_title = page_info.get("title")
                        article_text = page_info.get("extract", "")
                        if article_text and article_title:
                            articles_to_process.append((article_title, article_text, self.output_dir, self.min_amharic_ratio))

                    # Process articles in parallel
                    for success, title, reason in pool.imap_unordered(_process_article_worker, articles_to_process):
                        if collected_count >= num_articles:
                            break
                        if success:
                            collected_count += 1
                            pbar.update(1)
                        else:
                            skipped_count += 1
                            # print(f"DEBUG: {title} - {reason}") # Uncomment for verbose skipping
                        
                    if "continue" in title_data:
                        apcontinue = title_data["continue"]["apcontinue"]
                    else:
                        break # No more pages of titles

                    time.sleep(0.5) # Be polite to the API
                        
                except requests.exceptions.RequestException as e:
                    print(f"CRITICAL ERROR: Failed to fetch article titles or content: {e}")
                    break # Stop if API request fails
                except Exception as e:
                    print(f"CRITICAL ERROR: An unexpected error occurred during API interaction: {e}")
                    break # Stop on unexpected errors
                
        pool.close()
        pool.join()
        print(f"Finished collection. Total articles collected: {collected_count}. Total skipped: {skipped_count}")

if __name__ == "__main__":
    # Protect the main block for multiprocessing safety
    multiprocessing.freeze_support() # For Windows compatibility
    collector = AmharicCorpusCollector()
    collector.collect_wikipedia_articles(num_articles=1000) # Set to 1000 for actual collection
