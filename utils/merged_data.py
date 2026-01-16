import polars as pl
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source directory containing data files
DATA_DIR = Path(r"c:\Users\hp\Desktop\projet\nlp\projet3_Q&A\unprocessed_data")

# Output file path
OUTPUT_FILE = Path(r"c:\Users\hp\Desktop\projet\nlp\projet3_Q&A\dataset\merged_arabic_qa.csv")

# Unified schema - all data will be mapped to these columns
UNIFIED_SCHEMA = [
    "id",           # Unique identifier
    "question",     # The question text
    "answer",       # The answer text
    "context",      # Context/passage text (if available)
    "title",        # Document title (if available)
    "category",     # Category/label (if available)
    "source_file"   # Original source file name
]

# Column name mappings for different schemas
COLUMN_MAPPINGS = {
    # JSONL format (arabic-train.jsonl)
    "question_text": "question",
    "passage_text": "context",
    "document_title": "title",
    
    

    
    # test.csv format
    "label": "category",
    
    # General mappings
    "answers": "answer",
    "text": "answer",
}


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def flatten_nested_json(data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten a nested dictionary structure."""
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_json(v, new_key, sep).items())
        elif isinstance(v, list):
            # Handle list of answers - extract first answer text
            if k == 'answers' and len(v) > 0:
                if isinstance(v[0], dict) and 'text' in v[0]:
                    items.append((new_key, v[0]['text']))
                elif isinstance(v[0], str):
                    items.append((new_key, v[0]))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


import re

def extract_answer_from_complex(answer_value: Any) -> str:
    """Extract answer text from various formats."""
    if answer_value is None:
        return ""
    
    s_val = str(answer_value)
    
    # Strategy 1: Regex for specifically dirty numpy/python string representations
    # Matches patterns like: "{'text': array(['Answer Text'], dtype=object)}"
    # or "{'text': ['Answer Text']}"
    # regex looks for: 'text': (optional array encapsulation) ['MATCH_THIS']
    match = re.search(r"'text':\s*(?:array\()?\s*\[\s*['\"](.*?)['\"]", s_val)
    if match:
        return match.group(1)
        
    if isinstance(answer_value, str):
        # Try to parse as JSON if it looks like JSON
        if answer_value.startswith('[') or answer_value.startswith('{'):
            try:
                parsed = json.loads(answer_value.replace("'", '"'))
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict) and 'text' in parsed[0]:
                        return str(parsed[0]['text'])
                    return str(parsed[0])
                elif isinstance(parsed, dict) and 'text' in parsed:
                    return str(parsed['text'])
                return str(parsed)
            except:
                return answer_value
        return answer_value
    
    if isinstance(answer_value, list):
        if len(answer_value) > 0:
            if isinstance(answer_value[0], dict) and 'text' in answer_value[0]:
                return str(answer_value[0]['text'])
            return str(answer_value[0])
        return ""
    
    if isinstance(answer_value, dict):
        if 'text' in answer_value:
            return str(answer_value['text'])
        return str(answer_value)
    
    return str(answer_value)


def normalize_dataframe(df: pl.DataFrame, source_file: str) -> pl.DataFrame:
    """
    Normalize a dataframe to the unified schema.
    Maps columns to standard names and adds missing columns.
    """
    # Get current columns
    current_cols = df.columns
    
    # Rename columns based on mappings
    rename_dict = {}
    for old_name in current_cols:
        if old_name in COLUMN_MAPPINGS:
            rename_dict[old_name] = COLUMN_MAPPINGS[old_name]
    
    if rename_dict:
        df = df.rename(rename_dict)
    
    # Handle special case for 'answers' column that needs extraction
    if 'answer' in df.columns:
        df = df.with_columns(
            pl.col('answer').map_elements(
                extract_answer_from_complex, 
                return_dtype=pl.Utf8
            ).alias('answer')
        )
    elif 'answers' in df.columns:
        df = df.with_columns(
            pl.col('answers').map_elements(
                extract_answer_from_complex,
                return_dtype=pl.Utf8
            ).alias('answer')
        )
        df = df.drop('answers')
    
    # Add source file column
    df = df.with_columns(pl.lit(source_file).alias('source_file'))
    
    # Add missing columns with null values
    for col in UNIFIED_SCHEMA:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    
    # Select only unified schema columns (in order)
    df = df.select(UNIFIED_SCHEMA)
    
    # Cast all columns to string for consistency
    df = df.cast({col: pl.Utf8 for col in UNIFIED_SCHEMA})
    
    return df


def load_jsonl_file(file_path: Path) -> Optional[pl.DataFrame]:
    """Load a JSONL file and return a Polars DataFrame."""
    logger.info(f"Loading JSONL file: {file_path.name}")
    
    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    flat_record = flatten_nested_json(record)
                    records.append(flat_record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {file_path.name}: {e}")
                    continue
        
        if not records:
            logger.warning(f"No valid records found in {file_path.name}")
            return None
        
        df = pl.DataFrame(records)
        logger.info(f"Loaded {len(df)} records from {file_path.name}")
        return normalize_dataframe(df, file_path.name)
    
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return None


def load_json_file(file_path: Path) -> Optional[pl.DataFrame]:
    """Load a JSON file and return a Polars DataFrame."""
    logger.info(f"Loading JSON file: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        records = []
        
        if isinstance(data, list):
            # Array of objects
            for item in data:
                if isinstance(item, dict):
                    flat_record = flatten_nested_json(item)
                    records.append(flat_record)
        elif isinstance(data, dict):
            # Check for nested data structures (like SQuAD format)
            if 'data' in data:
                for article in data.get('data', []):
                    title = article.get('title', '')
                    for paragraph in article.get('paragraphs', []):
                        context = paragraph.get('context', '')
                        for qa in paragraph.get('qas', []):
                            record = {
                                'id': qa.get('id', ''),
                                'question': qa.get('question', ''),
                                'context': context,
                                'title': title,
                            }
                            # Extract answer
                            answers = qa.get('answers', [])
                            if answers:
                                record['answer'] = answers[0].get('text', '')
                            else:
                                record['answer'] = ''
                            records.append(record)
            else:
                # Single object - flatten it
                flat_record = flatten_nested_json(data)
                records.append(flat_record)
        
        if not records:
            logger.warning(f"No valid records found in {file_path.name}")
            return None
        
        df = pl.DataFrame(records)
        logger.info(f"Loaded {len(df)} records from {file_path.name}")
        return normalize_dataframe(df, file_path.name)
    
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return None


def load_csv_file(file_path: Path) -> Optional[pl.DataFrame]:
    """Load a CSV file and return a Polars DataFrame."""
    logger.info(f"Loading CSV file: {file_path.name}")
    
    try:
        # Try different parsing options for robustness
        try:
            df = pl.read_csv(
                file_path,
                encoding='utf-8',
                infer_schema_length=10000,
                ignore_errors=True,
                truncate_ragged_lines=True
            )
        except Exception as e1:
            logger.warning(f"Standard parsing failed for {file_path.name}, trying with different options: {e1}")
            try:
                df = pl.read_csv(
                    file_path,
                    encoding='utf-8',
                    infer_schema_length=None,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    quote_char='"',
                    eol_char='\n'
                )
            except Exception as e2:
                logger.error(f"Failed to load {file_path.name}: {e2}")
                return None
        
        logger.info(f"Loaded {len(df)} records from {file_path.name}")
        return normalize_dataframe(df, file_path.name)
    
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return None


def load_data_file(file_path: Path) -> Optional[pl.DataFrame]:
    """Load a data file based on its extension."""
    extension = file_path.suffix.lower()
    
    if extension == '.jsonl':
        return load_jsonl_file(file_path)
    elif extension == '.json':
        return load_json_file(file_path)
    elif extension == '.csv':
        return load_csv_file(file_path)
    else:
        logger.warning(f"Unsupported file format: {extension}")
        return None


def merge_all_data(data_dir: Path) -> pl.DataFrame:
    """
    Merge all data files from the specified directory into a single DataFrame.
    """
    all_dataframes: List[pl.DataFrame] = []
    
    # Supported extensions
    extensions = ['*.json', '*.jsonl', '*.csv']
    
    for ext in extensions:
        for file_path in data_dir.glob(ext):
            # Skip any output files we might have created or excluded files
            if 'master_data' in file_path.name or 'ahqad.csv' in file_path.name.lower():
                continue
            
            df = load_data_file(file_path)
            if df is not None and len(df) > 0:
                all_dataframes.append(df)
                logger.info(f"Successfully processed: {file_path.name} ({len(df)} rows)")
    
    if not all_dataframes:
        raise ValueError("No data files were successfully loaded!")
    
    # Concatenate all dataframes
    logger.info(f"Merging {len(all_dataframes)} dataframes...")
    merged_df = pl.concat(all_dataframes, how='vertical_relaxed')
    
    return merged_df


def clean_and_deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the merged dataframe by removing duplicates and empty rows.
    """
    initial_rows = len(df)
    
    # Remove rows where both question and answer are null or empty
    df = df.filter(
        (pl.col('question').is_not_null() & (pl.col('question') != '')) |
        (pl.col('answer').is_not_null() & (pl.col('answer') != ''))
    )
    
    after_null_filter = len(df)
    logger.info(f"Removed {initial_rows - after_null_filter} empty rows")
    
    # Remove exact duplicates based on question and answer
    df = df.unique(subset=['question', 'answer'], maintain_order=True)
    
    after_dedup = len(df)
    logger.info(f"Removed {after_null_filter - after_dedup} duplicate rows")
    
    # Add a unique row ID
    df = df.with_row_index('row_id').cast({'row_id': pl.Utf8})
    
    # Reorder columns with row_id first
    final_columns = ['row_id'] + UNIFIED_SCHEMA
    df = df.select(final_columns)
    
    return df


def main():
    """Main function to execute the data merging pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Data Merging Pipeline")
    logger.info("=" * 60)
    
    # Verify source directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    logger.info(f"Source directory: {DATA_DIR}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    
    # Step 1: Merge all data files
    logger.info("\nStep 1: Loading and merging data files...")
    merged_df = merge_all_data(DATA_DIR)
    logger.info(f"Total merged rows: {len(merged_df)}")
    
    # Step 2: Clean and deduplicate
    logger.info("\nStep 2: Cleaning and deduplicating data...")
    cleaned_df = clean_and_deduplicate(merged_df)
    logger.info(f"Final row count: {len(cleaned_df)}")
    
    # Step 3: Save to CSV
    logger.info("\nStep 3: Saving to CSV...")
    cleaned_df.write_csv(OUTPUT_FILE)
    logger.info(f"Successfully saved to: {OUTPUT_FILE}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(cleaned_df)}")
    logger.info(f"Columns: {cleaned_df.columns}")
    
    # Show distribution by source file
    source_counts = cleaned_df.group_by('source_file').agg(pl.count().alias('count'))
    logger.info("\nRecords by source file:")
    for row in source_counts.iter_rows(named=True):
        logger.info(f"  {row['source_file']}: {row['count']}")
    
    # Show sample of the data
    logger.info("\nSample of merged data (first 3 rows):")
    print(cleaned_df.head(3))
    
    logger.info("\n" + "=" * 60)
    logger.info("Data merging completed successfully!")
    logger.info("=" * 60)
    
    return cleaned_df


if __name__ == "__main__":
    df = main()
