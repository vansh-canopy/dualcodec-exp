#!/usr/bin/env python3
"""
Data loader script for Emilia Dataset.
Converts the train.ipynb notebook functionality into a standalone script with proper logging.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
import argparse


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_loader.log')
        ]
    )


def download_emilia_dataset(
    local_dir: str, 
    max_shards: int = 10,
    language: str = "EN"
) -> str:
    """
    Download Emilia dataset from HuggingFace Hub.
    
    Args:
        local_dir: Local directory to store the dataset
        max_shards: Maximum number of shards to download
        language: Language code (EN, DE, etc.)
    
    Returns:
        Path to the downloaded dataset directory
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Downloading Emilia dataset to {local_dir}")
    logger.info(f"Language: {language}, Max shards: {max_shards}")
    
    try:
        # Create pattern for the specific language and shard count
        pattern = f"Emilia/{language}/{language}-B{0:06d}*.tar"
        if max_shards > 1:
            # For multiple shards, use a more general pattern
            pattern = f"Emilia/{language}/{language}-B00000*.tar"
        
        downloaded_path = snapshot_download(
            repo_id="amphion/Emilia-Dataset",
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=pattern,
            max_workers=8,
        )
        
        logger.info(f"Dataset downloaded successfully to {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def load_dataset_shards(
    local_dir: str,
    max_shards: int = 10,
    language: str = "EN"
) -> Dataset:
    """
    Load and concatenate dataset shards.
    
    Args:
        local_dir: Directory containing the dataset
        max_shards: Maximum number of shards to load
        language: Language code
    
    Returns:
        Concatenated dataset
    """
    logger = logging.getLogger(__name__)
    
    # Generate tar file paths
    tar_paths = [f"{language}-B{i:06d}.tar" for i in range(max_shards)]
    dataset_dir = os.path.join(local_dir, "Emilia", language)
    
    logger.info(f"Loading {max_shards} shards from {dataset_dir}")
    
    ds_list = []
    
    try:
        for i, tar_filename in enumerate(tqdm(tar_paths, desc="Loading shards"), start=1):
            tar_path = os.path.join(dataset_dir, tar_filename)
            
            if not os.path.exists(tar_path):
                logger.warning(f"Shard file not found: {tar_path}")
                continue
                
            logger.info(f"Loading shard {i}/{max_shards}: {tar_filename}")
            
            ds = load_dataset(
                dataset_dir,
                data_files={language.lower(): tar_filename},
                split=language.lower()
            )
            
            ds_list.append(ds)
            logger.debug(f"Loaded shard {i} with {len(ds)} examples")
        
        if not ds_list:
            raise ValueError("No dataset shards were successfully loaded")
        
        logger.info(f"Concatenating {len(ds_list)} datasets")
        big_ds = concatenate_datasets(ds_list)
        
        logger.info(f"Final dataset size: {len(big_ds)} examples")
        return big_ds
        
    except Exception as e:
        logger.error(f"Failed to load dataset shards: {e}")
        raise


def main():
    """Main function to run the data loading process."""
    parser = argparse.ArgumentParser(description="Load Emilia Dataset")
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="/workspace/emilia_dataset",
        help="Local directory to store the dataset"
    )
    parser.add_argument(
        "--max_shards", 
        type=int, 
        default=10,
        help="Maximum number of shards to download/load"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="EN",
        help="Language code (EN, DE, etc.)"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--download_only", 
        action="store_true",
        help="Only download the dataset, don't load it"
    )
    parser.add_argument(
        "--load_only", 
        action="store_true",
        help="Only load existing dataset, don't download"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Emilia dataset processing")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(args.local_dir, exist_ok=True)
        
        # Download dataset (unless load_only is specified)
        if not args.load_only:
            download_emilia_dataset(
                local_dir=args.local_dir,
                max_shards=args.max_shards,
                language=args.language
            )
        
        # Load dataset (unless download_only is specified)
        if not args.download_only:
            dataset = load_dataset_shards(
                local_dir=args.local_dir,
                max_shards=args.max_shards,
                language=args.language
            )
            
            logger.info("Dataset loading completed successfully")
            logger.info(f"Total examples: {len(dataset)}")
            
            # Optional: Save dataset info
            info_file = os.path.join(args.local_dir, "dataset_info.txt")
            with open(info_file, "w") as f:
                f.write(f"Language: {args.language}\n")
                f.write(f"Shards loaded: {args.max_shards}\n")
                f.write(f"Total examples: {len(dataset)}\n")
                f.write(f"Dataset features: {list(dataset.features.keys())}\n")
            
            logger.info(f"Dataset info saved to {info_file}")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 