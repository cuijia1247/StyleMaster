#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiArt3 Small Dataset Script
Created for SubStyle Classification project

This script copies the first 200 images from each subfolder in WikiArt3/train/ and WikiArt3/test/
to WikiArt3_small/train/ and WikiArt3_small/test/ respectively, maintaining the same folder structure.
"""

import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

def create_directory_structure(base_path):
    """
    Create the target directory structure
    """
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir, test_dir

def get_image_files(folder_path, limit=200):
    """
    Get the first N image files from a folder
    """
    # Common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    # Sort files to ensure consistent selection
    image_files.sort()
    
    # Return first N files
    return image_files[:limit]

def copy_images_from_folder(source_folder, target_folder, limit=200):
    """
    Copy first N images from source folder to target folder
    """
    if not os.path.exists(source_folder):
        print(f"Warning: Source folder {source_folder} does not exist")
        return 0
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Get image files
    image_files = get_image_files(source_folder, limit)
    
    copied_count = 0
    for image_file in image_files:
        try:
            filename = os.path.basename(image_file)
            target_path = os.path.join(target_folder, filename)
            shutil.copy2(image_file, target_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {image_file}: {e}")
    
    return copied_count

def process_dataset_split(source_base, target_base, split_name, limit=200):
    """
    Process train or test split
    """
    source_split = os.path.join(source_base, split_name)
    target_split = os.path.join(target_base, split_name)
    
    if not os.path.exists(source_split):
        print(f"Warning: Source {split_name} directory does not exist: {source_split}")
        return
    
    # Get all subdirectories (style folders)
    subdirs = [d for d in os.listdir(source_split) 
               if os.path.isdir(os.path.join(source_split, d))]
    
    print(f"Processing {split_name} split with {len(subdirs)} style folders...")
    
    total_copied = 0
    for subdir in tqdm(subdirs, desc=f"Processing {split_name} styles"):
        source_subdir = os.path.join(source_split, subdir)
        target_subdir = os.path.join(target_split, subdir)
        
        copied_count = copy_images_from_folder(source_subdir, target_subdir, limit)
        total_copied += copied_count
        
        if copied_count > 0:
            print(f"  {subdir}: {copied_count} images copied")
    
    print(f"Total images copied in {split_name}: {total_copied}")

def main():
    """
    Main function to create WikiArt3_small dataset
    """
    # Define paths
    base_dir = "/home/cuijia1247/Codes/SubStyleClassfication/data"
    source_dir = os.path.join(base_dir, "WikiArt3")
    target_dir = os.path.join(base_dir, "WikiArt3_small")
    
    # Number of images to copy from each subfolder
    images_per_style = 200
    
    print("WikiArt3 Small Dataset Creation")
    print("=" * 50)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Images per style: {images_per_style}")
    print()
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create target directory structure
    print("Creating target directory structure...")
    create_directory_structure(target_dir)
    
    # Process train split
    print("\nProcessing train split...")
    process_dataset_split(source_dir, target_dir, "train", images_per_style)
    
    # Process test split
    print("\nProcessing test split...")
    process_dataset_split(source_dir, target_dir, "test", images_per_style)
    
    print("\nWikiArt3_small dataset creation completed!")
    
    # Print summary
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    
    if os.path.exists(train_dir):
        train_styles = len([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
        print(f"Train styles: {train_styles}")
    
    if os.path.exists(test_dir):
        test_styles = len([d for d in os.listdir(test_dir) 
                          if os.path.isdir(os.path.join(test_dir, d))])
        print(f"Test styles: {test_styles}")

if __name__ == "__main__":
    main()
