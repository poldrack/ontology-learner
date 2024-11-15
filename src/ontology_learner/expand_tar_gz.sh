#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

# Get the source and destination directories from the arguments
SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory '$SOURCE_DIR' does not exist."
  exit 1
fi

# Check if the destination directory exists, create it if it doesn't
if [ ! -d "$DEST_DIR" ]; then
  echo "Destination directory '$DEST_DIR' does not exist. Creating it now..."
  mkdir -p "$DEST_DIR"
fi

# Loop through each .tar.gz file in the source directory
for TAR_FILE in "$SOURCE_DIR"/*.tar.gz; do
  # Check if the file exists (handles the case of no .tar.gz files)
  if [ ! -f "$TAR_FILE" ]; then
    echo "No .tar.gz files found in the source directory."
    break
  fi

  # Extract the file to the destination directory
  echo "Extracting '$TAR_FILE' to '$DEST_DIR'..."
  tar -xzvf "$TAR_FILE" -C "$DEST_DIR"
  
  if [ $? -eq 0 ]; then
    echo "Extraction of '$TAR_FILE' successful."
  else
    echo "Error extracting '$TAR_FILE'."
  fi
done

echo "All extractions complete."
