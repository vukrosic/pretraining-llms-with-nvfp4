#!/usr/bin/env python3
"""
PDF to Markdown Converter with OCR
Extracts text from PDF files and converts to well-formatted Markdown.
"""

import os
import sys
from pathlib import Path
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re

def extract_text_from_pdf(pdf_path, use_ocr=False):
    """
    Extract text from PDF using either direct text extraction or OCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        use_ocr (bool): Whether to use OCR for text extraction
    
    Returns:
        str: Extracted text content
    """
    text_content = []
    
    if use_ocr:
        print("Using OCR to extract text from PDF...")
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1}/{len(images)}...")
                
                # Use OCR to extract text from image
                page_text = pytesseract.image_to_string(image, lang='eng')
                text_content.append(f"## Page {i+1}\n\n{page_text}\n\n")
                
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            print("Falling back to direct text extraction...")
            use_ocr = False
    
    if not use_ocr:
        print("Using direct text extraction from PDF...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    print(f"Processing page {i+1}/{len(pdf.pages)}...")
                    
                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"## Page {i+1}\n\n{page_text}\n\n")
                        
        except Exception as e:
            print(f"Direct text extraction failed: {e}")
            return None
    
    return "\n".join(text_content)

def clean_and_format_text(text):
    """
    Clean and format the extracted text for better Markdown output.
    
    Args:
        text (str): Raw extracted text
    
    Returns:
        str: Cleaned and formatted text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Fix common OCR issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Add space between word and number
    text = re.sub(r'(\d)(\w)', r'\1 \2', text)  # Add space between number and word
    
    # Clean up bullet points and lists
    text = re.sub(r'^[\s]*[•·▪▫]\s*', '- ', text, flags=re.MULTILINE)
    
    # Clean up section headers (lines that are all caps or start with numbers)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            cleaned_lines.append('')
            continue
            
        # Convert all-caps lines to headers (if they're not too long)
        if line.isupper() and len(line.strip()) < 100 and len(line.strip()) > 3:
            cleaned_lines.append(f"### {line.title()}")
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def save_markdown(text, output_path):
    """
    Save the extracted text as a Markdown file.
    
    Args:
        text (str): Text content to save
        output_path (str): Path for the output Markdown file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Markdown file saved to: {output_path}")
    except Exception as e:
        print(f"Error saving Markdown file: {e}")

def main():
    """Main function to convert PDF to Markdown."""
    
    # Check if PDF file exists
    pdf_path = "Pretraining_Large_Language_Models_with_NVFP4.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        return
    
    # Determine output filename
    output_path = "Pretraining_Large_Language_Models_with_NVFP4.md"
    
    print(f"Converting PDF: {pdf_path}")
    print(f"Output file: {output_path}")
    
    # Try direct text extraction first
    print("\n=== Attempting direct text extraction ===")
    text = extract_text_from_pdf(pdf_path, use_ocr=False)
    
    # If direct extraction fails or returns little text, try OCR
    if not text or len(text.strip()) < 100:
        print("\n=== Direct extraction yielded little text, trying OCR ===")
        text = extract_text_from_pdf(pdf_path, use_ocr=True)
    
    if not text:
        print("Error: Could not extract text from PDF!")
        return
    
    print(f"\nExtracted {len(text)} characters of text")
    
    # Clean and format the text
    print("Cleaning and formatting text...")
    cleaned_text = clean_and_format_text(text)
    
    # Add a title and metadata
    markdown_content = f"""# Pretraining Large Language Models with NVFP4

*Extracted from PDF document*

---

{cleaned_text}

---

*This document was automatically converted from PDF to Markdown.*
"""
    
    # Save the Markdown file
    save_markdown(markdown_content, output_path)
    
    print(f"\nConversion complete! Check '{output_path}' for the result.")

if __name__ == "__main__":
    main()
