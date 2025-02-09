# epub-to-audiobook

# EPUB/HTML to Audiobook Converter

This Python-based tool converts the content of EPUB or HTML files into an audiobook. It extracts text from the file, filters out unwanted metadata, segments the text into natural sentences and groups them into manageable chunks, and then converts each chunk into an MP3 file using a pre-trained Text-to-Speech (TTS) model (Tacotron2-DDC for German). Finally, it concatenates the individual audio files into a complete audiobook.

## Features

- **Text Extraction:** Supports both EPUB and HTML file formats.
- **Text Filtering:** Removes unwanted metadata (e.g., ISBNs, headers in uppercase).
- **Sentence Segmentation & Chunking:** Splits the text into sentences and groups them into chunks (max. 1000 characters) without cutting sentences in half.
- **Text-to-Speech (TTS):** Uses a pre-trained TTS model to convert text chunks into MP3 audio files.
- **Audio Concatenation:** Merges individual audio chunks into a final complete audiobook.
- **Web Interface:** An interactive interface built with Gradio allows file upload and real-time progress updates.
- **Timeout Management:** Each text chunk is processed in a separate thread with a 90-second timeout to prevent hangs.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/epub-to-audiobook.git
   cd epub-to-audiobook

2.	**(Optional) Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

2.	(Optional) Create and Activate a Virtual Environment:

