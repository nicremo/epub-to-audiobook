# EPUB/HTML to Audiobook Converter

This project is a Python-based tool that converts the content of EPUB or HTML files into an audiobook. The tool extracts the book text, filters out unwanted metadata, segments the text into meaningful sentences, and groups these into manageable chunks. Each chunk is then converted into an MP3 audio file using a pre-trained Text-to-Speech (TTS) model (Tacotron2-DDC for German). Finally, the individual chapter audio files are concatenated into a complete audiobook.

## Features

- **Text Extraction:** Supports both EPUB and HTML file formats.
- **Text Filtering:** Removes unwanted metadata (e.g., ISBNs, headers in uppercase) so that only the relevant book content remains.
- **Sentence Segmentation & Chunking:** Splits the text into individual sentences using NLTK and groups them into chunks (max. 1000 characters) without cutting sentences in half.
- **Text-to-Speech (TTS):** Uses a pre-trained TTS model to convert text chunks into MP3 audio files.
- **Audio Concatenation:** Merges the generated audio chunks into a final complete audiobook.
- **Web Interface:** An interactive Gradio web interface allows file upload, shows progress in real-time, and provides both the individual chapter audios and the complete audiobook for playback and download.
- **Timeout Management:** Each text chunk is processed in a separate thread with a 90-second timeout to prevent hanging.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nicremo/epub-to-audiobook.git
   cd epub-to-audiobook

2.	**(Optional) Create and Activate a Virtual Environment**:
	```bash
	python -m venv venv
	source venv/bin/activate   # On Windows: venv\Scripts\activate

3.	**Install Dependencies**:
	```bash
 	pip install -r requirements.txt
Note: For pydub to work, ensure that ffmpeg is installed on your system.

## Usage

Run the tool by executing:

	python epub_to_audiobook.py

This will launch a Gradio web interface where you can upload an EPUB or HTML file. The conversion process is displayed in real time, and you will receive both the individual chapter audio files and the complete audiobook (in MP3 format).

## **Credits & Fork Notice**

This repository is a fork based on the work of Hussain Mustafa. The original source code was published on his website:
Source Code & Tutorials.

## License

This project is licensed under the MIT License – see the LICENSE file for details.

