import os                     # Module for operating system functions (e.g., creating directories, handling file paths)
import base64                 # Used for encoding binary data in Base64 (important for embedding audio files as data URIs in HTML)
import torch                  # PyTorch is used to run the TTS model (a deep learning framework)
from ebooklib import epub     # Used to read and parse EPUB files
from bs4 import BeautifulSoup # BeautifulSoup for parsing HTML/XML and extracting the plain text
import nltk                   # Natural Language Toolkit for text processing (e.g., sentence tokenization)
import re                     # Regular expressions for text cleaning and filtering
from TTS.api import TTS       # Interface to a pre-trained Text-to-Speech model
from pydub import AudioSegment# Used to load, process, and concatenate audio files
import gradio as gr           # Gradio is used to create an interactive web interface
import concurrent.futures     # Allows parallel processing and timeout management

# Download the required NLTK resources (the Punkt models for sentence tokenization)
nltk.download('punkt')
nltk.download('punkt_tab')

# Ensure that the output directory exists to store generated audio files
os.makedirs("outputs", exist_ok=True)

# Define the TTS model (Tacotron2-DDC for German) and set the device (GPU if available, otherwise CPU)
MODEL_NAME = "tts_models/de/thorsten/tacotron2-DDC"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tts = TTS(model_name=MODEL_NAME).to(device)

# ------------------------------------------------------------------------------
# Function: extract_text_from_epub
# ------------------------------------------------------------------------------
def extract_text_from_epub(epub_path):
    """
    Reads an EPUB file and extracts the plain text from it.
    
    Process:
      - Uses ebooklib to open and parse the EPUB file.
      - For each XHTML item, BeautifulSoup is used to parse and remove HTML tags.
      - The extracted text lines are joined, and empty lines are removed.
    """
    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        raise RuntimeError(f"Error reading the EPUB file: {e}")
    text = ""
    # Process each item that contains XHTML content
    for item in book.get_items():
        if item.media_type == "application/xhtml+xml":
            soup = BeautifulSoup(item.content, "html.parser")
            cleaned_text = soup.get_text(separator="\n").strip()
            text += cleaned_text + "\n\n"
    # Remove extra blank lines
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

# ------------------------------------------------------------------------------
# Function: extract_text_from_html
# ------------------------------------------------------------------------------
def extract_text_from_html(html_content):
    """
    Parses HTML content and extracts the plain text.
    
    - Uses BeautifulSoup to remove HTML tags.
    - Replaces HTML entities (e.g., &nbsp;) with normal characters.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n").strip()
    text = text.replace("\xa0", " ")
    return text

# ------------------------------------------------------------------------------
# Function: filter_relevant_text
# ------------------------------------------------------------------------------
def filter_relevant_text(text: str) -> str:
    """
    Filters the extracted text to remove lines that likely contain only metadata or headers,
    such as lines composed solely of numbers, special characters, or all uppercase (without
    typical sentence-ending punctuation).
    """
    lines = text.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Remove lines consisting only of digits, non-word characters, or underscores
        if re.match(r'^[\d\W_]+$', stripped):
            continue
        # Remove short uppercase lines (likely headers) without sentence-ending punctuation
        if stripped.isupper() and len(stripped) < 50 and not re.search(r'[.!?]$', stripped):
            continue
        filtered.append(stripped)
    return "\n".join(filtered)

# ------------------------------------------------------------------------------
# Function: clean_text_for_tts
# ------------------------------------------------------------------------------
def clean_text_for_tts(text: str) -> str:
    """
    Performs additional cleaning steps to prepare the text for the TTS model.
    
    - Removes control characters.
    - Replaces multiple spaces with a single space.
    - Reduces repeated characters (e.g., "WWWWWWW" becomes "WWW").
    """
    # Remove control characters (ASCII 0-31 and 127)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Reduce repetitions of any character (more than 3 in a row) to three repetitions
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    return text.strip()

# ------------------------------------------------------------------------------
# Function: text_to_audio
# ------------------------------------------------------------------------------
def text_to_audio(text, output_dir, max_chunk_length=1000):
    """
    Converts the cleaned text into audio files.
    
    Process:
      1. Filter and clean the entire text.
      2. Split the text into sentences using nltk.tokenize.sent_tokenize (German language).
      3. Combine sentences into chunks that do not exceed max_chunk_length characters.
      4. For each chunk:
         - Clean the chunk further.
         - Call the TTS model in a separate thread with a 90-second timeout to avoid hangs.
         - Load the generated audio file.
      5. Concatenate all audio chunks into a final audiobook.
      6. Update the web interface (HTML output) after each step.
    """
    # Step 1: Filter and clean the entire text
    text = filter_relevant_text(text)
    text = clean_text_for_tts(text)
    
    # Step 2: Split text into sentences (German)
    sentences = nltk.tokenize.sent_tokenize(text, language="german")
    chunks = []
    current_chunk = ""
    # Step 3: Group sentences into chunks without exceeding the character limit
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    audio_segments = []  # List to store loaded audio segments
    audio_paths = []     # List to store the file paths of generated MP3 files
    yield build_audio_html([], []), None  # Initial UI update
    
    # Step 4: Process each text chunk
    for idx, chunk in enumerate(chunks):
        # Further clean each chunk
        chunk = clean_text_for_tts(chunk)
        output_path = os.path.join(output_dir, f"part_{idx+1}.mp3")
        print(f"🔊 Generating audio for chunk {idx+1}/{len(chunks)}...")
        # Use concurrent.futures to execute the TTS call with a timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tts.tts_to_file, text=chunk, file_path=output_path)
            try:
                # Wait up to 90 seconds for this chunk
                future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                print(f"Timeout in chunk {idx+1} – skipping this chunk.")
                output_path = None
            except Exception as e:
                print(f"Error generating chunk {idx+1}: {e}")
                output_path = None

        try:
            if output_path:
                audio_segment = AudioSegment.from_file(output_path)
            else:
                audio_segment = None
        except Exception as e:
            print(f"Error loading file {output_path}: {e}")
            audio_segment = None
        
        if audio_segment:
            audio_segments.append(audio_segment)
        audio_paths.append(output_path)
        yield build_audio_html(audio_paths, [None]*len(audio_paths)), None

    # Step 5: Concatenate all audio chunks into a final audiobook
    try:
        valid_segments = [seg for seg in audio_segments if seg is not None]
        if not valid_segments:
            raise RuntimeError("No valid audio files found for concatenation.")
        final_audio = sum(valid_segments)
        final_audio_path = os.path.join(output_dir, "audiobook.mp3")
        final_audio.export(final_audio_path, format="mp3")
    except Exception as e:
        print(f"Error concatenating audio files: {e}")
        final_audio_path = None
        yield build_audio_html(audio_paths, [None]*len(audio_paths) + [f"<div class='error'>Error: {e}</div>"]), None
        return

    # Step 6: Final UI update with the final audiobook
    yield build_audio_html(audio_paths, [None]*len(audio_paths)), final_audio_path

# ------------------------------------------------------------------------------
# Function: file_to_data_uri
# ------------------------------------------------------------------------------
def file_to_data_uri(file_path, mime="audio/mp3"):
    """
    Reads a file, encodes it in Base64, and returns a data URI.
    
    This enables the audio file to be directly embedded in an HTML audio player.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

# ------------------------------------------------------------------------------
# Function: process_file
# ------------------------------------------------------------------------------
def process_file(file_obj):
    """
    Reads the uploaded file and decides whether it is an EPUB file (using extract_text_from_epub)
    or HTML content (using extract_text_from_html). The extracted text is then filtered,
    cleaned, and passed to text_to_audio() for conversion.
    """
    try:
        file_path = file_obj.name
        print(f"📖 Processing file: {file_path}")
        if file_path.lower().endswith('.epub'):
            text = extract_text_from_epub(file_path)
        else:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            text = extract_text_from_html(content)
        # Filter and clean the extracted text
        text = filter_relevant_text(text)
        text = clean_text_for_tts(text)
        print(f"✏️ Extracted text (first 500 characters):\n{text[:500]}...\n")
    except Exception as e:
        yield f"<div class='error'>Error processing file: {e}</div>", None
        return
    yield from text_to_audio(text, output_dir="outputs")

# ------------------------------------------------------------------------------
# Function: build_audio_html
# ------------------------------------------------------------------------------
def build_audio_html(audio_paths, errors):
    """
    Constructs an HTML block that displays an audio player and download link for each chapter.
    
    CSS styling is embedded for a clean and appealing visual presentation.
    """
    css = """
    <style>
    .audio-container {
      border: 1px solid #ccc;
      padding: 10px;
      border-radius: 5px;
      margin-bottom: 20px;
      background: #f9f9f9;
    }
    .audio-container p {
      font-weight: bold;
      margin: 0 0 5px 0;
    }
    .audio-container audio {
      width: 100%;
      margin-bottom: 5px;
    }
    .audio-container a {
      color: #007bff;
      text-decoration: none;
    }
    .audio-container a:hover {
      text-decoration: underline;
    }
    .error {
      color: red;
      font-weight: bold;
    }
    </style>
    """
    html = css
    for i, path in enumerate(audio_paths):
        html += f"<div class='audio-container'>"
        html += f"<p>Chapter {i+1}</p>"
        if path:
            data_uri = file_to_data_uri(path)
            html += f"<audio controls src='{data_uri}'></audio><br>"
            html += f"<a href='{data_uri}' download='Chapter_{i+1}.mp3'>Download Chapter {i+1}</a>"
        else:
            html += "<div class='error'>Error in this chapter.</div>"
        html += "</div>"
    return html

# ------------------------------------------------------------------------------
# Gradio Web Interface
# ------------------------------------------------------------------------------
demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="📂 Upload a file (EPUB or HTML)"),
    outputs=[
        gr.HTML(label="🎵 Chapter Audios"),    # Displays the HTML overview of chapters
        gr.Audio(label="📖 Complete Audiobook") # Plays the concatenated audiobook audio
    ],
    title="📖 EPUB/HTML to Audiobook Converter",
    description="Upload an EPUB or HTML file. The text is segmented based on natural sentence boundaries, filtered, and further cleaned so that only the relevant book content is converted into audio files."
)

# Enable real-time updates and launch the interface with a public share link
demo.queue()
demo.launch(share=True)
