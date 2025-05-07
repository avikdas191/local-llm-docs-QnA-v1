# import pip
# pip.main(['install', '<package_name>'])
# pip install -q nltk
# pip install -q einops
# pip install -q pyngrok
# pip install -q faiss-cpu
# pip install -q matplotlib
# pip install -q pdfplumber
# pip install -q python-docx
# pip install -q flask-ngrok
# pip install -q -U sentence-transformers

import os
import re
import warnings
from datetime import datetime
from random import random

import docx
import faiss
import nltk
import numpy as np
import pdfplumber
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_ngrok import run_with_ngrok
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pyngrok import ngrok
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Download the 'punk' tokenizer models from NLTK
# nltk.download('punk')

# Create an instance of the Flask class
app = Flask(__name__)
# Use the run_with_ngrok function to enable Ngrok tunneling for the Flask app
run_with_ngrok(app)

# Define the path to a folder in Google Drive where documents are stored
folder_path = 'documents'


# Document Parsing
# ----------------


def extract_text_by_page_with_metrics_from_folder(folder_path):
    text_by_page = []

    # Iterate over each file in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # print(file_path)
        if filename.endswith(".pdf"):
            # Extract text and metrics from each PDF file
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()

                    # Replace newline characters ('\n') with spaces (' ')
                    page_text = page_text.replace('\n', ' ')

                    # Append page data to the list
                    text_by_page.append({
                        # Calculate character count
                        "char_count": len(page_text),
                        # Calculate word count
                        "word_count": len(word_tokenize(page_text)),
                        # Calculate sentence count (using raw tokenization)
                        "sentence_count": len(sent_tokenize(page_text)),
                        # Calculate average token count per word
                        "token_count": len(page_text) / 4,  # 1 token = ~4 characters,
                        "text": page_text
                    })

        elif filename.endswith(".docx"):
            # Extract text and metrics from each Word document
            doc = docx.Document(file_path)
            merged_text = ""
            word_count = 0

            for paragraph in doc.paragraphs:
                # Replace non-breaking space ('\xa0') with regular space (' ')
                paragraph_text = paragraph.text.replace('\xa0', ' ')

                # Tokenize paragraph text to calculate word count
                words = word_tokenize(paragraph_text)
                paragraph_word_count = len(words)

                if paragraph_word_count < 50:
                    # Merge the current paragraph with the previous one
                    merged_text += " " + paragraph_text
                else:
                    # Append the merged text (if any) to the list
                    if merged_text:
                        text_by_page.append({
                            "char_count": len(merged_text),
                            "word_count": len(word_tokenize(merged_text)),
                            "sentence_count": len(sent_tokenize(merged_text)),
                            "token_count": len(merged_text) / 4,
                            "text": merged_text.strip()
                        })

                    # Reset for the next paragraph
                    merged_text = paragraph_text
                    word_count = paragraph_word_count

            # Append the last merged text (if any) to the list
            if merged_text:
                text_by_page.append({
                    "char_count": len(merged_text),
                    "word_count": len(word_tokenize(merged_text)),
                    "sentence_count": len(sent_tokenize(merged_text)),
                    "token_count": len(merged_text) / 4,
                    "text": merged_text.strip()
                })

    return text_by_page


# Extract text with metrics from PDF and Word files in the specified folder
# text_segments_by_page = extract_text_by_page_with_metrics_from_folder(folder_path)
# print(text_segments_by_page[17:19])

# Create a DataFrame from a list of dictionaries
# text_segments_by_page_df = pd.DataFrame(text_segments_by_page)

# Display the first 5 rows of the DataFrame to inspect its contents
# print(text_segments_by_page_df.head())

# Splitting pages into sentences
# ------------------------------


def split_pages_into_sentences(file):
    # Initialize the English language model provided by spaCy
    nlp = English()

    # Adding a sentencizer pipeline, see https://spacy.io/api/sentencizer
    nlp.add_pipe("sentencizer")
    # it is not just splitting sentences by ., but also it is accounting for other
    # rules and statistics. And it is robust.

    for file_item in file:
        file_item["sentences"] = list(nlp(file_item["text"]).sents)

        # Make sure all sentences are strings (the default type is a spaCy datatype)
        file_item["sentences"] = [str(sentence) for sentence in file_item["sentences"]]

        # Count the sentences
        file_item["page_sentence_count_spacy"] = len(file_item["sentences"])

    return file


# text_segments_by_page = split_pages_into_sentences(text_segments_by_page)
# print(text_segments_by_page[17:19])

# Chunking our sentences together (Chunk_size and overlap)
# --------------------------------------------------------


def chunk_sentences(file):
    # Define split size to turn groups of sentences into chunks
    num_sentence_chunk_size = 4

    # Create a function to split lists of texts recursively into chunk size
    # e.g.  if chunk size = 10, then [20] -> [10, 10] or [25] -> [10, 10, 5]
    def split_list(input_list: list[str],
                   slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
        return [input_list[i:i + slice_size + 2] for i in range(0, len(input_list), slice_size)]
        # +2 is for overlapping

    # Loop through pages and texts and split sentences into chunks
    for file_item in file:
        file_item["sentence_chunks"] = split_list(input_list=file_item["sentences"],
                                                  slice_size=num_sentence_chunk_size)
        file_item["num_chunks"] = len(file_item["sentence_chunks"])

    return file


# text_segments_by_page = chunk_sentences(text_segments_by_page)
# print(text_segments_by_page[17:19])

# Splitting each chunk into its own item (chunk information included)
# -------------------------------------------------------------------


def split_chunk_into_own_item(file):
    # Split each chunk into its own item
    page_and_chunk = []
    for file_item in file:
        for sentence_chunk in file_item["sentence_chunks"]:
            chunk_dict = {}
            # chunk_dict["page_number"] = file_item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one
            # paragraph
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" => ". A" (will work for any capital letter)

            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get some stats on our chunks
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 chars

            page_and_chunk.append(chunk_dict)

    return page_and_chunk


# pages_and_chunks = split_chunk_into_own_item(text_segments_by_page)
# print(pages_and_chunks[17:19])

# Filter chunks of text based on word count (data visualization)
# --------------------------------------------------------------


def filter_chunks_on_word_count(file):
    df = pd.DataFrame(file)
    # print(df)

    # Filter our DataFrame for rows with under 11 words
    word_length = 11
    file_over_min_word = df[df["chunk_word_count"] >= word_length].to_dict(orient="records")

    return file_over_min_word


# pages_and_chunks_over_min_word = filter_chunks_on_word_count(pages_and_chunks)
# print(pages_and_chunks_over_min_word[17:19])

# Perform embedding on our text chunks
# ------------------------------------


def embedding_on_text_chunks(file):
    # Suppress the FutureWarning from the transformers library
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

    # Load the pre-trained model
    sent_trans_model = SentenceTransformer('all-mpnet-base-v2')

    # Embedding = sent_trans_model.encode("Hello World !!!")
    # Embedding.shape  # For Debugging

    for file_item in file:
        # Access the 'sentence_chunk' value of the current item and use the sent_trans_model to encode it
        # Set convert_to_tensor=False to get the embedding as a numpy array (not a PyTorch tensor)
        file_item["embedding"] = sent_trans_model.encode(file_item["sentence_chunk"], convert_to_tensor=False)

    return file


# pages_and_chunks_over_min_word = embedding_on_text_chunks(pages_and_chunks_over_min_word)
# print(pages_and_chunks_over_min_word[17:19])

# Saving embeddings into csv file
# -------------------------------

# # Save embeddings to file
# sentence_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_word)
# embeddings_df_save_path = "sentence_chunks_and_embeddings_df.csv"
# sentence_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# Fetch data from the csv file
# ----------------------------


def fetch_data_from_csv():
    # Import texts and embedding df
    chunk_and_embedding_df = pd.read_csv("sentence_chunks_and_embeddings_df.csv")

    # Convert embedding column back to np.array (it got converted to string when it saved to CSV)
    chunk_and_embedding_df["embedding"] = chunk_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Stack embeddings into a NumPy array
    embedding_array = np.stack(chunk_and_embedding_df["embedding"].tolist(), axis=0).astype('float32')

    # Convert texts and embedding df to list of dicts
    chunk_and_embedding = chunk_and_embedding_df.to_dict(orient="records")

    return embedding_array, chunk_and_embedding_df


[embeddings_array, chunks_and_embeddings_df] = fetch_data_from_csv()
# print(chunks_and_embeddings_df) # Debugging


# print(chunks_and_embeddings_df.describe())

# Data visualization plots and images
# -----------------------------------

# # Plot histograms for chunk_char_count, chunk_word_count, and chunk_token_count
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 3, 1)
# plt.hist(chunks_and_embeddings_df['chunk_char_count'], bins=30, color='skyblue', alpha=0.7)
# plt.title('Character Count Distribution')
# plt.xlabel('Chunk Character Count')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 3, 2)
# plt.hist(chunks_and_embeddings_df['chunk_word_count'], bins=30, color='salmon', alpha=0.7)
# plt.title('Word Count Distribution')
# plt.xlabel('Chunk Word Count')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 3, 3)
# plt.hist(chunks_and_embeddings_df['chunk_token_count'], bins=30, color='lightgreen', alpha=0.7)
# plt.title('Token Count Distribution')
# plt.xlabel('Chunk Token Count')
# plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()

# # Scatter plot between chunk_word_count and chunk_char_count
# plt.figure(figsize=(8, 6))
# plt.scatter(chunks_and_embeddings_df['chunk_char_count'], chunks_and_embeddings_df['chunk_word_count'], alpha=0.5)
# plt.title('Relationship between Character Count and Word Count')
# plt.xlabel('Chunk Character Count')
# plt.ylabel('Chunk Word Count')
# plt.show()

# Defining sentence transformer and llm
# -------------------------------------


def sent_trans_and_llm():
    #  Define the name of the pre-trained SentenceTransformer model to use
    sent_trans_model_name = "all-mpnet-base-v2"

    # Create an instance of SentenceTransformer using the specified model name
    sent_trans = SentenceTransformer(sent_trans_model_name)

    # Specify the name of the pre-trained question answering model to use
    qa_model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

    # Load the tokenizer associated with the specified pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    # Load the pre-trained question answering model
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

    # Create a question-answering pipeline using the loaded model and tokenizer
    qa_pipe = pipeline(
        "question-answering",
        model=qa_model,  # Specify the loaded question answering model
        tokenizer=tokenizer,  # Specify the loaded tokenizer
        max_length=512,  # Set the maximum length of input sequences
        max_answer_length=1000,  # Set the maximum length of the generated answers
    )

    return sent_trans, qa_pipe


[sent_trans_model, qa_pipeline] = sent_trans_and_llm()

# Faiss similarity search, integration with LLM and UI implementation
# -------------------------------------------------------------------

app = Flask(__name__)

data = {
    "hi": {
        "patterns": ["hello", "hi", "hey"],
        "responses": ["Hi there!", "Hello!", "Hey!"]
    },
    "how_are_you": {
        "patterns": ["how are you", "how are you doing"],
        "responses": ["I'm doing well, thank you!", "Not too bad, thanks for asking!", "I'm great, how about you?"]
    },
    "fine_great": {
        "patterns": ["fine", "great", "I am fine", "I am doing great"],
        "responses": ["Great to hear that, How can I help you?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye"],
        "responses": ["Goodbye!", "See you later!", "Bye!"]
    },
    "bot_name": {
        "patterns": ["what are you", "what's your name", "who are you", "your name"],
        "responses": ["I'm a chatbot!", "I go by many names, but you can call me ChatBot.",
                      "I'm your friendly ChatBot!"]
    },
    "thanks": {
        "patterns": ["thank you", "thanks"],
        "responses": ["You're welcome!", "No problem!", "Anytime!"]
    },
    "help": {
        "patterns": ["help", "what can you do", "need assistance"],
        "responses": ["Sure, I can assist you with various tasks. Just ask me anything!",
                      "I'm here to help. Feel free to ask me anything!",
                      "If you need assistance, don't hesitate to ask!"]
    },
    "confusion": {
        "patterns": ["I'm confused", "don't know"],
        "responses": ["ok, let me know when you are ready", "sorry to hear that"]
    },
    "create": {
        "patterns": ["who created you", "who is your creator"],
        "responses": ["Avi created me using PyTorch's machine learning library.", "top secret ;)"]
    },
    "exoplanet": {
        "patterns": ["what are exoplanets", "what is exoplanet"],
        "responses": ["Exoplanets are planets that orbit stars outside our solar system"]
    },
    "detect_exoplanet": {
        "patterns": ["how do we detect exoplanet", "detect exoplanet"],
        "responses": [
            "Exoplanets are detected using various methods, including transit photometry (observing the dimming of a "
            "star as a planet passes in front of it) and radial velocity (detecting the wobble of a star caused by an "
            "orbiting planet)."]
    },
    "dark_matter": {
        "patterns": ["what is dark matter", "dark matter"],
        "responses": [
            "Dark matter is an invisible substance that makes up about 27% of the universe's mass and energy, "
            "yet its nature remains unknown and it does not emit, absorb, or reflect light."]
    },
    "classify_galaxies": {
        "patterns": ["galaxies classified", "classify galaxy"],
        "responses": [
            "Galaxies are classified based on their shape, including spiral, elliptical, and irregular, determined by "
            "their visual appearance and structure."]
    },
    "types_galaxies": {
        "patterns": ["What are the different types of galaxies", "type galaxy", "types galaxies"],
        "responses": [
            "The different types of galaxies include spiral galaxies (with arms of stars and gas), elliptical "
            "galaxies (oval-shaped and mostly older stars), and irregular galaxies (lacking a defined shape, "
            "often with active star formation)."]
    },
    "gravitational_waves": {
        "patterns": ["gravitational waves", "gravitational wave"],
        "responses": ["ok, let me know when you are ready", "sorry to hear that"]
    },
    "detect_gravitational_waves": {
        "patterns": ["How are the gravitational waves detected", "detect gravitational waves"],
        "responses": [
            "Gravitational waves are detected using laser interferometers that measure tiny fluctuations in spacetime "
            "caused by passing gravitational waves."]
    },
    "time_date": {
        "patterns": ["time", "date", "what is the time", "what time is it", "what's today's date", "today's date"],
        "responses": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
}


# Function to preprocess input text
def preprocess(text):
    text = text.lower()
    return text


# Function to generate response based on input text using LLM
def generate_response(input_text):
    # Initialize a FAISS index optimized for cosine similarity
    dimension = embeddings_array.shape[1]  # Dimension of embeddings
    faiss_index = faiss.IndexFlatIP(dimension)  # IP (inner product) index for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)

    # Add normalized embeddings to the FAISS index
    faiss_index.add(embeddings_array)

    input_text = preprocess(input_text)

    # Check if input_text matches any predefined patterns
    for intent, intent_data in data.items():
        for pattern in intent_data["patterns"]:
            if pattern in input_text:
                if intent == "time_date":
                    return random.choice(intent_data["responses"])
                else:
                    return random.choice(intent_data["responses"])

    # Encode user input into an embedding using Sentence Transformer
    query_embedding = sent_trans_model.encode(input_text, convert_to_tensor=False)

    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(np.array([query_embedding]).astype('float32'))

    # Perform similarity search with the FAISS index using search
    k = 5  # Number of nearest neighbors to retrieve
    D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
    # Search for k nearest neighbors based on the query_embedding

    # Retrieve relevant text chunks based on indices for the current user input
    relevant_indices = I[0]
    print(relevant_indices)  # For debugging
    relevant_chunks = chunks_and_embeddings_df.loc[relevant_indices, 'sentence_chunk'].tolist()
    # Select specific rows and column from the DataFrame

    # Combine retrieved text chunks into a single context for the user input
    llm_context = " ".join(relevant_chunks)

    # Perform question-answering for the user input using the combined context
    result = qa_pipeline(question=input_text, context=llm_context)

    # Return the answer from the LLM-based question-answering
    return result['answer']


# Route for index page
@app.route('/')
def index():
    return render_template('design.html')
    # Render the HTML template located in the templates folder


# Route for handling chat requests
@app.route('/chat', methods=['POST'])
def chat():
    # Extract the 'user_input' from the JSON data sent in the request;
    # default to empty string if not provided
    user_input = request.json.get("user_input", "")
    bot_response = generate_response(user_input)
    return jsonify(bot_response)  # Return the bot's response as JSON


if __name__ == '__main__':
    # Set the authentication token for ngrok
    ngrok.set_auth_token("2foStyihzlfOmrmx3290SpE5qQF_7uBHAAt6t9GFvSt7rdoNB")

    # Use ngrok to expose the local server running on port 5000 (HTTP protocol)
    public_url = ngrok.connect(addr="5000", proto="http")

    # Print a formatted message with the public URL generated by ngrok
    print(f"\033[1m To access the ChatBot, Please click here ==>\033[0m {public_url}")

    # Start the Flask application
    app.run()
