import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from django.http import HttpResponse
from django.shortcuts import render
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
import string
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class YouTubeSummarizer:
    def __init__(self):
        # Initialize BERT summarization model
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_video_id(self, youtube_url):
        """
        Extract YouTube video ID from URL
        """
        parsed_url = urlparse(youtube_url)
        video_id = parse_qs(parsed_url.query).get('v')
        return video_id[0] if video_id else None

    def get_transcript(self, youtube_link):
        """
        Retrieve and clean YouTube video transcript
        """
        video_id = self.get_video_id(youtube_link)
        if not video_id:
            return ""

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([t['text'] for t in transcript_list])
            
            # Clean transcript
            transcript = transcript.translate(str.maketrans('', '', string.punctuation))
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            
            return transcript
        except Exception as e:
            print(f"Error retrieving transcript: {e}")
            return ""

    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stopwords, and lemmatizing
        """
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            cleaned_words = [
                self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words and word.isalpha()
            ]
            cleaned_sentences.append(" ".join(cleaned_words))
        
        return cleaned_sentences

    def extract_important_sentences(self, sentences):
        """
        Use sentence embeddings and cosine similarity to extract important sentences
        """
        embeddings = [self.nlp(sentence).vector for sentence in sentences]
        similarities = cosine_similarity(embeddings)
        scores = similarities.sum(axis=1)
        
        ranked_sentences = [sentences[i] for i in np.argsort(scores)[-5:]]  # Select top 5 sentences
        return " ".join(ranked_sentences)

    def summarize_text(self, text, max_length=150, min_length=50):
        """
        Generate summary using BERT-based model
        """
        if not text:
            return "No text to summarize"
        
        # Preprocess the text
        cleaned_sentences = self.preprocess_text(text)
        
        # Extract important sentences using similarity
        important_sentences = self.extract_important_sentences(cleaned_sentences)
        
        # Prepare inputs for summarization
        inputs = self.tokenizer(
            important_sentences, 
            max_length=1024, 
            return_tensors="pt", 
            truncation=True
        )

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            max_length=max_length, 
            min_length=min_length,
            early_stopping=True
        )

        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )

        return summary

def index(request):
    """
    Render initial index page
    """
    return render(request, 'index.html')

def analyze(request):
    """
    Process YouTube link and generate summary
    """
    # Disable HuggingFace warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # Get YouTube URL from POST request
    youtube_url = request.POST.get('text', '')
    
    # Initialize summarizer
    summarizer = YouTubeSummarizer()
    
    # Get transcript
    transcript = summarizer.get_transcript(youtube_url)
    
    # Generate summary
    summary = summarizer.summarize_text(transcript)
    
    # Render results
    context = {
        'analyzed_text': summary,
        'original_transcript': transcript
    }
    
    return render(request, 'analyze.html', context)
