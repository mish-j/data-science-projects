# YouTube Video Text Summarization Using NLP and Django

This project provides an automatic YouTube video summarizer using Natural Language Processing (NLP) techniques and Django. It extracts the transcript of a YouTube video and generates a concise summary of the video's content.

## Key Features

- **YouTube Video Transcript Extraction**: Retrieves subtitles or closed captions of YouTube videos using the YouTube Transcript API.
- **Text Preprocessing**: Cleans the transcript by removing unnecessary elements like timestamps and special characters.
- **NLP Summarization**: Uses transformer-based NLP models like BART or GPT for generating summaries.
- **Django Web Interface**: Provides a simple web interface where users can input a YouTube URL and receive a summary of the video.

## Installation and Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Virtual environment (optional but recommended)

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/youtube-video-summarizer.git
cd youtube-video-summarizer

