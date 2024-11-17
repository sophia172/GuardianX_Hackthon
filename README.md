# GuardianX: Camera AI Hardware Project

## Overview

LGuardianX is an AI-driven project for the Llama Impact Hackathon designed to enhance emergency response systems by leveraging camera hardware and AI models. The system captures images, analyzes them for potential threats or emergencies, and facilitates communication with emergency services through AI-generated responses.

## Features

- **Image Capture and Analysis**: Utilizes camera hardware to capture images and analyze them for potential threats or emergency situations.
- **AI-Driven Communication**: Employs AI models to generate responses and communicate with emergency services.
- **Location Detection**: Automatically fetches the location of the incident using IP-based geolocation.
- **Audio Transcription**: Records and transcribes audio to assist in emergency communication.
- **Voice Synthesis**: Uses ElevenLabs API for generating voice responses.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/llama-hackathon.git
   cd llama-hackathon
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Copy the `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

   - `GROQ_API_KEY`: Your Groq API key.
   - `ELEVENLABS_API_KEY`: Your ElevenLabs API key.
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `b7e1c05fa6c77f`: Your IPInfo token for location services.

## Usage

1. **Run the Main Program**:
   Start the program using:
   ```bash
   python groq_prompt.py
   ```

2. **Functionality**:
   - The system will open camera feeds, capture images, and analyze them for emergency situations.
   - If a threat is detected, it will initiate a conversation with emergency services using AI-generated responses.
   - The program will log all activities and decisions in `debug_groq.log`.

## Project Structure

- `groq_prompt.py`: Main script for running the AI-driven emergency response system.
- `json_structure.py`: Defines the data models and prompts used for AI decision-making.
- `.env`: Environment variables for API keys and tokens.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Llama, ElevenLabs, and Groq for providing the APIs and tools that power this project.
- Special thanks to the Llama-Hackathon London team for their hard work and dedication.
