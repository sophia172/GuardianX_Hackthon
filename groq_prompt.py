from markdown_it.parser_block import LOGGER
from openai import OpenAI
import json
import asyncio
import aiohttp
import logging
import os
import re
import instructor
import time
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import base64
from json_structure import (
    DecisionInfo,
    decision_prompts,
    QuestionInfo,
    question_prompts,
    SituationInfo,
    situation_prompts,
)
from utils import timing
import os
import cv2
import base64
import asyncio
import aiohttp
import numpy as np
import logging
from elevenlabs import play
from elevenlabs.client import ElevenLabs, Voice, VoiceSettings

# Load the Groq API key from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug_groq.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def speak(iscop, text: str):
    try:
        role = "Police" if iscop else "Caller"
        logging.info(f"{role} speaking: {text}")
        client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

        audio = client.generate(
            text=text,
            voice=Voice(
                voice_id="oSxgHGECdZSwGi5US8gW" if iscop else "onwK4e9ZLuTAKqWW03F9",
                settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=False,
                ),
            ),
            model="eleven_turbo_v2_5",
        )

        # Check if play returns a coroutine
        if audio is not None:
            await play(audio)
        else:
            logger.error("Audio generation failed, play function returned None.")

    except Exception as e:
        logger.error(f"Error in speak function: {e}")


async def get_location():
    ipinfo_token = "b7e1c05fa6c77f"
    url = f"https://ipinfo.io?token={ipinfo_token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            # Extract relevant information
            city = data.get("city")
            region = data.get("region")
            country = data.get("country")
            lat_long = data.get("loc").split(",")
            latitude = lat_long[0]
            longitude = lat_long[1]

            logger.info(
                f"Location fetched: {city}, {region}, {country}, {latitude}, {longitude}"
            )
            return city, region, country, latitude, longitude


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        logger.info(f"Image encoded from path: {image_path}")
        return encoded_image


async def capture_images(cap1, cap2, output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)
    logging.debug("Cameras are already opened, capturing images.")
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    logging.debug("Images captured.")
    if not (ret1 and ret2):
        logging.error("Failed to capture images from the cameras.")
        return None

    stitched_image = cv2.hconcat([frame1, frame2])

    logging.debug("Images stitched together.")

    file_path = os.path.join(output_dir, "image.png")
    counter = 1
    logging.debug(f"Checking if file path {file_path} already exists.")
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"image_{counter}.png")
        logging.debug(f"File path already exists, trying {file_path + str(counter)}")
        counter += 1

    cv2.imwrite(file_path, stitched_image)
    logging.info(f"Image saved at {file_path}")

    return file_path


async def police_talk(client, msg, history):
    logger.info(f"Police talk initiated with message: {msg}.")
    # Assume msg is the real text input from a police line call
    cleaned_msg = re.sub(r"\(.*?\)", "", msg)

    # Log the real police message
    logger.info(f"Real police message: {cleaned_msg}")

    # Append history to maintain context
    police_words = cleaned_msg + history
    logger.info(f"Police response: {police_words}")
    return police_words


async def caller_talk(client, question, history):
    await speak(False, question)
    msg = f"""You are GuardianX, an AI caller who has dialed the emergency services number to report on behalf of your user.
    You are reporting an emergency situation to the police operator.
    Your user may be impaired, and you are speaking on their behalf.
    Answer the operator's questions accurately and concisely.
    Do not create additional scenarios or characters.
    The police operator asked {question}. Answer the question in a few sentences.
    """
    # Append history to maintain context
    formated_police_question = text_response_json(
        client, msg + history, question_prompts, QuestionInfo, answer=bool
    )

    logger.info(f"Caller reformatted police questions to: {formated_police_question}")
    return formated_police_question


async def image_response(base64_image, msg, system_msg):
    logger.info(f"Image check initiated with prompts: \n{msg}")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_msg,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64_image},
                    },
                ],
            },
        ],
        max_tokens=8192,
    )
    words = response.choices[0].message.content
    await speak(False, words)

    logger.info(f"Caller image response received: {words}.")
    return words


def image_check(base64_image, prompts, format_structure, answer=bool):
    logger.info("Image check initiated.")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    groq = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Ensure prompts is a dictionary
    if not isinstance(prompts, dict):
        logger.error("Prompts should be a dictionary.")
        return None

    # Request image description from the model
    description_response = groq.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64_image},
                    },
                    {
                        "type": "text",
                        "text": "Please describe the image.",
                    },
                ],
            }
        ],
        max_tokens=512,
    )

    image_description = description_response.choices[0].message.content
    logger.info(f"Image description generated: {image_description}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Image description: {image_description}\n"
                        + "\n".join(
                            [
                                f"Please answer the following question in JSON format. "
                                f"Question: {prompts[key]} "
                                f'Expected format: {{ "{key}": {answer} }}. '
                                f'Example: {{ "{key}": true }} if the answer is yes.'
                                for key in prompts
                            ]
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64_image},
                    },
                ],
            }
        ],
        max_tokens=8192,
    )

    json_response = format2json(response.choices[0].message.content, format_structure)

    logger.info(f"Image check response received: {json_response}")
    return json_response.dict()


FORMAT_CLIENT = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)


def format2json(json_str, structure):
    response = FORMAT_CLIENT.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        response_model=structure,
        messages=[
            {
                "role": "user",
                "content": json_str,
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response


async def transcribe_and_respond():
    global stop_recording
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    global stop_recording
    stop_recording = False
    await record_audio("recorded_audio.wav")
    with open("recorded_audio.wav", "rb") as audio_file:
        start_time = time.time()
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo", file=audio_file
        )
        end_time = time.time()
        logging.info(f"Transcription: {transcription.text}")
        logging.info(f"Time taken for transcription: {end_time - start_time} seconds")

        return transcription.text


async def record_audio(filename="recorded_audio.wav"):
    global stop_recording
    fs = 44100  # Sample rate
    print("Recording... Press Enter to stop.")

    try:
        # Start recording audio
        myrecording = sd.rec(int(10 * fs), samplerate=fs, channels=2)

        # Wait for the user to press Enter
        input()

        # Stop the recording
        sd.stop()
        write(filename, fs, myrecording)  # Write data to WAV file
        print("Recording finished.")
    except Exception as e:
        logging.error(f"Error during recording: {e}")


async def trigger_conversation(client, base64_image, conversation_line=10):
    logger.info("Triggering conversation with police operator.")
    answer = []
    history = ""  # Initialize history as an empty string
    # Removed the fictional initial message for the police talk
    while conversation_line > 0:
        # Removed the LLM-based police question generation
        # Transcribe the police operator's response
        real_police_input = await transcribe_and_respond()
        question = await police_talk(client, real_police_input, history)
        history += question

        character_description = """
        You are GuardianX, an AI caller who has dialed the emergency services number to report on behalf of your user.
        You were triggered by camera detection of an emergency situation.
        You are reporting an emergency situation to the police operator.
        Your user may be impaired, and you are speaking on their behalf.
        Answer the operator's questions accurately and concisely.
        Do not create additional scenarios or characters.
        If a specific address is asked, provide the coordinates of the location.
        """
        # Ensure the caller's response is correctly interpreted
        location = await get_location()
        location_str = ", ".join(location)  # Convert location tuple to a string
        caller_answer = await image_response(
            base64_image,
            history,
            (character_description + "location is: " + location_str),
        )

        # Check if caller_answer is None
        if caller_answer is None:
            logger.error("Received None from image_response, breaking the loop.")
            break

        history += caller_answer
        logger.info(f"Caller answer: {caller_answer}")

        conversation_line -= 1
    logger.info("Conversation with police operator completed.")


@timing
def text_response_json(client, system_prompt, prompts, format_structure, answer=bool):
    logger.info(f"Text response JSON initiated with system prompt: {system_prompt}")
    response = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "\n".join(
                    [f"Options for `{key}` are: in {answer} format." for key in prompts]
                ),
            },
        ],
        max_tokens=8192,
    )

    logger.info(f"Text response received: {response.choices[0].message.content}")
    json_response = format2json(response.choices[0].message.content, format_structure)
    return json_response.dict()


async def text_response(client, msg, system_msg):
    logger.info(f"Text response initiated with message: {msg}")
    response = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": msg,
            },
        ],
        max_tokens=8192,
    )
    words = response.choices[0].message.content

    logger.info(f"Text response received: {words}")
    return words  # Return the words directly


async def main():
    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(1)
    if not cap1.isOpened() or not cap2.isOpened():
        logger.error("Failed to open one or more cameras.")
        return

    logging.info("Cameras opened, giving time for shutter to adjust.")
    # time.sleep(4)
    logging.info("Ready to capture images.")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    iteration_count = 0
    max_iterations = 10  # Limit the number of iterations
    conversation_in_progress = False  # Flag to track conversation status

    while iteration_count < max_iterations:
        if not conversation_in_progress:
            logging.info("Starting image capture.")
            image_path = await capture_images(cap1, cap2)
            logging.info("Image capture complete.")
            if image_path:
                logging.info("Sending image to image check.")
                base64_image = encode_image(image_path)  # Ensure this returns a string
                logger.info("Starting image check for call decision.")
                call_decision = image_check(
                    base64_image,  # Pass the correct base64_image string
                    decision_prompts,
                    DecisionInfo,
                )
                if any(call_decision.values()):
                    logger.info(
                        "Emergency situation detected, proceeding with situation analysis."
                    )
                    # situation = image_check(
                    #     client, base64_image, situation_prompts, SituationInfo
                    # )
                    # situation["location"] = await get_location()
                    # conversation_in_progress = (
                    #     True  # Set flag to true when conversation starts
                    # )
                    await trigger_conversation(
                        client, base64_image, conversation_line=5
                    )
                    conversation_in_progress = (
                        False  # Reset flag after conversation ends
                    )
                else:
                    logger.info("No emergency situation detected.")

        iteration_count += 1
        await asyncio.sleep(1)  # Add a delay between iterations

    logger.info("Max iterations reached, stopping the program.")


if __name__ == "__main__":
    logger.info("Program started.")
    asyncio.run(main())
    logger.info("Program finished.")
