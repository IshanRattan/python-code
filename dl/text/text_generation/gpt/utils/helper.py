
import openai

from configs import config

def transcribe(audio: str) -> str:
    openai.api_key = config.openaiApiKey

    with open(audio, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file,
            model=config.sttVersion,
            response_format=config.sttRespFormat,
            language=config.sttLanguage)
    return transcript


def action(btn: str) -> str:
    if btn == config.audioBtnLabel:
        return 'Push to Stop️'
    else:
        return config.audioBtnLabel