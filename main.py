from modal import Stub, Image, web_endpoint
import os

from TTS.api import TTS
from fastapi.responses import FileResponse, HTTPException, Form
import langid
from typing import Annotated
from pydantic import BaseModel

# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

stub = Stub("xtts")

def download_models():
    tts.download_model("tts_models/multilingual/multi-dataset/xtts_v1")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")

image = (
    Image.debian_slim()
    .apt_install("ffmpeg", "wget")
    .pip_install(
        "TTS",
        "torch",
        "langid"
    )
    .run_function(download_models, gpu="any")
)

stub.image = image


class XTTSInputs(BaseModel):
    prompt: Annotated[str, Form()]
    language: Annotated[str, Form()] = "en"
    input_audio_file_path: Annotated[str | None, Form()] = None
    input_audio_raw: Annotated[bytes | None, Form()] = None

@stub.function(gpu="any")
@web_endpoint(method="POST")
def predict(
    input: XTTSInputs,
):
    if input.input_audio_file_path == None and input.input_audio_raw == None:
        raise HTTPException(status_code=400, detail="Please provide either input_audio_file_path or input_audio_raw")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
    tts.to("cuda")

    supported_languages=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn"]

    if input.language not in supported_languages:
        raise ValueError("Language you put in is not in is not in our Supported Languages, please choose from dropdown")
    
    language_predicted=langid.classify(prompt)[0].strip() # strip need as there is space at end!

    if language_predicted == "zh": 
        #we use zh-cn 
        language_predicted = "zh-cn"
    print(f"Detected language:{language_predicted}, Chosen language:{language}")

    if len(prompt)>10:
        #allow any language for short text as some may be common
        if language_predicted != language:
            #Please duplicate and remove this check if you really want this
            #Or auto-detector fails to identify language (which it can on pretty short text or mixed text)
            raise ValueError(f"Auto-Predicted Language in prompt (detected: {language_predicted}) does not match language you chose (chosen: {input.language}) , please choose correct language id. If you think this is incorrect please duplicate this space and modify code.")
    
    if len(input.prompt)<2:
        raise ValueError("Please give a longer prompt text")
        return (
                None,
                None,
            )
    if len(input.prompt)>200:
        raise ValueError("Text length limited to 200 characters for this demo, please try shorter text")
        return (
                None,
                None,
            )  
        
    try: 
        os.system("wget https://music-gen-exp-clean-shiner.s3.amazonaws.com/calabhammer.wav -O input.wav")
        audio_file_path="input.wav"
        tts.tts_to_file(
            text=input.prompt,
            file_path="output.wav",
            speaker_wav=audio_file_path,
            language=input.language,
        )
    except RuntimeError as e :
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(f"Exit due to: Unrecoverable exception caused by language:{input.language} prompt:{input.prompt}", flush=True)
            raise ValueError("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")
            if not DEVICE_ASSERT_DETECTED:
                DEVICE_ASSERT_DETECTED=1
                DEVICE_ASSERT_PROMPT=prompt
                DEVICE_ASSERT_LANG=language

            
            # HF Space specific.. This error is unrecoverable need to restart space 
            # api.restart_space(repo_id=repo_id)
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            raise e
    return FileResponse("output.wav", media_type="audio/wav")
    # else:
    #     gr.Warning("Please accept the Terms & Condition!")
    #     return (
    #             None,
    #             None,
    #         ) 
