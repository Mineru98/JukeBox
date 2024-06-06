import os
import uuid
import torch
import warnings
import torchaudio
import gradio as gr
from einops import rearrange
from langdetect import detect
from functools import lru_cache
from datetime import datetime
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

warnings.filterwarnings("ignore")

os.environ["HF_TOKEN"] = "hf_xxxx"

class JukeBox:
    def __init__(self):
        self.output_path = "output"
        self.origin_lang = "kor_Hang"
        self.target_lang = "eng_Latn"
        self.selected_model = "NHNDQ/nllb-finetuned-ko2en"
        self.prompt = ""
        self.max_len = 1024
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    @lru_cache(maxsize=10)
    def lang_check(self, text: str):
        check_lang = detect(text)
        return 'ko' in check_lang
    
            
    def lang_check_then_convert(self, text: str):
        if self.lang_check(text):
            output = self.predict(text)
            if len(output) > 0:
                return output[0]["translation_text"]
            else:
                raise ValueError("Translation failed")
        else:
            return text
        
    
    def predict(self, text: str):
        return self.get_translator().predict(text)

    
    @lru_cache
    def get_sound_generator(self):
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        return model, model_config
    
    
    @lru_cache
    def get_translator(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.selected_model,
            cache_dir=os.path.join("models", "tokenizers")
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.selected_model,
            cache_dir=os.path.join("models")
        )

        gpu_count = torch.cuda.device_count()
        

        if gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(gpu_count)))
            torch.multiprocessing.set_start_method('spawn')
        self.model.to(self.device)

        translator = pipeline(
            'translation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            src_lang=self.origin_lang,
            tgt_lang=self.target_lang,
            max_length=self.max_len
        )
        
        return translator
 

    def generate_audio(self, prompt, sampler_type_dropdown, seconds_total=30, steps=100, cfg_scale=7, sigma_min_slider=0.3, sigma_max_slider=500):
        prompt = self.lang_check_then_convert(prompt)
        
        print(f"Prompt received: {prompt}")
        print(f"Settings: Duration={seconds_total}s, Steps={steps}, CFG Scale={cfg_scale}")

        device = self.device
        print(f"Using device: {device}")

        # Fetch the Hugging Face token from the environment variable
        hf_token = os.getenv('HF_TOKEN')
        print(f"Hugging Face token: {hf_token}")

        # Use pre-loaded model and configuration
        model, model_config = self.get_sound_generator()
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]

        print(f"Sample rate: {sample_rate}, Sample size: {sample_size}")

        model = model.to(device)
        print("Model moved to device.")

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total
        }]
        print(f"Conditioning: {conditioning}")

        # Generate stereo audio
        print("Generating audio...")
        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=sigma_min_slider,
            sigma_max=sigma_max_slider,
            sampler_type=sampler_type_dropdown, 
            device=device
        )
        print("Audio generated.")

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        print("Audio rearranged.")

        # Peak normalize, clip, convert to int16
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        max_length = sample_rate * seconds_total
        if output.shape[1] > max_length:
            output = output[:, :max_length]
            print(f"Audio trimmed to {seconds_total} seconds.")
        
        now = datetime.now()
        current_date = now.strftime("%Y_%m_%d")
        if not os.path.exists(f"{self.output_path}/{current_date}"):
            os.makedirs(f"{self.output_path}/{current_date}")
        unique_filename = f"{str(uuid.uuid4())}.mav"
        print(f"Saving audio to file: {unique_filename}")

        # Save to file
        torchaudio.save(unique_filename, output, sample_rate)
        print(f"Audio saved: {unique_filename}")
        
        # Return the path to the generated audio file
        return unique_filename


    def rendering(self):
        interface = gr.Interface(
            fn=self.generate_audio,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Enter your text prompt here"),
                gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde"),
                gr.Slider(0, 47, value=30, step=1, label="Duration in Seconds"),
                gr.Slider(10, 150, value=100, step=10, label="Number of Diffusion Steps"),
                gr.Slider(1, 15, value=7, step=0.1, label="CFG Scale"),        
                gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.3, label="Sigma min"),
                gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max"),
            ],
            outputs=gr.Audio(type="filepath", label="Generated Audio"),
            title="Stable Audio Generator",
            description="Generate variable-length stereo audio at 44.1kHz from text prompts using Stable Audio Open 1.0.",
            examples=[
                [
                    "Create a serene soundscape of a quiet beach at sunset.",  # Text prompt
                    "dpmpp-2m-sde",  # Sampler type
                    45,  # Duration in Seconds
                    100,  # Number of Diffusion Steps
                    10,  # CFG Scale
                    0.5,  # Sigma min
                    800  # Sigma max
                ],
                [
                    "clapping",  # Text prompt
                    "dpmpp-3m-sde",  # Sampler type
                    30,  # Duration in Seconds
                    100,  # Number of Diffusion Steps
                    7,  # CFG Scale
                    0.5,  # Sigma min
                    500  # Sigma max
                ],
                [
                    "Simulate a forest ambiance with birds chirping and wind rustling through the leaves.",  # Text prompt
                    "k-dpm-fast",  # Sampler type
                    60,  # Duration in Seconds
                    140,  # Number of Diffusion Steps
                    7.5,  # CFG Scale
                    0.3,  # Sigma min
                    700  # Sigma max
                ],
                [
                    "Recreate a gentle rainfall with distant thunder.",  # Text prompt
                    "dpmpp-3m-sde",  # Sampler type
                    35,  # Duration in Seconds
                    110,  # Number of Diffusion Steps
                    8,  # CFG Scale
                    0.1,  # Sigma min
                    500  # Sigma max
                ],
                [
                    "Imagine a jazz cafe environment with soft music and ambient chatter.",  # Text prompt
                    "k-lms",  # Sampler type
                    25,  # Duration in Seconds
                    90,  # Number of Diffusion Steps
                    6,  # CFG Scale
                    0.4,  # Sigma min
                    650  # Sigma max
                ],
                [
                    "Rock beat played in a treated studio, session drumming on an acoustic kit.",
                    "dpmpp-2m-sde",  # Sampler type
                    30,  # Duration in Seconds
                    100,  # Number of Diffusion Steps
                    7,  # CFG Scale
                    0.3,  # Sigma min
                    500  # Sigma max
                ]
            ]
        )
        interface.queue(max_size=10).launch()
                        


if __name__ == "__main__":
    instance = JukeBox()
    instance.rendering()