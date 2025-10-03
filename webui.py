import json
import os
import sys
import threading
import time

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
from indextts.utils.device import select_device
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
default_port = int(os.getenv('GRADIO_PORT', '7860'))
parser.add_argument("--port", type=int, default=default_port, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
# Unified device selection (honors INDEXTTS_* env vars). Command-line --fp16 forces fp16 on top.
device, auto_fp16 = select_device(require_cuda=True)
final_fp16 = cmd_args.fp16 or auto_fp16
if cmd_args.fp16 and not auto_fp16:
    print(">> FP16 explicitly requested via --fp16")
print(f">> WebUI using device={device}, fp16={final_fp16}")
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=final_fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                device=device,
                )
# Supported languages list
LANGUAGES = {
    "English": "en_US",
}
EMO_CHOICES = [i18n("Same as speaker voice"),
                i18n("Use emotion reference audio"),
                i18n("Use emotion vector control"),
                i18n("Use emotion text description")]
EMO_CHOICES_BASE = EMO_CHOICES[:3]  # base options
EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES  # all options (including text description)

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             example.get("emo_text") is not None]
                             )

def normalize_emo_vec(emo_vec):
    # emotion factors for better user experience
    k_vec = [0.75,0.70,0.80,0.80,0.75,0.75,0.55,0.45]
    tmp = np.array(k_vec) * np.array(emo_vec)
    if np.sum(tmp) > 0.8:
        tmp = tmp * 0.8/ np.sum(tmp)
    return tmp.tolist()

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text, emo_random,
               # optional fields accepted from API clients (Voxta)
               language: str = "en",
               emotion: str = "",
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        # normalize emo_alpha for better user experience
        emo_weight = emo_weight * 0.8
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = normalize_emo_vec(vec)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    # Prefer explicit emo_text provided by caller; fall back to 'emotion' named label if present
    if emo_text == "":
        emo_text = emotion if emotion != "" else None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       language=language,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("Audio Generation")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("Speaker reference audio"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("Text"),key="input_text_single", placeholder=i18n("Enter target text"), info=f"{i18n('Model version')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("Synthesize"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("Result"), visible=True,key="output_audio")
        experimental_checkbox = gr.Checkbox(label=i18n("Show experimental features"), value=False)
        with gr.Accordion(i18n("Feature settings")):
            # Emotion control options
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_BASE,
                    type="index",
                    value=EMO_CHOICES_BASE[0],
                    label=i18n("Emotion control method"),
                )

        # Emotion reference audio group
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("Upload emotion reference audio"), type="filepath")

        # Emotion random sampling
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("Randomize emotion"), value=False)

        # Emotion vector control group
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("Happy"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("Angry"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("Sad"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("Afraid"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("Disgusted"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("Melancholic"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("Surprised"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("Calm"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(
                    label=i18n("Emotion description text"),
                    placeholder=i18n("Enter emotion description (leave blank to use the main text automatically)"),
                    value="",
                    info=i18n("Example: hurt, fearful, or excited"),
                )

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("Emotion weight"), minimum=0.0, maximum=1.0, value=0.8, step=0.01)

        with gr.Accordion(i18n("Advanced generation parameters"), open=False, visible=False) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**{i18n('GPT2 sampling settings')}** _{i18n('Parameters affect audio diversity and generation speed; see')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._"
                    )
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("Enable sampling"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(
                            label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1
                        )
                        length_penalty = gr.Number(
                            label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1
                        )
                    max_mel_tokens = gr.Slider(
                        label="max_mel_tokens",
                        value=1500,
                        minimum=50,
                        maximum=tts.cfg.gpt.max_mel_tokens,
                        step=10,
                        info=i18n("Maximum generated tokens; too small may truncate audio"),
                        key="max_mel_tokens",
                    )
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("Segmentation settings")}** _{i18n("Parameters affect audio quality and generation speed")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("Max tokens per segment"),
                            value=initial_value,
                            minimum=20,
                            maximum=tts.cfg.gpt.max_text_tokens,
                            step=2,
                            key="max_text_tokens_per_segment",
                            info=i18n(
                                "Recommended between 80 and 200. Larger values produce longer segments; smaller values produce shorter segments. Extremely small or large values may reduce audio quality."
                            ),
                        )
            with gr.Accordion(i18n("Preview segmentation"), open=True) as segments_settings:
                segments_preview = gr.Dataframe(
                    headers=[i18n("Index"), i18n("Segment text"), i18n("Token count")],
                    key="segments_preview",
                    wrap=True,
                )
            advanced_params = [
                do_sample,
                top_p,
                top_k,
                temperature,
                length_penalty,
                num_beams,
                repetition_penalty,
                max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        if len(example_cases) > 2:
            example_table = gr.Examples(
                examples=example_cases[:-2],
                examples_per_page=20,
                inputs=[
                    prompt_audio,
                    emo_control_method,
                    input_text_single,
                    emo_upload,
                    emo_weight,
                    emo_text,
                    vec1,
                    vec2,
                    vec3,
                    vec4,
                    vec5,
                    vec6,
                    vec7,
                    vec8,
                    experimental_checkbox,
                ],
            )
        elif len(example_cases) > 0:
            example_table = gr.Examples(
                examples=example_cases,
                examples_per_page=20,
                inputs=[
                    prompt_audio,
                    emo_control_method,
                    input_text_single,
                    emo_upload,
                    emo_weight,
                    emo_text,
                    vec1,
                    vec2,
                    vec3,
                    vec4,
                    vec5,
                    vec6,
                    vec7,
                    vec8,
                    experimental_checkbox,
                ],
            )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("Index"), i18n("Segment text"), i18n("Token count")])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_select(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def on_experimental_change(is_exp):
        # Toggle emotion control options
        # Note: the third returned value is not used
        if is_exp:
            return gr.update(choices=EMO_CHOICES_EXPERIMENTAL, value=EMO_CHOICES_EXPERIMENTAL[0]), gr.update(visible=True),gr.update(value=example_cases)
        else:
            return gr.update(choices=EMO_CHOICES_BASE, value=EMO_CHOICES_BASE[0]), gr.update(visible=False),gr.update(value=example_cases[:-2])

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox],
        outputs=[emo_control_method, advanced_settings_group,example_table.dataset]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
