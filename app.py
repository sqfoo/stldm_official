import torch
import numpy as np
import gradio as gr

from stldm import InferenceHub
from stldm.config import STLDM_HKO
from data.dutils import resize
from utilspp import gradio_visualize, gradio_gif

def nowcasting(file, cfg_str, ensemble_no):
    # Model Setup
    Forecastor = InferenceHub(
      model_config=STLDM_HKO, 
      cfg_str=cfg_str,
      model_type='HF'
    )

    # Data Preparation
    x = torch.tensor(np.load(file.name))
    if x.ndim not in (5, 4):
        raise ValueError("Please specify the input has the format of (T C H W)")
    
    if x.max() > 1:
        x = x / 255.0
    x = x.clamp(0, 1)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    x = resize(x, 128) # resize the data to 128 x 128
    
    if x.shape[1] < 5:
        raise ValueError("The input should have at least 5 frames for STLDM to predict")
    x = x[0, -5:]
    
    out = {}
    for i in range(ensemble_no):
      y_pred = Forecastor(input_x=x, include_mu=False)
      out[f'Ensemble {i+1}'] = torch.cat((x, y_pred), dim=0)

    figure = gradio_gif(out, len(out['Ensemble 1']))

    return figure

with gr.Blocks() as demo:
    gr.Markdown("# STLDM official demo for nowcasting")
    gr.Markdown("Please upload the radar sequences with **at least 5 frames** in the format of .npy file, and **STLDM** will predict the future 20 frames based on the past 5 frames.")
    gr.Markdown('Please refer to [paper](https://arxiv.org/abs/2512.21118) and [code](https://github.com/sqfoo/stldm_official) for more details about STLDM.')

    file_input = gr.File(label="Upload the input radar squences", file_types=[".npy"])
    cfg_str = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Classifier Free Guidance Scale")
    ensemble_no = gr.Slider(1, 10, value=2, step=1, label="How many ensemble predictions?")

    output = gr.Image(label="Nowcasting Results")
    btn = gr.Button("Forecast Now!")
    btn.click(fn=nowcasting, inputs=[file_input, cfg_str, ensemble_no], outputs=output)

demo.launch()