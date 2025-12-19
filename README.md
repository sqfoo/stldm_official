# STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting

This is the official code implementation of the paper **STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting** submitted to TMLR.

## Setup Environment

Create a new conda environment:

```bash
conda create -n stldm python=3.9
conda activate stldm
```

Install related packages:

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Preparation

There are three radar reflectivity datasets being evaluated with **STLDM** and other baselines: SEVIR, HKO-7, and MeteoNet.

### SEVIR

For the SEVIR dataset, please refer to [https://github.com/amazon-science/earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer) for downloading the SEVIR dataset. Please make sure the downloaded files are stored in the following way:

```
data/
├─ SEVIR/
│   ├─ data/
│   │   └─ vil/
|   |      ├─ 2017/
|   |      ├─ 2018/
|   |      └─ 2019/
│   └─ CATALOG.csv
├─ ...
```

### HKO-7

For the HKO-7 dataset, please refer to [https://github.com/sxjscience/HKO-7](https://github.com/sxjscience/HKO-7) for downloading the HKO-7 dataset. Please make sure the downloaded files are stored in the following way:

```
data/
├─ HKO-7/
│   ├─ hko_data/
│   │   └─ mask_dat.npz
│   ├─ radarPNG/
│   │   ├─ 2009/
│   │   └─ ...
│   ├─ radarPNG_mask/
│   │   ├─ 2009/
│   │   └─ ...
│   └─ samplers/
├─ ...
```

The samplers files have been saved inside ```data/HKO-7/samplers/``` for you.

### MeteoNet

For the MeteoNet dataset, please refer to [https://github.com/DeminYu98/DiffCast](https://github.com/DeminYu98/DiffCast) to download the pre-processed h5 file, or you can follow the provided instruction to pre-process the raw dataset found on [the official MeteoNet website](https://meteofrance.github.io/meteonet/english/data/rain-radar/). Please make sure the file is named as ```meteo.h5``` and stored in the following way:

```
data/
├─ meteonet/
│   ├─ meteo.h5
├─ ...
```

### FYI 💡

If you really want to change the suggested way to save those data files above, remember to update the corresponding file directories as well in the following ways:

- The SEVIR dataset: ```SEVIR_ROOT_DIR``` in ```data/dutils.py```
- The HKO-7 dataset: ```__C.ROOT_DIR```, ```possible_hko_png_paths``` and ```possible_hko_mask_paths``` in ```nowcasting/config.py```
- The MeteoNet dataset: ```METEO_FILE_DIR``` in ```data/dutils.py```

## Training

You can train the **STLDM** with the script ```train.py``` with the following command:

``` bash
python train.py -d HKO7_5_20 --seq_len 5 --out_len 20 -m STLDM_HKO --type "3D" 
```

In particular, there are a few arguments to set:

- ```-d``` / ```--dataset``` : The dataset config found in ```data/config.py```. Please set the corresponding ```--seq_len``` (input sequence length) and ```--out_len``` (output sequence length) as well.
- ```-m``` / ```--model``` : The STLDM config found in ```stldm/__init__.py```
- ```--type```: is to specify whether is ```"3D"``` (**Spatiotemporal**) or ```"2S"``` (**Spatial**) Visual Enhancement.

## Evaluation and Sampling

### Evaluation 

For the evaluation of **STLDM**, we generate ten ensemble predictions of **STLDM** and evaluate them.

First, let's run the ensemble generation script, ```ens_gen.py``` to generate the ensemble prediction and save it as an npy file, with the following command:

```bash
python ens_gen.py -d HKO7_5_20 -m STLDM_HKO --type "3D" -f "model_checkpoint" --c_str 1.0 --e_id 0
```

Other than the arguments above, there are stll a few parameters to set:

- ```-f```: the relative/absolute path to **STLDM** checkpoint
- ```--c_str```: Classifier-Free Guidance strength, it is disabled when set to 0.0
- ```--e_id```: Represent the $e\_id$ th ensemble prediction, starting from 0

Then, we run the evaluation script, ```ens_eval.py``` to evaluate the generated ensemble predictions (labeled starting from 0) with the following command:

```bash
python ens_eval.py -d HKO7_5_20 --out_len 20 --e_file "filepath_{}.npy" --ens_no 10
```

Again, other than the arguments specified above, there are still a few parameters to set:

- ```--e_file```: The format of the ensemble predictions, replace the $e\_id$ by $\{ \}$. Make sure the predictions are labelled, starting from 0.
- ```--ens_no```: Total number of ensemble predictions, i.e., 10

### Sampling

Other than the evaluation process, we also provide a demo file, ```demo.ipynb``` to show you how to set up and call the **STLDM$** to generate samples for your side implementation. In this demo, we include three different configurations:

- **SpatioTemporal** Visual Enhancement with image size of *128*
- **Spatial** Visual Enhancement with image size of *128*
- **SpatioTemporal** Visual Enhancement with image size of *256*

You can download their corresponding modek checkpoints from [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/sqfoo_connect_ust_hk/IgATefXlByydRaKlqYnC3hIyAUNk5ftNZBXJz0yKa7d89yE?e=BLk0V3).

## Credits and Acknowledgment

We would like to thank these developers and credit their code.

- [FACL](https://github.com/argenycw/FACL)
- [OpenSTL](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/README.md)
- [DiffCast](https://github.com/DeminYu98/DiffCast)
- [denoising_diffusion_pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch)

## Citation

If you find this work helpful, please cite the following:

```bib
@article{foo2025stldm,
  author  = {Foo, Shi Quan and Wong, Chi-Ho and Gao, Zhihan and Yeung, Dit-Yan and Wong, Ka-Hing and Wong, Wai-Kin},
  title   = {STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=f4oJwXn3qg},
}
```
