# nerf-pytorch-concise

## A PyTorch Re-Implementation

### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

<p align="center">
    <img src="imgs/pipeline.jpg"/>
</p>


A PyTorch re-implementation of [Neural Radiance Fields](http://tancik.com/nerf).

## Readability Matters!

Since the [original release](https://github.com/bmild/nerf), and the [concurrent pytorch implementation](https://github.com/yenchenlin/nerf-pytorch) wrote all the codes basically in just two files, `run_nerf.py` and `run_nerf_helper.py`, which makes it very hard to read, along with many unnecessary operations, I re-organize the whole code architecture as below, which is more concise and readable.

```shell
├── core
│   ├── datasets
│   │   ├── create_dataloader.py
│   │   ├── create_dataset.py
│   │   ├── create_rays.py
│   │   ├── load_blender.py
│   │   ├── load_deepvoxels.py
│   │   └── load_llff.py
│   ├── models
│   │   ├── create_model.py
│   │   ├── embedder.py
│   │   └── nerf.py
│   └── utils
│       ├── create_configs.py
│       ├── metrics.py
│       ├── test_nerf_utils.py
│       └── train_nerf_utils.py
├── run_nerf.py
```

## Modifications

Besides the re-organization, I re-implement some of the core functions in NeRF:

+ `core/datasets/dataset.py`: a unified dataset for blender, llff data type, with all `N*H*W` rays generated;
+ `core/datasets/dataloader.py`: a unified dataloader for both `no_batching=True/False`;
+ `core/datasets/create_rays.py`: a vectorized implementation of rays generation, `N*H*W` rays once;
+ `core/models/embedder.py`: a vectorized implementation of positional encoding;

With all the modifications describes above, this NeRF version is a little🫠 faster than the [concurrent pytorch implementation](https://github.com/yenchenlin/nerf-pytorch), which may illustrates the bottleneck may relies on the caching of data, according to the [re-implementaion](https://github.com/krrish94/nerf-pytorch) by [krrish94](https://github.com/krrish94).

## How To Run?

### Quick Start

Download data for two example datasets: `lego` and `fern`

```shell
bash download_example_data.sh
```

To train a low-res `lego` NeRF:

```shell
python run_nerf.py --config configs/lego.txt
```

To train a high-res `lego` NeRF:

```shell
python run_nerf.py --config configs/lego-official.txt
```

To train a low-res `fern` NeRF:

```shell
python run_nerf.py --config configs/fern.txt
```

## Sample Results from the Repo


### On Synthetic Data `lego`

<p align="center"> 
    <img src="imgs/lego-lowres.gif">
</p>


### On Real Data `llff`

<p align="center"> 
    <img src="imgs/fern-lowres.gif">
</p>

## Citation

```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

