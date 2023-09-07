<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/swinir.png" alt="Algorithm icon">
  <h1 align="center">infer_swinir_super_resolution</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/swinir_super_resolution">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/swinir_super_resolution">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/swinir_super_resolution/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/swinir_super_resolution.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run SwinIR super resolution. This plugin can enlarge an image by a factor 4 each side.

More than a simple linear interpolation, this plugin can add details while upscaling.

| ![Low res cat](https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/cat.jpeg "Before")  |
|:-------------------------------------------------------------------------------------------------------------------------|
| *Original image*                                                                                                         |

| ![High res cat](https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/cat_x4.jpeg "After") |
|:---------------------------------------------------------------------------------------------------------------------------|
| *Output image*                                                                                                             |

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Initialize the workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_swnir_super_resolution", auto_connect=True)
   
# Run on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/cat.jpeg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).


## :pencil: Set algorithm parameters

- **use_gan** (bool) - Default True: If True, algorithm will use GAN method to upscale image, else will use PSNR method.    
- **large_model** (bool) - Default False: If True, algorithm will use the large model, else will use medium model.
- **cuda** (bool) - Default True: Run with cuda or cpu.
- **tile** (int) - Default 256: Size of tile. Instead of passing whole image to the deep learning model, which consumes 
a lot of memory, model is fed with square tiles of fixed size one by one.
- **overlap_ratio** (float) - Default 0.1: Overlap between tiles in percentage. Overlapping tiles then blending the 
results lead to a smoother image. Set it to 0 to have no overlap like in the original repo. 1,0 is max overlap.

**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_swinir_super_resolution", auto_connect=True)

algo.set_parameters({
    "use_gan": "True",
    "large_model": "False",
    "cuda": "True",
    "tile": "256",
    "overlap_ratio": "0.1"
})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/cat.jpeg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())

```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_swinir_super_resolution", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_swinir_super_resolution/main/icons/cat.jpeg")

# Iterate over outputs
for output in algo.get_outputs()
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

[optional]