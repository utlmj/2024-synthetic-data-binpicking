# generate_synthetic_dataset
This folder contains the code for generating a synthetic dataset using the provided models.  
Open `plane_rb_tex.blend`, which should have `render_dataset.py` preloaded. If not, load it, then click on run.  
Use `render_settings.txt` before running the python code to indicate the variable limits, model names and save paths!  
The `fillet.blend` files can be exchanged for other models (indicate this in `render_settings.txt`).  


The image generation can also be run from the command line:  
`>> blender -b plane_rb_tex.blend -P render_dataset.py`  
