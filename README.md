# Image Captioning with Visual Attention on Recipe-20k


### Before running this, please create a copy of reciepe_20k dataset on a seperate folder. Along with that, create a virtual environment. 
## To run this project : 
1. Install required libraries using : `pip install -r requirements.txt`. 
2. Run `wandb login` on command line. Make an account and login. This will save the loss plots and logs from each run. 
3. Change the paths in `configs/config.json`. The keys are accordingly :  
- `img_path` : Image path to reciepe20k dataset.
- `csv_path` : CSV path that stores all the info (`complete.csv`). 
- `json_path` :  Path to `reciepes_20k.json` file
- `pickle_save_path` : Path to a folder where pickle files will be saved to save time during a rerun.
- `aug_img_path` : Path to folder where the augmentations will be generated and saved. In case running it for the first time, set this as path and the key `load_augs_from_path` as `false`. In case of a rerun, set this to **"NA"** and `load_augs_from_path` as `true`. In case of no augmentations, set this as **"NA"** and `load_augs_from_path` as `false`.
- `load_augs_from_path` : See above.
- `generate_features` : For first time run, the code will pregenerate the feature maps from Inception-V3, and save them as numpy arrays in the same folder as images. In case of first run, set this to `true`, and in case of a rerun set it as `false`.
- `checkpoint_path` : Tensorflow saves models as checkpoints. This is Path to folder where it will save the checkpoints.
- `EPOCHS` : Number of Epochs to train the model for.
- `BATCH_SIZE` : Batch size for dataset.
- `BUFFER_SIZE` : Buffers are prefetched. Set this to 1000, in case of a normal experiment.
- `embed_dim`: 256, This works fine for our dataset.
- `units`: 512, This works fine for our dataset.
- `features_shape`: 2048, This works fine for our dataset.
- `atts_features_shape`: 64 This works fine for our dataset.
4. The outputs are logged to wandb. The plots are currently disabled, as they are creating windows to display plots.

