# Neural-Tree-Search
Neural Tree Search for active slam

## muax 😘
Muax provides help for using DeepMind's [mctx](https://github.com/deepmind/mctx) on gym-style environments. 

## Installation
Muax is required to be installed for this project:
```sh
pip install muax
```

Habitat is also needed:
```sh
pip install habitat-sim
pip install habitat-api
```

## Setup
The project requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-Tree-Search/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data

## Getting started
To run the code:
```python
python main.py
```

To use the trained model, specify the path to the model in `mini_args.py` `load_model`. For example,
```
load_model='/home/fangbowen/Neural-Tree-Search/tmp/models/nts4/model_best.npy'
```

## Results

![bst-nts4_eval_img2-1-2-30 2023-05-04 19_06_43](https://user-images.githubusercontent.com/104526323/236348674-c8bce570-650e-42f1-8df3-d9d2186ff1b4.gif)

![cmpnts](https://user-images.githubusercontent.com/104526323/236352277-df5c63ff-2792-4dfc-8053-d50b91ea7650.png)
