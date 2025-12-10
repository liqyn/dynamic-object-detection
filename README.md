# dynamic-object-detection

#### MIT 6.8300 Spring 2025 Final Project

### Website: https://liqyn.github.io/dynamic-object-detection/

Using residuals between RAFT predicted optical flow and ego motion-induced geometric optical flow to detect moving objects from a mobile platform.

https://github.com/user-attachments/assets/d7710c1c-179d-4a12-83a8-948953c0368a

#### Dependencies

```
sudo apt-get install ffmpeg x264 libx264-dev
git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install . && cd ..
git submodule init
pip install -e .
```

#### Requirements

Tested on a system with an i9-14900HX, GeForce RTX 4090 Laptop GPU (16GB), 32GB RAM. May not work on systems
with less memory, even if `batch_size` is decreased.


### Learned

#### Demo

To run the evaluation data in our blog: 

Download the following rosbags:
[lewis data](https://drive.google.com/drive/folders/1Acb1YivDiLH8GGbfQibzC3x0WAQHrKoj?usp=sharing) (ROS2)
[ground truth](https://drive.google.com/drive/folders/1-Wm7sUjgPOP8kUV-A41tbCGWSMELjQqT?usp=sharing) (ROS2)

Download our [best checkpoint](https://drive.google.com/file/d/1HVLkAcEFT4VPR14o5PkYS5Y7levi7Iqz/view?usp=sharing) and save as ./Pytorch-UNet/checkpoints/best_model.pth

Download [SAM weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

```
export EVAL_BAG_PATH=/path/to/lewis_bag/
export EVAL_GT_BAG_PATH=/path/to/ground_truth/
export RAFT=/path/to/dynamic-object-detection/RAFT
export SAM=/path/to/SAM/weights/
export UNET=/path/to/dynamic-object-detection/Pytorch-UNet
python3 dynamic_object_detection/learned/offline_learned.py -p config/lewis_learned_eval.yaml
```

Videos and evaluation metrics will be saved to '/out/lewis_learned...'

### Non-Learned

#### Demo

To run the evaluation data in our blog, download the following rosbags:
[hamilton data](https://drive.google.com/file/d/1kZmhye7E61mLJtyaEFTm_aBValKu3VF5/view?usp=sharing) (ROS1), 
[ground truth](https://drive.google.com/drive/folders/1qGDTkIi9izoh6WXzFa-ODQmevd7g-kpr?usp=drive_link) (ROS2)

```
export BAG_PATH=/path/to/hamilton_data.bag
export RAFT=/path/to/dynamic-object-detection/RAFT/
python3 dynamic_object_detection/offline.py -p config/hamilton.yaml
```
Edit `config/hamilton.yaml` to experiment with different parameters.

*Note*: All operations assume undistorted images. Our data is already undistorted.

#### Evaluation

The code for evaluation metrics is in `eval/eval.ipynb`. Change the following lines in the second cell:
```
os.environ['BAG_PATH'] = os.path.expanduser('/path/to/hamilton_data.bag')
gt_bag = '~/path/to/gt_data/'
```
Then change the `runs` variable in the last cell to the list of runs that you want to evaluate (names of the pkl/yaml/mp4 outputs, without extension). Run the entire notebook. Outputs will be printed at the bottom.
