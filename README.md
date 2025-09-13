# dynamic-object-detection

#### MIT 6.8300 Spring 2025 Final Project

### Website: https://andyli23.github.io/dynamic-object-detection/

Using residuals between RAFT predicted optical flow and ego motion-induced geometric optical flow to detect moving objects from a mobile platform.

https://github.com/user-attachments/assets/0100457d-c09d-4f91-9d25-a054e7a9ddd3

#### Dependencies

```
sudo apt-get install ffmpeg x264 libx264-dev
git clone https://github.com/mbpeterson70/robotdatapy && cd robotdatapy && pip install . && cd ..
pip install -e .
```

#### Requirements

Tested on a system with an i9-14900HX, GeForce RTX 4090 Laptop GPU (16GB), 32GB RAM. May not work on systems
with less memory, even if `batch_size` is decreased.

#### Demo

To run the evaluation data in our blog, download the following rosbags:
[hamilton data](https://drive.google.com/file/d/1kZmhye7E61mLJtyaEFTm_aBValKu3VF5/view?usp=sharing) (ROS1), 
[ground truth](https://drive.google.com/drive/folders/1qGDTkIi9izoh6WXzFa-ODQmevd7g-kpr?usp=drive_link) (ROS2)

```
export BAG_PATH=/path/to/KITTI/data
export RAFT=/path/to/dynamic-object-detection/RAFT/
python3 dynamic_object_detection/offline_KITTI.py -p config/kitti.yaml --base /path/to/KITTI/data --seq 0003
```
Edit `config/kitti.yaml` to experiment with different parameters.

*Note*: All operations assume undistorted images. Our data is already undistorted.

#### Evaluation

The code for evaluation metrics is in `eval/eval.ipynb`. Change the following lines in the second cell:
```
os.environ['BAG_PATH'] = os.path.expanduser('/path/to/hamilton_data.bag')
gt_bag = '~/path/to/gt_data/'
```
Then change the `runs` variable in the last cell to the list of runs that you want to evaluate (names of the pkl/yaml/mp4 outputs, without extension). Run the entire notebook. Outputs will be printed at the bottom.
