# Scene Recognition

## 1. Prepare dataset

1. Prepare dataset in the following format

   ```
   lbscene
   -- train
     -- escalator
       -- *.png
     -- other
       -- *.png
     -- staircase
       -- *.png
   -- val
     -- escalator
       -- *.png
     -- other
       -- *.png
     -- staircase
       -- *.png
   ```


## 2. Train model

1. Start training using transfer learning in Pytorch

   ```bash
   python3 train_placesCNN.py -a alexnet /media/shan/Data/lb-	dataset/classification/lbscene/ --resume 	checkpoint/alexnet_places365_python36.pth.tar --epochs 30 -b 256 --lr 0.0001
   ```

2. Visualize training using Tensorboard

   ```bash
   tensorboard --logdir logs/trainval/[start time]/
   ```

## 3. Convert model

1. Convert into ONNX

   ```bash
   python3 convert_onnx.py
   ```

2. Convert into IR model using model optimizer, preprocessing is handled in this optimization

   ```bash
   source /opt/intel/openvino/bin/setupvars.sh
   sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /home/shan/github/scene-recognition/model/alexnet.onnx --mean_values [123.675,116.28,103.53] --scale_values [58.395,57.12,57.375] --reverse_input_channels
   ```


## 4. Test performance

1. Clone my fork version of Open model zoo and checkout to `lb-scene` branch

   ```bash
   git clone https://github.com/songshan0321/open_model_zoo.git
   cd open_model_zoo
   git checkout lb-scene
   ```
2. Setup environment 

   ```
   cd tools/accuracy_checker
   python3 -m virtualenv -p python3.7 ./venv
   source venv/bin/activate
   python3 setup.py install
   ```

3. Convert dataset into cifar10 format

   ```bash
   cd scene-recognition/PNG-2-CIFAR10
   ./resize-script.sh
   python convert-images-to-cifar-format.py
   ```

4. Run the checker and check the result

   ```bash
   cd open_model_zoo/tools/accuracy_checker
   source venv/bin/activate
accuracy_check -c lb_scene/config.yml -m /path/to/IRmodel -s /path/to/source/data
   ```

   Output:
   
   ```bash
   Processing info:
   model: Alexnet
   launcher: dlsdk
   device: CPU
   dataset: sample_dataset
   OpenCV version: 4.1.2-openvino
   IE version: 2.1.custom_releases/2019/R3_cb6cad9663aea3d282e0e8b3e0bf359df665d5d0
   Loaded CPU plugin version: 2.1.30677
   174 objects processed in 2.228 seconds                                              
   accuracy: 90.80%
   ```

