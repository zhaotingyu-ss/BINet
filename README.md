<<<<<<< HEAD
# Rethinking-superpixel-segmentation-from-biologically-inspired-mechanisms.
Rethinking superpixel segmentation from biologically inspired mechanisms.（BINet）
=======
# RETHINKING SUPERPIXEL SEGMENTATION FROM BIOLOGICALLY INSPIRED MECHANISMS

This is a PyTorch implementation of the superpixel segmentation network introduced in Applied Soft Computing  paper (2024):


## Prerequisites
The training code was mainly developed and tested with python 3.8, PyTorch 1.8.0, CUDA 11.1, and Ubuntu 18.04.

During test, we make use of the component connection method in [SSN](https://github.com/NVlabs/ssn_superpixels) to enforce the connectivity 
in superpixels. The code has been included in ```/third_paty/cython```. To compile it:
 ```
cd third_party/cython/
python setup.py install --user
cd ../..
```

## Data preparation 
To generate training and test dataset, please first download the data from the original [BSDS500 dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz), 
and extract it to  ```<BSDS_DIR>```. Then, run 
```
cd data_preprocessing
python pre_process_bsd500.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
python pre_process_bsd500_ori_sz.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
cd ..
```
The code will generate three folders under the ```<DUMP_DIR>```, named as ```/train```, ```/val```, and ```/test```, and three ```.txt``` files 
record the absolute path of the images, named as ```train.txt```, ```val.txt```, and ```test.txt```.


## Training
Once the data is prepared, we should be able to train the model by running the following command:
```
python main.py --data=<DATA_DIR> --savepath=<PATH_TO_SAVE_CKPT> --workers 4 --input_img_height 208 --input_img_width 208 --print_freq 20 --gpu 0 --batch-size 16  --suffix '_myTrain' 
```
If you want to continue training from a ckpt, just add --pretrained=<PATH_TO_CKPT>. 

The training log can be viewed from the `tensorboard` session by running
```
tensorboard --logdir=<CKPT_LOG_DIR> --port=8888
```

If everything is set up properly, reasonable segmentation should be observed after 10 epochs.

## Testing
We provide test code to generate: 1) superpixel visualization and 2) the```.csv``` files  for evaluation. 

To test on BSDS500, run
```
python run_infer_bsds.py --data_dir=<DUMP_DIR> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```

To test on NYUv2, please first extract our pre-processed dataset from ```/nyu_test_set/nyu_preprocess_tst.tar.gz``` 
to ```<NYU_TEST>``` , or follow the [intruction on the superpixel benchmark](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/DATASETS.md)
 to generate the test dataset, and then run
```
python run_infer_nyu.py --data_dir=<NYU_TEST> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```

To test on KITTI, run
```
python run_infer_kitti.py --data_dir=<DUMP_DIR> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```

## Evaluation
We use the code from [superpixel benchmark](https://github.com/davidstutz/superpixel-benchmark) for superpixel evaluation. 
A detailed  [instruction](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/BUILDING.md) is available in the repository, please
 
(1) download the code and build it accordingly;

(2) edit the variables ```$SUPERPIXELS```, ```IMG_PATH``` and ```GT_PATH``` in ```/eval_spixel/my_eval.sh```,
example:

```
IMG_PATH='/home/name/superpixel/BINet/BSDS500/test'
GT_PATH='/home/name/superpixel/BINet/BSDS500/test/map_csv'

../../bin_eval_summary_cli /home/name/superpixel/BINet/eval/test_multiscale_enforce_connect/SPixelNet_nSpixel_${SUPERPIXEL}/map_csv $IMG_PATH $GT_PATH

```

(3)run 
```
cp /eval_spixel/my_eval.sh <path/to/the/benchmark>/examples/bash/
cd  <path/to/the/benchmark>/examples/
bash my_eval.sh
```

(4) run 
 ```
cp ./eval_spixel/my_eval.sh <path/to/the/benchmark>/examples/bash/
cd  <path/to/the/benchmark>/examples/

#the results will be saved to: /home/name/superpixel/BINet/eval/test_multiscale_enforce_connect/SPixelNet_nSpixel_54/map_csv/
bash my_eval.sh
 ```
several files should be generated in the ```map_csv``` folders in the corresponding test outputs including summary.txt, result.txt etc;

(5) cd BINet/eval_spixel for comparision
```
python plot_benchmark_curve.py --path '/home/name/superpixel/BINet/eval/test_multiscale_enforce_connect/' #will generate the similar curves in the paper
for single algorithm
```

>>>>>>> 719495c (v1)
