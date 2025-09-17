# Concrete Crack detection on CCIC Dataset

## Models were trained on a public available CCIC dataset

This dataset comprises 40,000 RGB images with a resolution of $227 \times 227$ pixels. The dataset is divided into two categories: positive (crack) and negative (non-crack), each containing 20,000 images.

The CCIC dataset can be found in [CCIC Dataset]([https://direccion-del-enlace.com](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification))

## How to consume the models (from the dataset)

We used Keras version 2.6, with Tensorflow 2.6 as backend, Python 3.6, along with CUDA Toolkit version 11.0. 

You can evaluate a sample image by performing the following:

```python
python predict.py MODEL_NAME MODEL_PATH IMAGE_TEST_FILE
```
To have the script automatically load the .bin file, use the same model name and ensure that the .bin file (class list) is in the same directory as the model.

Example:
```python
python predict.py efficientNetB0 /efficientNetB0/efficientNetB0/images/test/crack/image1.jpg

Prediction:
image1,	0


```
