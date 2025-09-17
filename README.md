# Concrete Crack detection on CCIC Dataset

## Models were trained on a public available CCIC dataset

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
