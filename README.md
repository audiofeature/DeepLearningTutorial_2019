# Deep Learning Tutorial on Convolutional Neural Networks (CNN)
## using Python and Keras

(c) 2019 by Thomas Lidy - https://www.linkedin.com/in/thomaslidy

This is a <b>hands-on programming tutorial for Deep Learning algorithms</b> using Python 3.6 and [Keras 2.2](https://keras.io) on top of [Tensorflow](https://www.tensorflow.org). 

It was prepared in the course of Innovationslehrgang Data Science and Deep Learning.

## How to use this Tutorial

For the tutorials, we use [JuPyter notebook](https://jupyter.org), which allows to program and execute Python code interactively in the browser.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' contents in your browser by clicking on each *.ipynb file (e.g. viewing this directly on Github).

The tutorial will open in a new window your browser for viewing.

### Interactive Coding

If you want to follow the tutorials by actually executing the code on your computer, clone this repository from Github:

```
git clone https://github.com/audiofeature/DeepLearningTutorial_2019.git
```

or download as zip file from the same URL.


Then [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the tutorial's folder and start from the command line:

`jupyter notebook`

Your web browser will open showing a list of files. Start the tutorials by clicking on each chapter.


## Table of Contents

1. <b>[Car_recognition.ipynb](Car_recognition.ipynb)</b><br/>
   This tutorial shows how images are loaded into Python and classified binary into "cars" and "not cars" using
   a) a Fully Connected neural network and b) a Convolutional Neural Network.
   The Keras Sequential model is presented as well as several techniques on how to improve a model, including Batch Normalization, ReLU activation, Dropout and Data Augmentation.

2. <b>[Music\_speech\_classification.ipynb](Music_speech_classification.ipynb)</b><br/>
   This tutorial shows how music is distinguished from speech, loading audio files into Python and classifying them either into "music" or "speech" using different architectures and parameters of a Convolutional Neural Network. It also includes techniques such as Batch Normalization,
   ReLU Activation and Dropout.


## Installation of Pre-requisites

## Install Python

Note: On most Mac and Linux systems Python is already pre-installed. 

Check with `python --version` on the command line whether you have Python 3.6.x installed.

Otherwise install Python 3.6 from https://www.python.org/downloads/

## Install Python libraries:

### Mac OS

If you haven't installed Python PIP earlier, start a Terminal and do the following: 

```
xcode-select --install
easy_install pip 
```

### All OS (incl. Mac OS)

#### On the iDSDL Lab machine

As most of the prerequisites are installed on the Lab server already, only do:

```
pip install --user librosa
```

#### On all other computers

On command line or terminal execute the following: 

Note: If you do not use (ana)conda or virtualenv, and you want to install the libraries system-wide on Linux or Mac, use `sudo`.

```
cd DeepLearningTutorial_2019
pip install -r requirements.txt
```

Then try if you can start:
```
jupyter notebook
```

Note: If you have problems with installing some libraries on **Mac OS X**, check answers 2 and 3 [here](http://stackoverflow.com/questions/29485741/unable-to-upgrade-python-six-package-in-mac-osx-10-10-2).


## Optional for GPU computation

If you want to train your neural networks on your GPU for faster training, also install the following for NVidia GPUs (for other ones, follow the vendor guidelines):

Note: it is not mandatory to use a GPU for this tutorial.

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us) 
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (optional, for further speedup)

And also:
```
pip install tensorflow-gpu
```

For more details see e.g. https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25

### Check if installed correctly

To check if Keras and Tensorflow are installed correctly, type:

```
python -c "import keras"
```

If everything is installed correctly, it should print: `Using Tensorflow backend.`
 
If the GPU is configured correctly, it should also print `Using gpu device 0: GeForce GTX 1080 Ti` or similar.



# Source Credits

## Python libraries

The following helper Python libraries are used in these tutorials:

* `image_preprocessing.py`: by Thomas Lidy and Alexander Schindler
* `audiofile_read.py` and `rp_extract.py`: by Thomas Lidy and Alexander Schindler, taken from the [RP_extract](https://github.com/tuwien-musicir/rp_extract) git repository
* `wavio.py`: by Warren Weckesser

## Data Sources

The data sets we use in the tutorials are from the following sources: 

* Car Data Set:
Images of side views of cars for use in evaluating object detection algorithms. The images were collected at UIUC. Contains 1050 training images (550 car and 500 non-car images) and 170 single-scale test images, containing 200 cars at roughly the same scale as in the training images.
http://cogcomp.cs.illinois.edu/Data/Car/

* Music Speech Data Set:
by George Tzanetakis
Collected for the purposes of music/speech discrimination. Consists of 128 tracks, each 30 seconds long. Each class (music/speech) has 64 examples in 22050Hz Mono 16-bit WAV audio format.
http://marsyasweb.appspot.com/download/data_sets/