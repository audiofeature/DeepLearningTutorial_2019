# Deep Learning Tutorial on Convolutional Neural Networks (CNN)
## using Python and Keras

(c) 2019 by Thomas Lidy - https://www.linkedin.com/in/thomaslidy

This is a <b>hands-on programming tutorial for Deep learning algorithms</b> using Python 3.6 and [Keras 2.2](https://keras.io) on top of [Tensorflow](https://www.tensorflow.org). 

It was prepared in the course of Innovationslehrgang Data Science and Deep Learning.

## How to use this Tutorial

For the tutorials, we use [JuPyter notebook](https://jupyter.org), which allows to program and execute Python code interactively in the browser.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' content in your browser, by clicking on each *.ipynb file (e.g. viewing this directly on Github).

The tutorial will open in a new window your browser for viewing.

### Interactive Coding

If you want to follow the tutorials by actually executing the code on your computer, clone this repository from Github:

```
git clone https://github.com/audiofeature/DeepLearningTutorial_2019.git
```

or download as zip file from the same URL.


Then  [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the tutorial's folder and start from the command line:

`jupyter notebook`

Your web browser will open showing a list of files. Start the tutorials by clicking on each chapter.


## Table of Contents

1. <b>[Car_recognition.ipynb](Car_recognition.ipynb)</b><br/>
   This tutorial shows how images are loaded into Python and classified binary into "cars" and "not cars" using
   a) a Fully Connected neural network and b) a Convolutional Neural Network.

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

If you haven't installed Python PIP earlier, start a Terminal and do the follwoing: 

```
xcode-select --install
easy_install pip 
```

### All OS (incl. Mac OS)

On command line or terminal execute the following: (on Windows leave out `sudo`)
```
sudo pip install jupyter
```

On the command line, try if you can start:
```
jupyter notebook
```

Install the remaining Python libraries needed:

Either by:

```
sudo pip install Keras>=1.2.0 Theano==0.8.2 scikit-learn>=0.17 pandas librosa
```

or, if you downloaded or cloned this repository, by:

```
cd DL_Tutorial
sudo pip install -r requirements.txt
```

If you have problems with installing some libraries on **Mac OS X**, check answers 2 and 3 [here](http://stackoverflow.com/questions/29485741/unable-to-upgrade-python-six-package-in-mac-osx-10-10-2).

## Configure Keras to use Theano

Since we use Theano as the Deep Learning computation backend, but Keras is configured to use TensorFlow by default, we have to change this in the `keras.json` configuration file, which is in the `.keras` folder of the user's HOME directory.

Copy the `keras.json` included in the `DL_Tutorial` to one of the following target directories (you can overwrite an existing file):

* Windows: `C:\Users\<user>\.keras\`
* Mac: `/Users/<user>/.keras`
* Linux: `/home/<user>/.keras`

An alternative is to change these 2 lines in your `keras.json` file to the following:
```
{
    "image_dim_ordering": "th",
    "backend": "theano"
}
```

See https://keras.io/backend/ for details or http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/ for a step by step guide.

### Optional for GPU computation

If you want to train your neural networks on your GPU, also install the following (not needed for the tutorials):

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (optional, for further speedup)

To permanently configure Keras/Theano to use the GPU place a file `.theanorc` in your home directory with the following content:

```
[global]
device = gpu
floatX = float32
mode=FAST_RUN
```

### Check if installed correctly

To check whether Python, Keras and Theano were installed correctly, do:

`
python test_keras.py
`

If everything is installed correctly, it should print `Using Theano backend.`<br/>
If the GPU is configured correctly, it should also print `Using gpu device 0: GeForce GTX 980 Ti` or similar.



# Source Credits

## Python libraries

The following helper Python libraries are used in these tutorials:

* `image_preprocessing.py`: by Thomas Lidy and Alexander Schindler
* `audiofile_read.py` and `rp_extract.py`: by Thomas Lidy and Alexander Schindler, taken from the [RP_extract](https://github.com/tuwien-musicir/rp_extract) git repository
* `wavio.py`: by Warren Weckesser

## Data Sources

The data sets we use in the tutorials are from the following sources: (a copy is included in this repository, so no need to download them)

* Car Data Set:
Images of side views of cars for use in evaluating object detection algorithms. The images were collected at UIUC. Contains 1050 training images (550 car and 500 non-car images) and 170 single-scale test images, containing 200 cars at roughly the same scale as in the training images.
http://cogcomp.cs.illinois.edu/Data/Car/

* Music Speech Data Set:
by George Tzanetakis
Collected for the purposes of music/speech discrimination. Consists of 128 tracks, each 30 seconds long. Each class (music/speech) has 64 examples in 22050Hz Mono 16-bit WAV audio format.
http://marsyasweb.appspot.com/download/data_sets/