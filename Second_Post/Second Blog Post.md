# Let's Prove It !

Ecological scientists have collected abundant data: climate change, underground water movement, air condition, etc. With machine learning, we can learn knowledge from those data, which will help solve environmental problems.

![Environmental change with ML](./environment_change_with_ML.jpg)

Most of the time, machine learning means complex mathematics concepts and many lines of code, which creeps many people out. **But with Azure Machine Learning, your task will go easier.**

We thus bring to you, a comparison on the development procedure of a **Convolutional Neural Network (CNN)** - a ***Deep Learning Model*** (subset of Machine Learning) using two different sets of methodologies:

- ***Traditional Methodology*** - Importing **Python** libraries, **Dataset collection-preprocessing**, breaking our heads over it, etc.

- ***Azure ML Methodology*** - Easy tasks, Easier tasks, Way more Easier tasks, etc.
<br/><br/>

## Traditional Methodology 🧠
Below is a step-by-step approach on how we developed a **CNN Model**, which performs the task of **Object Recognition, Classification** and **Detection**.

We have used the **[cifar10](https://www.tensorflow.org/datasets/catalog/cifar10)** Dataset to Train the Model, which provides it with the capability to recognize and classify **10 Different Classes** of objects.

<div align="center">
	<img src="./Object Recognition GitHub Repo.png" height="100%" width="100%"/><br/><br/>
</div>

[This](https://github.com/Manab784/Object-Recognition-and-Classification-System) is the link to the **GitHub Repository** containing all the required code for building the Model.

### *Step 1*

Import the required libraries. 

```
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

import PIL.Image as Image
from cv2 import *
```

The most **Important** and **Noteworthy** libraries are - **[Keras](https://keras.io/), [cv2](https://pypi.org/project/opencv-python/) and [NumPy](https://numpy.org/).**


### *Step 2*

**Prepare, Pre-process** and **Filter** the **Dataset**. As the Dataset chosen here is relatively *small* compared to worldly data, this might not seem to be too much of a tedious task.

```
# Set random seed for purposes of reproducibility
seed = 21

# Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
```


### *Step 3*

Decide the **Dimension** of the Model (*Conv2D - a 2D Model*) and create various layers of the **Model**, namely **Input, Hidden** and **Output** layers. 

The **Input Size** of each layer is dependent on the size of the previous as well as input layers. The **Activation Function** must be selected according to the operation performed by the Model. Make sure to include a **Dropout Rate** at each layer to avoid **Overfitting** ( A Model which predicts 'too closely or exactly' to a particular set of data is said to be an Overfitted Model )

```
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))
```


### *Step 4*

Declare the number of **Epochs** you'd like to run on the Model and finally, **Compile** it !

*Don't forget to print out the **Accuracy** of your model. You'd want it to be as accurate as possible.*

Finally, **Save** your Model, for future use and integrations with other projects/applications you might build !

```
epochs = 25
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
numpy.random.seed(seed)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

model.save('CNN_ImageProcessing.h5')
```

***Phew ! That was so much work*** ‼

<div align="center">
	<img src="./Hard Work.png" height="100%" width="100%"/><br/><br/>
</div>

Now, let's look into how you'd perform the same set of tasks (way lesser and easier tasks) using **Azure ML** 🎉
<br/><br/>

## Easy (Azure ML) Methodology 🥂
Below is a step-by-step approach to one of the most **Difficult** ways of using the user-friendly **Azure ML** Service - via the **Command Line Interface (CLI)** , and we are confident, you will still find this easier than the previous procedure. 

***That's a bold statement to make ain't it ?***

### *Step 1*

Install the **Azure ML Extensions** for the **Azure CLI** by running the following command:

```
$ az extension add -n azure-cli-ml
```

You now have access to the Azure ML Command Line Tools via the CLI.


### *Step 2*

Create a new resource group for the **Azure ML Workspace**, by running the following command.

```
$ az group create -n ml -l southindia
$ az ml workspace create -w mldws -g ml
```

*Note: Remember to change the region to the **Azure Region** nearest to you. (Change 'southindia' to your desired region. Lookup this [Gist](https://gist.github.com/ausfestivus/04e55c7d80229069bf3bc75870630ec8) by ***ausfestivus*** if you aren't sure about the region closest to you.)*


### *Step 3*

Change to the **Working Directory** in your shell and export an Azure ML Workspace config to the disk.

```
$ az ml folder attach -w mlws -g ml -e mls
```

Finally, install the **python** extension to interact with Azure ML from within python by running the following command:

```
python3 -m pip install azure-cli azureml-sdk
```

Now, simply navigate to the **Machine Learning** workspace on the **Azure Portal** and open the **Azure Machine Learning** interface to use the Notebook Viewer provided.

To run your code on this **Environment**, click on **Compute** -> **Create Compute Instance**. When completed, an option to start a **Jupyter Notebook** pops up automatically.


### *Step 4*

**Surprise !!** 🌟 All the handwork's done above. All you need to do now is:

- Create a **Machine Learning** instance.
- Choose from **Trained Datasets**, **Automated ML**, **Designer Drag and Drop Model Deployment Interface** and loads more. (More about these on upcoming blogs !)
- Relax ‼️

<div align="center">
	<img src="./Easy.png" height="100%" width="100%"/><br/><br/>
</div>

***If we're doing the right math***, not only is it way easier to create a Model using Azure ML, but the lines of code reduce drastically too !

***We'd like to call an end to this exercise of comparison and we think, Azure ML is MILES AHEAD of the traditional method of ML Model Development.*** 

Let us know what you think ! Catch you in the next one 🎉

---
