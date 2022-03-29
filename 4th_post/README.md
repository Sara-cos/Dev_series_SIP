# An low-code way to predict cotton prices—Azure ML Designer

*Have you ever been afraid of writing code, dealing with incorrect data, or studying ML algorithms? **Your work will be easy with Azure ML Designer!***

The blog will demonstrate how to implement the model using the designer, which provides the drag and drop options for convenient model building and evaluations.

## Azure ML Designer

Azure ML Designer offers a great approach with low code solutions that help the coders the non coders to work and build models with minimal knowledge. Its workflow is easy:

* Drag-and-drop datasets and components onto the canvas.
* Connect the components to create a pipeline draft.
* Submit a pipeline run using the compute resources in your Azure Machine Learning workspace.
* Convert your training pipelines to inference pipelines.
* Publish your pipelines to a REST pipeline endpoint to submit a new pipeline that runs with different parameters and datasets.
  * Publish a training pipeline to reuse a single pipeline to train multiple models while changing parameters and datasets.
  * Publish a batch inference pipeline to make predictions on new data by using a previously trained model.

![](./designer-workflow-diagram.png)

In the following sections, you will walk through the steps to build machine learning model in Designer.

## The dataset

The dataset is a crop with respective price values. We would try to build a model to predict the cotton prices from the given dataset.
Will look at the patterns and relations that it forms with other dataset.

The dataset is available at (https://www.kaggle.com/datasets/kianwee/agricultural-raw-material-prices-19902020). 

![](/Fouth_post/dataset.jpg)

But we want this to be challenging, leaving the dataset to you for implementing on your own.


## Let's see the simple implementation of a sample dataset in designer

The dataset is automobile price prediction, and its just a sample dataset. 

The completed path will look like this ...

![](/Fouth_post/imple.jpg)

The run overview are overview log and some data over the present pipeline run.

![](/Fouth_post/run_overview.jpg)

The evaluation and more details

![](/Fouth_post/2.jpg)
![](/Fouth_post/3.jpg)

The model has it own details and run logs in train tab in designer ...

![](/Fouth_post/5.jpg)

## Do some exercise

![](/Fouth_post/your-turn-point.gif)

Take the above dataset, work on the designer. Drag and drop the required tags and work around with the analysis with the visuals provided by the evaluation part.

One of the best implementation of such solutions can be powered by the Azure Cloud and with the advancement of prediction capabilities using machine learning and deep learning in the domain of AI.

## To Learn More

[Create a Regression Model with Azure Machine Learning designer - Learn | Microsoft Docs](https://docs.microsoft.com/en-us/learn/modules/create-regression-model-azure-machine-learning-designer/)

[Azure Machine Learning designer | Microsoft Azure](https://azure.microsoft.com/en-us/services/machine-learning/designer/)




