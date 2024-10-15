# Autotrain

## About the Project

A complete pipeline has been developed, covering everything from data preparation and transformation into the appropriate formats for image segmentation or object detection, to splitting the data into training and validation sets. The training is based on a pre-trained model with data that is similar or relevant to the same task. This allows the model to leverage knowledge gained from previous workloads, speeding up fine-tuning and improving both efficiency and accuracy when handling new data.

The motivation behind this project stems from one of the most repetitive and time-consuming tasks in Machine Learning, particularly in Computer Vision: data labeling. This process, which is often done manually, can take a lot of time. This project aims to reduce that burden by automating part of the work, such as creating bounding boxes in images, saving precious hours — even when it comes to identifying kittens in your photos.

---
### Semi-Supervised Learning

Semi-supervised learning is a technique that combines a small amount of labeled data with a large amount of unlabeled data to train computer vision models more efficiently. This approach is particularly useful when labeling large volumes of data is costly or time-consuming. During training, the model uses the labeled data to learn basic patterns and the unlabeled data to refine these representations, resulting in higher accuracy and better generalization in real-world scenarios.

---
## Limitations
The following project has some technical limitations:

- It uses the YOLO v8 architecture, which can be easily updated to newer versions. However, at the time of development, this model was the most consistent and reliable for generalization.
- The current task classes are limited to two: segmentation and object detection. Updating the project to support more types of data for automatic retraining is planned for future versions.
- The data split is internally set to 70% for training and 30% for testing. Future updates will allow users to manually input these values or use techniques to optimize the data splitting process.

## Installation

<details>
<summary>CLI</summary>

- Download the build for your Operating System from the "Releases" tab.

- Clone the repository:

```bash
cd <your dir>
git clone https://github.com/brain-facens/autotrain.git

cd <your dir>/autotrain

# Install the necessary libraries
pip install -r requirements.txt
```

Note: The build you downloaded from the "Releases" tab should be placed inside the "autotrain" folder that you cloned.

Running:

- To format the dataset for segmentation:
```bash
./autotrain format segmentation --input_dir <your image directory to be formatted> --output_positive_dir <your directory to store the images that the model labeled automatically> --output_negative_dir <your directory to store the images that the model couldn't label automatically> --model <pre-trained base model to be used for labeling>
```
- To format the dataset for object detection:
```bash
./autotrain format object_detection --input_dir <your image directory to be formatted> --output_positive_dir <your directory to store the images that the model labeled automatically> --output_negative_dir <your directory to store the images that the model couldn't label automatically> --model <pre-trained base model to be used for labeling>
```
- To split the dataset:
```bash
./autotrain split_dataset --output_positive_dir <your formatted directory with COCO format to be split>
```
- To train the new model:
```bash
./autotrain train --model <base model to be retrained> --dataset_yaml <.yaml with COCO format> --device <cpu or CUDA> --epochs <number of epochs> --imgsz <image size, multiple of 32>
```
</details>
<details>
<summary>Usage Help</summary>

Need help using the project? Use the following command to better understand all available commands:

```bash
./autotrain --help
``` 

Or if you're having trouble with a specific command:

```bash
./autotrain <command> --help
```

## TODO

- [] Create an option for the user to input the desired data split division
- [] Increase the activities that the package supports for automatic retraining
- [] Expand to more architectures and networks to be retrained, as well as open up for Data Science, NLP, and LLM activities

## Collaborators

We would like to thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/86479444?s=400&u=ec56facf58f543ca43d3754cfd70c934ee2e7926&v=4" width="100px;" alt="Foto do Eduardo Weber Maldaner no GitHub"/><br>
        <sub>
          <b>Eduardo Weber Maldaner</b>
        </sub>
      </a>
    </td>
  </tr>
</table>          
