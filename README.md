# Assignment2
# MNIST-baselines
This repository trained and tested a simple CNN model on the * MNIST* dataset 
and proposed our own method to improve the performance or reduce the model size.
Additionally, it contains various state-of-the-art (SOTA) models for comparison.

## Requirements
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [PyTorch](http://pytorch.org/)

## Usage
### Create a model
1. Prepare `Model_Name.json` in [`config/`](./config)
2. Prepare `Model_Name.py` in [`models/`](./models)
3. Prepare `Trainer_Name.py` in [`trainers/`](./trainers) (optional) 
### Train a SOTA model
`python3 main.py --method Model_Name`
### Test a SOTA model
`python3 main.py --method Model_Name --test`
### Trained models
Available at [https://drive.google.com/drive/folders/1wQsRPMfqlsgEfL9VvEONtKhX-CMyKW4q?usp=share_link](https://drive.google.com/drive/folders/1wQsRPMfqlsgEfL9VvEONtKhX-CMyKW4q?usp=share_link)

### baseline
![baseline loss and accuracy.png](plot/reademePlots/img_2.png)
![confusion matrix.png](plot/reademePlots/img_3.png)
![well_classified.png](plot/reademePlots/img_4.png)
![miss_classified.png](plot/reademePlots/img_5.png)

### DOnet
![DOnet loss and accuracy.png](plot/reademePlots/img_8.png)
![DOnet confusion matrix.png](plot/reademePlots/img_9.png)

### DDOnet
![DDOnet loss and accuracy.png](plot/reademePlots/img_10.png)
![DDOnet confusion matrix.png](plot/reademePlots/img_11.png)

### Pruning-DDOnet
![Pruning-DDOnet loss and accuracy.png](plot/reademePlots/img_6.png)
![Pruning-DDOnet confusion matrix.png](plot/reademePlots/img_7.png)

### SOTA models
![accuracy.png](plot/reademePlots/img.png)
![loss.png](plot/reademePlots/img_1.png)