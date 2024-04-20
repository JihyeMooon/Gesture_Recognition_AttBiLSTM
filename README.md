# Introduction
Hand gestures are one of the most natural ways that humans use to express their thoughts. They have potential applications in interfaces for Virtual Reality and Augmented Reality, as well as in sign language recognition. With the developments and successful applications of deep learning models in image analysis, such as image classification and object detection, it has become possible to recognize hand gestures using deep learning models. Dynamic hand gesture recognition, as a branch of the video classification problem, is challenging in many ways.

# Motivation
 * In Fall 2019, we investigated how we could optimize gesture recognition methods using attention mechanisms. 
 * We hypothesized that **attention mechanisms optimize the training of deep neural networks for continuous gesture recognition from video.**

# Proposed Framework
To validate this hypothesis, we have designed a framework consisting of 2D-CNN and BiLSTM with Attention (Figs. 1-2).

<img src="https://github.com/JihyeMooon/Gesture_Recognition_3D_AttentionBiLSTM/assets/112595759/e004ffd1-c7b4-4ccf-a17a-20eadc9311fe" width="40%" height="40%">

**Fig. 1. Overall Archtechture**

<img src="https://github.com/JihyeMooon/Gesture_Recognition_3D_AttentionBiLSTM/assets/112595759/975c5780-a19f-44e1-bfc6-0c19da7f1b63" width="70%" height="70%">

**Fig. 2. BiLSTM with Attention (Att.BiLSTM)**

In Fig. 2, BiLSTM includes X_t that represents embedding vector extracted from CNNs and W_t that represents Bi-LSTM weights, where t is time frame. 
Through X_t and W_t, we compute the attention-added biLSTM vectors using below Eqs. (1) and (2).

<img src="https://github.com/JihyeMooon/Gesture_Recognition_3D_AttentionBiLSTM/assets/112595759/be13f110-a095-480c-921d-01dd74e992a2" width="40%" height="40%"> 

In Eq. (1), W_t is the output vector set of BiLSTM. X_t is the embedding feature extracted by CNNs. Using Eq. (1), we obtain attention weights A_t. 
In Eq. (2), the Attention weight A_t is added to BiLSTM's weights. Then we obtain X'_t tha fed to the softmax layer.

# Dataset and model training
 * We conducted this experiment using the 20DB-Jester Dataset V1, which consists of 27 classes of gestures. A total of 118,562 videos were used for the training set, while 14,787 videos were allocated for the validation set.

 * We trained two models: Baseline (2D-CNN + BiLSTM without attention) and our AttBiLSTM (2D-CNN + BiLSTM with attention). As a 2D-CNN model, we selected the pre-trained 2D-ResNet18 model.

 * The hyperparameters for the Bi-LSTM were selected as follows: an embedding size of 128, one layer, and a hidden layer size of 256. All models were trained using a batch size of 64. The input size of image was 112x112x3. 

# Results 
 
**Table 1. Accuracy scores for 28 guesture classifications (At Epoch 50)**
|  | Taining Acc. | Validation Acc.  | 
|---|---|---|
| Baseline | 37.15 |  28.10 |
| AttBiLSTM  | 76.15  | 64.12  |  

![image](https://github.com/JihyeMooon/Gesture_Recognition_3D_AttentionBiLSTM/assets/112595759/0cf5fa90-2409-45c8-bce5-47e30bc4890e)

Fig. 3. Training and validation accuracies during 50 epochs 

We demonstrate that attention significantly improves model accuracies. 
**By simply adding the attention equation to the model, we show that model training was greatly optimized, which provides much better accuracies for the gesture recognition even with the simple structure!**

# Limitations

 * We used 2D CNNs even though the input data was 3D video sequence; This is because we had no enough resource to train our model over the large video dataset. 
 * However, we show attention-mechanism significantly imporves the performance of gesture recognition models.

# Code Usage
 * Edit opts.py per your data.
 * Run main_normal_attention.py for "AttBiLSTM".
 * Run main_non_attention.py for "Baseline".

# Acknowledgement
 * We created this code with [Dr. Chen](https://scholar.google.com/citations?user=0ZMklOIAAAAJ&hl=en) for "CSE 5095 Advances in Deep Learning" Class Project in Dec. 2019.
   * Class instructor is [Dr. Ding](https://scholar.google.com/citations?user=7hR0r_EAAAAJ&hl=en) -- Many thanks to his great teaching for the class!
 * We referred opts.py and some codes from [3D-Resnets-Pytorch](https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master) to build ResNet modules for the video frames. 
 * If you have any questions, please feel free to contact me at husky.jihye.moon@gmail.com!

