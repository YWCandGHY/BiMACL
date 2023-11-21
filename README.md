# M2ACL: Multi-Level Motion Attention with Contrastive Learning for Few-shot Action Recognition

This repository contains the official implementation of the following paper:
> **M2ACL: Multi-Level Motion Attention with Contrastive Learning for Few-shot Action Recognition**<br>
> Xiamen University <br>
> In ICASSP 2024<br>

# TDSAM Visualization
In the original  paper, we highlighted that TDSAM is capable of correlating essential object information across adjacent frames. In this section, we present visualization results to illustrate this capability. Stay tuned for updates on the advancement of our research. If the paper is accepted, we plan to release both our code and visualization code simultaneously.

## Visualization Motivation
In the TDSAM module, we leverage motion features to compute the self-attention matrix, guiding the original feature map for enhancement. This motion attention integrates key object information from neighboring frames, naturally reflecting a correlation between the cross-attention features of two adjacent frames. Through experiments, we demonstrate that this ability to correlate key objects extends beyond just two neighboring frames. Furthermore, the motion features of multiple frames can establish correlations among key objects. This surprising finding suggests the potential of motion features to capture key object information not only within neighboring frames but also between support videos and query objects of the same class. Our latest paper validates this hypothesis, yielding even more promising results. Stay tuned to our GitHub account, where we will promptly share the code.

## Visualization implementation details
The structure of the motion feature in TDSAM is represented as (videos_num, frames_num, C, H, W), where "videos_num" denotes the number of videos, "frames_num" signifies the number of frames sampled per video, "C" represents the number of channels (C=256 in this case), and "H" and "W" correspond to the height and width of the feature map, both set to 7. Initially, we apply max pooling to resize the feature map to a 3×3 dimension. Subsequently, we utilize this 3×3 feature map to compute the self-attention  matrix and cross-attention matrix for each video. The resulting attention matrix takes on a shape of 9×9, and the weights within the attention matrix reflect the level of correlation between patch blocks in each frame.

## Visualizetion Results on UCF101
We present the visualization results for the self-attention matrix and cross-attention matrix pertaining to the Diving and Horse Riding categories, respectively. The self-attention matrix demonstrates TDSAM's capability to focus on essential object information within each frame through motion features. Similarly, the cross-attention matrix illustrates that the motion features within TDSAM can establish connections between the key object information of adjacent frames. This visualization serves as additional confirmation of the efficacy of our module, aligning with our underlying motivation: TDSAM effectively models spatio-temporal information, allowing for simultaneous consideration of the correspondence between key objects in neighboring frames.

<image src="https://github.com/YWCandGHY/BiMACL/assets/54340358/5eb8267b-3867-436a-a0db-c178258f3d96" width='45%'>

## Visualizetion Results on SSv2
We present the visualization results of the self-attention matrix and cross-attention matrix for the categories of Attaching and Pouring, respectively.

<image src="https://github.com/YWCandGHY/BiMACL/assets/54340358/c78b85a9-71e0-46ab-8092-1436de0c65f0" width='45%'>

## Visualizetion Results on Multi-frame Correlation
We further demonstrate through experiments that the ability to correlate key objects extends beyond two neighboring frames. Additionally, motion features between multiple frames can exhibit correlation with each other's key objects. The visualization results are presented below:

<image src="https://github.com/YWCandGHY/BiMACL/assets/54340358/07fcba26-5eb7-4d6b-a68b-dddccbfdd3b7" width='45%'>
  
We trust that the visualization outcomes contribute to advancing the functionality of the TDSAM module. Moreover, our forthcoming research endeavors will delve deeper into exploring the connection between motion features in support and query objects within the same task. The ensuing visualization results are outlined below:

