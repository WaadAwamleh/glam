# GLAMpoints: Greedily Learned Accurate Match Points for Medical Image Registration

GLAMpoints is an innovative approach to feature detection and image registration, specifically tailored to handle the challenges presented by medical imaging data. Leveraging a semi-supervised learning model, GLAMpoints focuses on maximizing the correctness of keypoint matches across images, especially in scenarios involving low contrast and high noise, typical of medical imagery such as retinal images.

## Features

- **High Robustness**: Designed to be highly effective in medical imaging contexts where traditional methods falter.
- **Advanced Keypoint Detection**: Utilizes advanced machine learning techniques to detect and prioritize keypoints that are most likely to lead to accurate matches.
- **Reinforcement Learning Integration**: Employs a reward-based mechanism to enhance the selection process of keypoints, reinforcing decisions that lead to successful image registration.
- **Optimized for Medical Images**: Tailored to work exceptionally well with images that exhibit variability in illumination, contrast, and noise.

## Installation

To set up GLAMpoints on your system, follow these instructions:


- clone the repository onto your local directory.
- set up your environment with conda.
- install all the requirements.

```python

git clone https://github.com/WaadAwamleh/glam.git
cd glam
conda create --name glam python=3.9 
conda activate glam
pip install -r requirements.txt
```
You are now able to start your model training by executing this line in glam:

```python
python training_glam_detector.py --data_path /path/to/data/folder
```

Citing the GLAMpoints paper and their repository:
https://arxiv.org/pdf/1908.06812
https://gitlab.com/retinai_sandro/glampoints
Copyright (2019), RetinAI Medical AG.
This software for the training and application of Greedily Learned Matching keypoints is being made available for individual research use only. For any commercial use contact RetinAI Medical AG.
For further details on obtaining a commercial license, contact RetinAI Medical AG Office (sales@retinai.com).
RETINAI MEDICAL AG MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THIS SOFTWARE.
This license file must be retained with all copies of the software, including any modified or derivative versions.
