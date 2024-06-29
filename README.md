# GLAMpoints: Greedily Learned Accurate Match Points for Medical Image Registration

GLAMpoints is an innovative approach to feature detection and image registration, specifically tailored to handle the challenges presented by medical imaging data. Leveraging a semi-supervised learning model, GLAMpoints focuses on maximizing the correctness of keypoint matches across images, especially in scenarios involving low contrast and high noise, typical of medical imagery such as retinal images.

## Features

- **High Robustness**: Designed to be highly effective in medical imaging contexts where traditional methods falter.
- **Advanced Keypoint Detection**: Utilizes advanced machine learning techniques to detect and prioritize keypoints that are most likely to lead to accurate matches.
- **Reinforcement Learning Integration**: Employs a reward-based mechanism to enhance the selection process of keypoints, reinforcing decisions that lead to successful image registration.
- **Optimized for Medical Images**: Tailored to work exceptionally well with images that exhibit variability in illumination, contrast, and noise.

## Installation

To set up GLAMpoints on your system, follow these instructions:

```python
git clone https://github.com/yourusername/glampoints.git
cd glampoints
pip install -r requirements.txt
```
