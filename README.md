# OGFL-PavementDamage
Orthogonal and Geometric Feature Learning for Pavement Damage Detection in Multi-Platform Imagery
Introduction
This repository provides the PyTorch implementation of our direction-aware framework for pavement damage detection. Our method integrates directional information across feature learning, loss optimization, and classification to achieve robust detection performance on multi-platform imagery (UAV aerial and vehicle-mounted cameras).
Key Features

Direction-aware Feature Enhancement (DFE): Orthogonal convolution branches (1×5, 5×1) with channel attention
Direction-aware IoU Loss: Aspect ratio-driven dynamic weighting for elongated crack detection
Adaptive Shape Classification: Geometry-based priors for damage type discrimination
Multi-platform Support: Robust performance across UAV and ground-based imagery
