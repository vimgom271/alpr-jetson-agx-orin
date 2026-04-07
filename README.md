Overview

This project is a production-grade Automatic License Plate Recognition (ALPR) pipeline built on an NVIDIA Jetson AGX Orin, designed to detect license plates from live video streams in real time. The system runs fully on-device with no internet dependency, leveraging NVIDIA’s accelerated AI stack for high-performance edge inference.

Motivation

Many ALPR systems rely on cloud processing or multi-stage pipelines that introduce latency and complexity. The goal of this project was to build a fast, efficient, and fully edge-based system focused specifically on accurate license plate detection using modern deep learning techniques.

System Architecture

The pipeline is built on NVIDIA DeepStream 7.1, which manages the full video analytics workflow. Video is captured from a live camera or RTSP stream and processed frame-by-frame through a YOLOv8-based model trained specifically for license plate detection.

DeepStream handles decoding, batching, inference, and visualization, while its built-in object tracker assigns unique IDs to detected license plates, allowing consistent tracking across frames in real time.

Training

A YOLOv8-based model was trained specifically for license plate detection using Kaggle and Google Colab. The training dataset consisted of approximately 98,000 labeled images.

The trained model was exported to TensorRT FP16 format for optimized inference on the Jetson AGX Orin, significantly reducing latency and improving throughput compared to standard PyTorch execution.

Tech Stack
Hardware: NVIDIA Jetson AGX Orin
Video Analytics: NVIDIA DeepStream 7.1
Model: YOLOv8 (license plate detection)
Inference Optimization: TensorRT FP16
Bounding Box Parsing: DeepStream-Yolo custom parser
Python Bindings: DeepStream Python bindings (v1.1.11, Python 3.10)
Training: Kaggle, Google Colab (~98,000 images)
Key Challenges Solved
Compiled DeepStream Python bindings (pyds) from source due to lack of Python 3.10 support
Configured custom YOLO parser to correctly interpret license plate bounding boxes
Built a complete AI stack on ARM architecture, including PyTorch and torchvision from source
Optimized model deployment using TensorRT for real-time performance on edge hardware
What I Learned

This project required a deep understanding of real-time video analytics pipelines, model optimization for edge deployment, ARM-based system configuration, and debugging complex integration issues across CUDA, TensorRT, and DeepStream.

Download my model here


Author
Vimal Gomathisankar

