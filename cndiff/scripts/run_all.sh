#!/bin/bash

# 🚀 Run Conditional Diffusion with different configs

python3 -m scripts.run_cndiff --cfg ./etth1.yaml
python3 -m scripts.run_cndiff --cfg ./ettm1.yaml
python3 -m scripts.run_cndiff --cfg ./exchange.yaml
python3 -m scripts.run_cndiff --cfg ./weather.yaml
python3 -m scripts.run_cndiff --cfg ./electricity.yaml
