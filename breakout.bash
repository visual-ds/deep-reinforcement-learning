docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp my-tensorflow-gpu-py3 python -m cProfile -o stats/local-gpu-docker ./breakout_deep_q_learning.py
# docker run --gpus all -it -p 8888:8888 my-tensorflow-gpu-py3-jupyter
