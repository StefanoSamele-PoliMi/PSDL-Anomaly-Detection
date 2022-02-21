FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install extras
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
CMD ["bash"]
WORKDIR /exp
