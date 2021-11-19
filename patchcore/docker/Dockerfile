FROM pytorchlightning/pytorch_lightning:1.3.3-py3.8-torch1.8

# Install extras
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
CMD ["bash"]
WORKDIR /exp
