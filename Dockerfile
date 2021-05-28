# Install base Python image
FROM python:3.7.10-buster

# Copy files to the container
COPY *.py /app/
COPY requirements.txt /app/
COPY model /app/

# Set working directory to previously added app directory
WORKDIR /app/

# Install dependencies
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

# Apex layer (apex allows for 16-bit training => x3-4 speedup)
# RUN git clone https://github.com/NVIDIA/apex \
# && cd apex \
# && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
# && cd ..

# Windows
# RUN git clone https://github.com/NVIDIA/apex
# RUN cd apex
# RUN python3 setup.py install
# RUN cd ..

# Expose the port uvicorn is running on
EXPOSE 80

# Run uvicorn server
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]