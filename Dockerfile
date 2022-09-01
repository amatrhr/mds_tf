FROM tensorflow/tensorflow:nightly-jupyter

# ENV PATH="/root/.local/bin:$PATH"


WORKDIR /home/mdsuser

RUN pip3 install pandas scipy networkx autograd loguru sklearn openpyxl jupyterlab pytest 
# RUN useradd -r -g users mdsuser -d /home/mdsuser

# USER mdsuser
