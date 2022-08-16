FROM tensorflow/tensorflow:latest-jupyter
RUN pip3 install poetry
RUN poetry init --python 3.8.10 --name tf_mds --author claurin --no-interaction
RUN poetry add pandas scipy networkx
RUN poetry update