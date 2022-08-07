FROM tensorflow/tensorflow:latest-jupyter
RUN pip3 install poetry
RUN poetry init --name tf_mds --author claurin --no-interaction
RUN poetry add pandas scipy networkx
RUN poetry update