FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

RUN pip3 install --upgrade pip
RUN git clone https://github.com/NVIDIA/NeMo.git /NeMo
RUN cd /NeMo && git fetch origin 6b0f0886f933c6e21c92b2f1981f66993134be7e
RUN cd /NeMo && git reset --hard 6b0f0886f933c6e21c92b2f1981f66993134be7e
RUN cd /NeMo && pip install -e .

RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_common.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_lightning.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_nlp.txt

RUN pip3 install git+https://github.com/NVIDIA/NeMo-Run.git@5ed6128f9285e61cfee73d780b663c9d780f20c7
RUN pip3 install git+https://github.com/NVIDIA/Megatron-LM.git@9c11ab4ca24ead28c3c1e29f8904f8258d7543cb

RUN pip3 install transformers==4.48.1
RUN pip3 install datasets==3.2.0
RUN pip3 install huggingface_hub==0.27.1

RUN apt-get -y install --no-install-recommends libgomp1 build-essential
RUN apt-get -y install openssh-server
RUN apt-get -y install openssh-client
RUN mkdir -p -m0755 /var/run/sshd
RUN systemctl enable ssh

RUN ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa
RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN echo "Host *" >> /root/.ssh/config
RUN echo "   StrictHostKeyChecking no" >> /root/.ssh/config
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config 

RUN echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN pip install sagemaker-training

