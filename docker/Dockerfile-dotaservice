FROM dota:latest
MAINTAINER Tim Zaman <timbobel@gmail.com>

RUN apt-get -q update \
 && apt-get install -y \
    python3.7 \
    python3.7-distutils \
 && curl -s https://bootstrap.pypa.io/get-pip.py | python3.7

WORKDIR /root

RUN mkdir /root/dotaservice
COPY setup.py README.md /root/dotaservice/
COPY dotaservice /root/dotaservice/dotaservice/
RUN pip3.7 install --user -e /root/dotaservice/

ENTRYPOINT ["python3.7", "-m", "dotaservice"]
