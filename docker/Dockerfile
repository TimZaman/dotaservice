FROM ubuntu:18.04
MAINTAINER Tim Zaman <timbobel@gmail.com>

# Make sure to install steam and Dota first, as those docker layers will be biggest.
# Install, update & upgrade packages
# Create user for the server
# This also creates the home directory we later need
# Clean TMP, apt-get cache and other stuff to make the image smaller
RUN apt-get -q update \
 && apt-get install -y \
    lib32stdc++6 \
    lib32gcc1 \
    curl && \
    apt-get -y upgrade && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

# Create Directory for SteamCMD
# Download SteamCMD
# Extract and delete archive
RUN mkdir -p /root/steamcmd \
 && cd /root/steamcmd \
 && curl -o steamcmd_linux.tar.gz "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz" \
 && tar zxf steamcmd_linux.tar.gz \
 && rm steamcmd_linux.tar.gz

ARG user
ARG pwd
ARG guard
RUN /root/steamcmd/steamcmd.sh +login ${user} ${pwd} ${guard} +app_update 570 +quit

RUN apt-get -q update \
 && apt-get install -y \
    python3.7 \
    python3.7-distutils \
 && curl -s https://bootstrap.pypa.io/get-pip.py | python3.7

WORKDIR /root

CMD mkdir /root/dotaservice
COPY setup.py README.md /root/dotaservice/
COPY dotaservice /root/dotaservice/dotaservice/
RUN pip3.7 install --user -e /root/dotaservice/

ENTRYPOINT ["python3.7", "-m", "dotaservice"]