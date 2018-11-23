# docker build -t dotabot .
# docker build -t dotabot . --build-arg user=***REMOVED*** --build-arg pwd=??? --build-arg guard=
# docker run -it --tmpfs "/root/Steam/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bot":size=1024M dotabot
# You need around 48Gb of disk space for this docker file, due to the dota install.

FROM ubuntu:16.04
MAINTAINER Tim Zaman <timbobel@gmail.com>

RUN apt-get -q update \
 && apt-get install -y \
    lib32gcc1 \
    curl \
    locales \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/steam \
 && curl --silent --show-error --fail --location --header "Accept: application/tar+gzip, application/gzip, application/octet-stream" -o - \
        https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz \
    | tar --no-same-owner -xz -C /opt/steam/

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

WORKDIR /root

ARG user
ARG pwd
ARG guard

RUN /opt/steam/steamcmd.sh +login ${user} ${pwd} ${guard} +app_update 570 +quit
# dota.sh can now be found in "/root/Steam/steamapps/common/dota 2 beta/game"

# Below installation is anything that's not related to the above dota install.
# This is to make sure the docker layer with the dota install caches nicely.

RUN apt-get -q update \
 && apt-get install -y --no-install-recommends \
    gcc \
    python3 python3-dev python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install pip setuptools --upgrade 

RUN pip3 install \
    flask \
    psutil