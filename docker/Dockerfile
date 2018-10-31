# docker build -t dotabot . --build-arg pwd=XXX --build-arg guard=XXX
FROM ubuntu:16.04
MAINTAINER Tim Zaman <timbobel@gmail.com>

RUN apt-get -q update \
 && apt-get -y install lib32gcc1 curl locales

RUN mkdir -p /opt/steam \
 && curl --silent --show-error --fail --location --header "Accept: application/tar+gzip, application/gzip, application/octet-stream" -o - \
        https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz \
    | tar --no-same-owner -xz -C /opt/steam/

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

# RUN /opt/steam/steamcmd.sh +login anonymous validate +quit
WORKDIR /root

ARG user=timbobel@hotmail.com
ARG pwd
ARG guard

# RUN /opt/steam/steamcmd.sh +login anonymous +app_update 570 +quit
# RUN /opt/steam/steamcmd.sh +login anonymous +force_install_dir ./dota2 +app_update 570 validate +quit
RUN /opt/steam/steamcmd.sh +login ${user} ${pwd} ${guard} +app_update 570 +quit

