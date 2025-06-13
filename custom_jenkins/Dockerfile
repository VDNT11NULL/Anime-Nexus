FROM jenkins/jenkins:lts
#this line pulls the lts ie long term support version of jenkins

USER root
# switch to root user to install dependencies

RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean
# these cmds install the docker engine and its dependencies

RUN groupadd -f docker && \
    usermod -aG docker jenkins
# Add Jenkins user to the Docker group (create if it doesn't exist)

RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

USER jenkins