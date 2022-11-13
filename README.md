# gpc-mineclip

Generative Pretrained Curricula for MineCraft

----

exploring language curricula in minecraft

## Installation

Make sure minedojo and mineclip are both installed on your system. This can be difficult. 
To verify, we have taken the validation scripts from `Minedojo` and `MineCLIP` and brought them into this repo.

Note, if you're using the docker container, you'll need to install mineclip yourself.
The docker container only comes with minedojo installed. 
To do so, run the following commands:

```bash
# download the docker image
docker pull minedojo/minedojo:latest

# spin up the image with access to resources
docker run --gpus all -it -d -p 8080:8080 minedojo/minedojo:latest tail -f /dev/null

# enter the image
docker exec -it <running_container_id> /bin/bash

# download this repo
git clone https://github.com/aadharna/gpc-mineclip.git

# install torch 1.13. We do this first because just intalling will fail because of some dependency sequencing issues
pip install torch

# install mineclip
pip install git+https://github.com/MineDojo/MineCLIP

# validate your installs!
python validate_minedojo.py
python validate_mineclip.py
python validate_mineagent.py

```
