# Create a workspace docker

Creates a docker image on top of a vllm image with your user as an active user, and a few QoL improvements  
You can safely mount your host home folder into this image without the root user messing up permissions  
Add ~/Projects/docker_bundle.tgz with your .ssh, .*rc and more before creation for these to end up in the resulting image
