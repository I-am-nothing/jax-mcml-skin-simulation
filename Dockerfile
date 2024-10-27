# Use CUDA 12.4.0 base image
FROM nvcr.io/nvidia/jax:24.04-py3

# Install SSH server and set root password for SSH access
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:XDDDDD' | chpasswd

# Allow root login and password authentication in SSH configuration
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Expose the SSH port
EXPOSE 22

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]