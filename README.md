# jax-mcml-skin-simulation
MCML (Monte Carlo Multi-Layered) on skin simulation using JAX

### Requirements
* Cuda 12.4
* GPU RAM > 12GiB

### Quick Start
using docker
```bash
docker build -t jax24.04-py3-ssh .
```
start container
```bash
docker run --privileged --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 2222:22 -v .:/root/work_dir/MCML_Jax -d jax24.04-py3-ssh
```

right now you can use Jetbrains Gateway and select pycharm IDE to connect your container

or

```bash
cd root/work_dir/MCML_Jax
python main.py
```

data will be generated at root/work_dir/MCML_Jax/data


### TODO
here still has cpu bottleneck, we need to figure it out
- [ ] design data loader to maximize photons to gpu ram
- [ ] use multiprocessing to load data
