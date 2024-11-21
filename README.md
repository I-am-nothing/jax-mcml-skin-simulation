# jax-mcml-skin-simulation
MCML (Monte Carlo Multi-Layered) on skin simulation using JAX

### Requirements
* Cuda 12.4
* GPU RAM > 12GiB
* docker

### Quick Start
using docker
```bash
docker-compose up -d
```

right now you can use Jetbrains Gateway and select pycharm IDE to connect your container

or

```bash
cd root/work_dir/MCML_Jax
python main.py
```

data will be generated at root/work_dir/MCML_Jax/data


### GPU Speed
you can run same 100000000 photons at same time when using same layers

run 100000000 photons on 3090ti average forwards 1600 times/sec

### TODO
here still has cpu bottleneck, we need to figure it out
- [ ] design data loader to maximize photons to gpu ram
- [ ] use multiprocessing to load data

