# Pytorch-profile
We can use tensorboard and pytorch-profile to see the trace, GPU kernel and memory of LLM training.

Command of benchmark.py:


torchrun --standalone --nproc_per_node 8 benchmark.py --profile

torchrun --nnodes 2 --node_rank=0 --master_addr=10.20.1.170 --nproc_per_node 8  benchmark.py -p 3d -b 20  -s 10   --zero 2 --use_fp8 -g -x --profile

torchrun --nnodes 2 --node_rank=1 --master_addr=10.20.1.170 --nproc_per_node 8  benchmark.py -p 3d -b 20  -s 10   --zero 2 --use_fp8 -g -x --profile


colossalai run --nproc_per_node 8 --hostfile hostfile  benchmark.py -p 3d -b 20  -s 10 --zero 2 --use_fp8 -g -x --profile



## Use tensorboard:

tensorboard --logdir=/home/nvme-share/home/wangbinluo/ColossalAI/examples/language/llama/profile/

ssh -L 6006:localhost:6006 wangbinluo@211.102.192.100 -p 30956

## Result:
![img_v3_02ej_81920f5e-cb94-4000-a4d0-ad8ea1dcfd4g](https://github.com/user-attachments/assets/be8b59c2-050b-46d3-ad3c-cf39dfa49935)
![img_v3_02ek_f5a3bb5c-7859-450d-8009-1c0e32d07e4g](https://github.com/user-attachments/assets/3c55936f-60ab-4d4f-a356-076d54d65f74)
![img_v3_02ej_f48ed326-6f03-46fc-b3f8-1293be3c67cg](https://github.com/user-attachments/assets/007df6d8-321e-49c9-a7ee-061e7a142758)

Notice: Profile will reduce the tflops of training, as profile needs extra communication to record.
Like from 21.50 to 18.8.
