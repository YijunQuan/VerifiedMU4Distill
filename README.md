# PURGE

Implementation of **PURGE** from:  
> **"Efficient Verified Machine Unlearning For Distillation"**  
---
![413260304-d83b26f4-6661-461d-914f-3a98e5124747](https://github.com/user-attachments/assets/6e3e5248-b23f-41bf-ba1e-33b15a9f7462)


## Use
Training (Example: Using MNIST with 8 teacher constituents and 4 student constituents on a 10% subset of the training set for 120 epochs)
```bash
python train.py --dataset 'MNIST' --nt 8 --ns 4 --num_epochs 120 --percent 10
```
Performance Evaluation
```bash
python eval.py --dataset 'MNIST' --nt 8 --ns 4 --num_epochs 120 --percent 10
```

## Results
We show the plots for both the unlearning speed (Speed-up against baseline SISA) and predictive performance (CIFAR100 with 32 teacher constituents) included in the paper below:
### Unlearning Speed
![speed_up_32_tshards_100_percent_slice_1](https://github.com/user-attachments/assets/bebe8076-407b-4138-8b72-613adaa5a887)
### Image Classification on CIFAR-100
![multi_baseline_32_tshards_100_percent](https://github.com/user-attachments/assets/43ac105d-e94a-453c-ae7f-02e8be3090f5)

