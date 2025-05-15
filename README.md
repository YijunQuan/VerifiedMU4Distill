# PURGE

Official Implementation of **PURGE** from:  
> **"Efficient Verified Machine Unlearning For Distillation"**  
> Paper Submitted to Neurips 2025
---
![413260304-d83b26f4-6661-461d-914f-3a98e5124747](https://github.com/user-attachments/assets/6e3e5248-b23f-41bf-ba1e-33b15a9f7462)

## 📋 Installation

```bash
git clone https://github.com/your_username/PURGE.git
cd PURGE
pip install -r requirements.txt
```

## 📖 Citation
If you use this code in your research, please cite:

```html
@article{quan2025efficient,
  title={Efficient Verified Machine Unlearning For Distillation},
  author={Quan, Yijun and Li, Zushu and Montana, Giovanni},
  journal={arXiv preprint arXiv:2503.22539},
  year={2025}
}
```

## Results
We show the plots for both the unlearning speed (Speed-up against baseline SISA) and predictive performance (CIFAR100 with 32 teacher constituents) included in the paper below:
### Unlearning Speed
![speed_up_32_tshards_100_percent_slice_1](https://github.com/user-attachments/assets/bebe8076-407b-4138-8b72-613adaa5a887)
### Image Classification on CIFAR-100
![multi_baseline_32_tshards_100_percent](https://github.com/user-attachments/assets/43ac105d-e94a-453c-ae7f-02e8be3090f5)

