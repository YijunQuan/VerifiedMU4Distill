# PURGE

**Paper accepted at NeurIPS 2025**

**Authors:** Yijun Quan, Zushu Li, Giovanni Montana

Implementation of **PURGE** from:
> **"Efficient Verified Machine Unlearning For Distillation"**

---

![413260304-d83b26f4-6661-461d-914f-3a98e5124747](https://github.com/user-attachments/assets/6e3e5248-b23f-41bf-ba1e-33b15a9f7462)

## Overview

This repository presents a novel method for **efficient verified machine unlearning in distillation settings** using SISA (Sharded, Isolated, Sliced, and Aggregated) slicing. Our approach significantly improves the efficiency of machine unlearning while maintaining high predictive accuracy through innovative teacher-student distillation with verified unlearning capabilities.

The method leverages SISA's sharding mechanism to enable fast, verified removal of training data from both teacher and student models, making it particularly suitable for applications requiring compliance with data protection regulations.

## Arguments

Key parameters for training and evaluation:

- **`nt`**: Number of teacher model constituents/shards
- **`ns`**: Number of student model constituents/shards  
- **`percent`**: Relative size of student training set to teacher training set (e.g., 10 means student uses 10% of teacher's training data)

## Quick Start

### Training
Example: Using MNIST with 8 teacher constituents and 4 student constituents where student uses 10% of teacher's training data, for 120 epochs:

```bash
python train.py --dataset 'MNIST' --nt 8 --ns 4 --num_epochs 120 --percent 10
```

### Performance Evaluation
```bash
python eval.py --dataset 'MNIST' --nt 8 --ns 4 --num_epochs 120 --percent 10
```

**Note on `percent` parameter:** This represents the relative size of the student training set compared to the teacher training set (NOT the removal percentage). For example, `--percent 10` means the student model is trained on 10% of the data that the teacher model uses.

## Results

Our method demonstrates significant improvements in both **unlearning speed** and **predictive accuracy** compared to baseline approaches. Key highlights:

- **Speedup**: Substantial acceleration in unlearning operations compared to baseline SISA implementations
- **Accuracy**: Maintained high predictive performance across benchmark datasets
- **Benchmarks**: Comprehensive evaluation on CIFAR-100 with various constituent configurations

We show the plots for both the unlearning speed (Speed-up against baseline SISA) and predictive performance (CIFAR-100 with 32 teacher constituents) included in the paper below:

### Unlearning Speed
![speed_up_32_tshards_100_percent_slice_1](https://github.com/user-attachments/assets/bebe8076-407b-4138-8b72-613adaa5a887)

### Image Classification on CIFAR-100
![multi_baseline_32_tshards_100_percent](https://github.com/user-attachments/assets/43ac105d-e94a-453c-ae7f-02e8be3090f5)

## Paper and Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{quan2025purge,
  title={Efficient Verified Machine Unlearning For Distillation},
  author={Quan, Yijun and Li, Zushu and Montana, Giovanni},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Contact

For questions or collaboration opportunities, please contact the lead author:

**Yijun Quan** - [GitHub](https://github.com/YijunQuan)

For technical issues, please open an issue in this repository.
