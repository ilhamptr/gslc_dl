# gslc_dl

# Transfer Learning untuk Klasifikasi Ras Hewan

Implementasi klasifikasi citra pada subset **Oxford-IIIT Pet Dataset** menggunakan PyTorch dan transfer learning. Eksperimen membandingkan pelatihan dari nol, feature extraction, progressive fine-tuning, pengaruh data augmentation, serta arsitektur modern berorientasi efisiensi.

## Dataset

Dataset yang digunakan adalah `torchvision.datasets.OxfordIIITPet`.

Eksperimen menggunakan 10 kelas yang terdiri atas 5 ras kucing dan 5 ras anjing:

### Kucing
- Bengal
- Birman
- Ragdoll
- Russian Blue
- Siamese

### Anjing
- American Pit Bull Terrier
- Staffordshire Bull Terrier
- Beagle
- Basset Hound
- Pomeranian

Data dibagi secara stratified menjadi:

- 70% training set
- 15% validation set
- 15% test set

Seed eksperimen ditetapkan sebesar `6767`.

## Kondisi Eksperimen

| Kode | Kondisi |
|---|---|
| B1 | ResNet-18 dilatih dari nol |
| B2 | ResNet-18 feature extraction |
| B3 | ResNet-18 progressive fine-tuning |
| B4 | Perbandingan fine-tuning dengan dan tanpa data augmentation |
| B5a | EfficientNet-B0 dengan konfigurasi terbaik |
| B5b | MobileNetV3-Small dengan konfigurasi terbaik |

Semua eksperimen menggunakan:

- Input berukuran 224 × 224 piksel
- Batch size 32
- AdamW optimizer
- CrossEntropyLoss
- Early stopping dengan patience 5
- Normalisasi ImageNet
- Differential learning rate untuk fine-tuning

## Ringkasan Hasil

| Kondisi | Trainable Parameters | Validation Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| B1 — ResNet-18 from scratch | 11.181.642 | 46,64% | 0,4755 |
| B2 — Feature extraction | 5.130 | 89,26% | 0,9026 |
| B3 — Fine-tuning dengan augmentasi | 8.398.858 | 90,94% | 0,9295 |
| B4 — Fine-tuning tanpa augmentasi | 8.398.858 | 92,62% | 0,9286 |
| B5a — EfficientNet-B0 | 3.168.550 | 92,95% | 0,9562 |
| B5b — MobileNetV3-Small | 1.337.586 | 89,60% | 0,9296 |

EfficientNet-B0 menghasilkan Test Macro F1 tertinggi dengan jumlah parameter total yang lebih kecil daripada ResNet-18.

