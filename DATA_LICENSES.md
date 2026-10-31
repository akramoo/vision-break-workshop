# Data and model licenses

**Educational only. Not for clinical use.** All images are published, anonymised research datasets.

## Children's chest X-rays: Kermany et al. 2018 (CC BY 4.0)

- Kermany, D. S., Goldbaum, M., Cai, W., et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell* 172(5): 1122-1131, 2018. https://doi.org/10.1016/j.cell.2018.02.010
- Data: Kermany, D., Zhang, K., Goldbaum, M. "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification." Mendeley Data, v3, 2018. https://doi.org/10.17632/rscbjbr9sj.3
- License: Creative Commons Attribution 4.0 (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
- Obtained through the Hugging Face mirror `hf-vision/chest-xray-pneumonia`.
- Used in `app/public/data/kermany/`: a seeded subset of the official train split (300 normal + 900 pneumonia, split into train and validation), the official 16-image validation set, and the full official test set (234 normal, 390 pneumonia). Changes: grayscale converted to 3 channels, resized to 224x224, re-encoded as JPEG (quality 85).
- Limitations: children aged 1 to 5 from one hospital (Guangzhou Women and Children's Medical Center). The Hugging Face copy has no patient IDs, so images of the same child may appear in more than one of our splits. The official test set contains 6 pixel-identical duplicate images.

## Adult chest X-rays: MedMNIST v2 / NIH ChestX-ray14 (CC0 1.0)

- Yang, J., Shi, R., Wei, D., et al. "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification." *Scientific Data* 10, 41, 2023. https://doi.org/10.1038/s41597-022-01721-8
- Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., Summers, R. M. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." CVPR 2017. https://arxiv.org/abs/1705.02315
- Source images: NIH Clinical Center, ChestX-ray14, released under CC0 1.0. https://nihcc.app.box.com/v/ChestXray-NIHCC
- ChestMNIST (224x224 version) from MedMNIST+, Zenodo record 10519652. https://zenodo.org/records/10519652
- Used in `app/public/data/adult/`: 200 images for the adult test set and 200 for the optional "mix in adult X-rays" fix, balanced 50/50 between pneumonia and no finding.
- Limitations: adult patients at another hospital. Labels were mined from radiology reports by text software, so some are wrong. Pneumonia is rare in this dataset (about 1.2%), which is why the samples are hand-balanced. The pneumonia-positive NIH images may also carry other findings, while the negatives are "no finding" (all 14 labels 0), so the app calls the adult negatives "no finding" rather than "normal" in the UI.

## Model: MobileNet v2 (Apache-2.0)

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018. https://arxiv.org/abs/1801.04381
- Weights: TensorFlow Hub `google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3`, trained on ImageNet, Apache License 2.0. Vendored in `app/public/models/mobilenet_v2/` (see `SOURCE.md` there for hashes).

## This repository

Code is MIT licensed (see `LICENSE`). The datasets and model keep their own licenses listed above.
