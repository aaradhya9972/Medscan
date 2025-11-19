import os
import shutil
import random

# ==============================
# Config
# ==============================
BASE_PATH = r"A:\Python\Dataset"

BREAST_CANCER_SRC = os.path.join(BASE_PATH, "Breast_Cancer")   # unsorted
BREAST_CANCER_DST = os.path.join(BASE_PATH, "Breast_Cancer_Split")  # sorted output

DENTAL_CAVITY_PATH = os.path.join(BASE_PATH, "Dental_Cavity")  # has train/test, no val
# Pneumonia dataset is already well-structured, no changes needed.

# Ratios
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ==============================
# Helpers
# ==============================

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset(source_dir, dest_dir, val_split=0.15, test_split=0.15):
    """
    Splits dataset (Healthy, Cancer folders) into train/val/test.
    """
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        val_size = int(total * val_split)
        test_size = int(total * test_split)

        val_files = images[:val_size]
        test_files = images[val_size:val_size+test_size]
        train_files = images[val_size+test_size:]

        for subset, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            subset_dir = os.path.join(dest_dir, subset, cls)
            make_dirs(subset_dir)
            for f in files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(subset_dir, f))


def create_val_split(dataset_dir, val_split=0.15):
    """
    Creates a validation split for datasets that already have train/test but no val.
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    if not os.path.exists(train_dir):
        print(f"❌ No 'train' directory found in {dataset_dir}")
        return

    classes = os.listdir(train_dir)
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        val_size = int(len(images) * val_split)
        val_files = images[:val_size]

        cls_val_dir = os.path.join(val_dir, cls)
        make_dirs(cls_val_dir)

        for f in val_files:
            shutil.move(os.path.join(cls_path, f), os.path.join(cls_val_dir, f))


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    random.seed(42)

    print("\n🔹 Splitting Breast Cancer dataset...")
    split_dataset(BREAST_CANCER_SRC, BREAST_CANCER_DST, VAL_SPLIT, TEST_SPLIT)
    print(f"✅ Breast Cancer dataset saved at: {BREAST_CANCER_DST}")

    print("\n🔹 Creating val split for Dental Cavity dataset...")
    create_val_split(DENTAL_CAVITY_PATH, VAL_SPLIT)
    print(f"✅ Dental Cavity dataset updated in place: {DENTAL_CAVITY_PATH}")

    print("\nℹ️ Pneumonia dataset already prepared. Nothing to do!\n")
