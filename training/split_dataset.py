import os
import random
import shutil
import yaml

def main():
    source_dir = "data/2_processed_dataset"
    base_out_dir = "data/dataset/swimmer_data"
    
    dirs = {
        "train_img": os.path.join(base_out_dir, "images/train"),
        "val_img": os.path.join(base_out_dir, "images/val"),
        "test_img": os.path.join(base_out_dir, "images/test"),
        "train_lbl": os.path.join(base_out_dir, "labels/train"),
        "val_lbl": os.path.join(base_out_dir, "labels/val"),
        "test_lbl": os.path.join(base_out_dir, "labels/test")
    }
    
    if os.path.exists(base_out_dir):
        shutil.rmtree(base_out_dir)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    valid_data = []
    
    for img in images:
        lbl = img.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(source_dir, lbl)):
            valid_data.append((img, lbl))
            
    if not valid_data:
        print("No valid image/label pairs found in", source_dir)
        return

    random.seed(42)
    random.shuffle(valid_data)
    
    total = len(valid_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = valid_data[:train_end]
    val_data = valid_data[train_end:val_end]
    test_data = valid_data[val_end:]

    def copy_data(data_list, img_dest, lbl_dest):
        for img_name, lbl_name in data_list:
            shutil.copy(os.path.join(source_dir, img_name), os.path.join(img_dest, img_name))
            shutil.copy(os.path.join(source_dir, lbl_name), os.path.join(lbl_dest, lbl_name))

    print(f"Total valid images found: {total}")
    print(f"Copying {len(train_data)} files to train...")
    copy_data(train_data, dirs["train_img"], dirs["train_lbl"])
    
    print(f"Copying {len(val_data)} files to val...")
    copy_data(val_data, dirs["val_img"], dirs["val_lbl"])
    
    print(f"Copying {len(test_data)} files to test...")
    copy_data(test_data, dirs["test_img"], dirs["test_lbl"])

    yaml_content = {
        "path": os.path.abspath(base_out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "swimmer"}
    }
    
    yaml_path = "data/dataset/swimmer_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nSuccess! Dataset split complete. Config saved to {yaml_path}")

if __name__ == "__main__":
    main()