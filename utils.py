import torch
import os
def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
def check_class_balance(data_dir):
    """
    Check class balance in the dataset and print statistics.
    
    Args:
        data_dir (str): Path to the dataset directory with Train and Test folders
    """
    # Check balance in training set
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    
    # Get all class folders
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print("Class distribution in training set:")
    total_train = 0
    class_counts_train = {}
    
    for cls in train_classes:
        class_path = os.path.join(train_dir, cls)
        # Count only image files
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts_train[cls] = count
        total_train += count
        print(f"  {cls}: {count} images")
    
    # Calculate percentages
    print("\nPercentage distribution in training set:")
    for cls, count in class_counts_train.items():
        percentage = (count / total_train) * 100
        print(f"  {cls}: {percentage:.2f}%")
    
    # Check balance in test set
    test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    print("\nClass distribution in test set:")
    total_test = 0
    class_counts_test = {}
    
    for cls in test_classes:
        class_path = os.path.join(test_dir, cls)
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts_test[cls] = count
        total_test += count
        print(f"  {cls}: {count} images")
    
    # Calculate percentages
    print("\nPercentage distribution in test set:")
    for cls, count in class_counts_test.items():
        percentage = (count / total_test) * 100
        print(f"  {cls}: {percentage:.2f}%")
    
    # Calculate train/test ratio
    print("\nTrain/Test split ratio:")
    train_ratio = total_train / (total_train + total_test) * 100
    test_ratio = total_test / (total_train + total_test) * 100
    print(f"  Train: {train_ratio:.2f}%")
    print(f"  Test: {test_ratio:.2f}%")
    
    return class_counts_train, class_counts_test

# Usage
if __name__ == "__main__":
    from config import DATA_DIR
    train_counts, test_counts = check_class_balance(DATA_DIR)