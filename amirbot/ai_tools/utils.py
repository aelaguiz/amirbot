from sklearn.model_selection import train_test_split

def split_dataset(data, train_size, validate_size, test_size):
    """
    Splits the dataset into training, validation, and test sets based on the specified percentages.
    
    Parameters:
    - data: The complete dataset as a list of TrainingExamples.
    - train_size: The percentage of the dataset to allocate to the training set.
    - validate_size: The percentage of the dataset to allocate to the validation set.
    - test_size: The percentage of the dataset to allocate to the test set.
    
    Returns:
    - A tuple containing the training set, validation set, and test set.
    """
    
    # Ensure that the percentages add up to 1 (or 100%)
    if (train_size + validate_size + test_size) != 1:
        raise ValueError("The sum of train, validate, and test sizes must equal 1")
    
    # First split to separate out the training set
    initial_train_size = train_size + validate_size
    train_val_set, test_set = train_test_split(data, test_size=test_size, shuffle=True)
    
    # Adjust validate_size for the second split
    validate_size_adjusted = validate_size / initial_train_size
    
    # Second split to separate out the validation set from the training set
    train_set, validate_set = train_test_split(train_val_set, test_size=validate_size_adjusted, shuffle=True)
    
    return train_set, validate_set, test_set