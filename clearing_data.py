import os

def balance_encodings(encoding_directory, target_count):
    """
    Balance the number of encodings for each person to the target count.

    Parameters:
    - encoding_directory: The directory containing subdirectories with face encodings.
    - target_count: The desired number of encodings for each person.
    """
    for person_folder in os.listdir(encoding_directory):
        person_path = os.path.join(encoding_directory, person_folder)

        if os.path.isdir(person_path):
            print(f"Balancing encodings for {person_folder}")
            
            # List all encoding files for the person
            encoding_files = [f for f in os.listdir(person_path) if f.endswith('.pkl')]

            # Check if the person has more encodings than the target count
            if len(encoding_files) > target_count:
                # Remove excess encodings
                excess_count = len(encoding_files) - target_count
                excess_files = encoding_files[:excess_count]

                for file in excess_files:
                    file_path = os.path.join(person_path, file)
                    os.remove(file_path)

                print(f"Removed {excess_count} excess encodings.")
            else:
                print(f"No need to balance for {person_folder}")

balance_encodings('encodings', 900)
