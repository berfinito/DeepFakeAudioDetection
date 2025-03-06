import os

custom_prefix = "wavefake"

# Get current folder and script name
current_folder = "./"
script_name = os.path.basename(__file__)

# Iterate through all files in the current folder
for file_name in os.listdir(current_folder):
    file_path = os.path.join(current_folder, file_name)

    # Skip folders and the script itself
    if not os.path.isfile(file_path) or file_name == script_name:
        continue

    # Define the new file name
    new_file_name = f"{custom_prefix}_{file_name}"
    new_file_path = os.path.join(current_folder, new_file_name)

    # Rename the file
    os.rename(file_path, new_file_path)
    print(f"Renamed: {file_name} → {new_file_name}")

print("\n All files renamed with the custom prefix!")
