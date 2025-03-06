from pathlib import Path

custom_suffix = "real"

# Get current folder
current_folder = Path.cwd()
script_name = Path(__file__).name

# Rename all files except the script itself
for file_path in current_folder.iterdir():
    if file_path.is_file() and file_path.name != script_name:
        # Split filename and extension
        stem, ext = file_path.stem, file_path.suffix
        new_name = f"{stem}_{custom_suffix}{ext}"  # Append suffix before extension
        new_path = current_folder / new_name

        # Rename the file
        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} → {new_name}")

print("\n All files renamed with the custom suffix!")
