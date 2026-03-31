import os

def clean_dynamic_filenames(directory):
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    count = 0
    marker = "_jpg.rf."

    for filename in os.listdir(directory):
        if marker in filename:
            name_parts = filename.split(marker)
            prefix = name_parts[0]
            
            extension = filename.split('.')[-1]
            new_name = f"{prefix}.{extension}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                count += 1
            except FileExistsError:
                print(f"[Skip] {new_name} already exists.")
            except Exception as e:
                print(f"[Error] Failed to rename {filename}: {e}")
    
    print(f"\nDone! Successfully cleaned {count} files.")

folder_path = "data/1_raw_frames"

clean_dynamic_filenames(folder_path)
