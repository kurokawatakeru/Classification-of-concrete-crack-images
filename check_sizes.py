import os
import glob

# Get all .pth files in the models directory
pth_files = glob.glob('models/*.pth')

print("Size of .pth files in the models directory:")
print("-" * 50)

for file_path in sorted(pth_files):
    # Get file size in bytes
    size_bytes = os.path.getsize(file_path)
    
    # Convert to human-readable format
    size_mb = size_bytes / (1024 * 1024)
    
    # Get just the filename
    filename = os.path.basename(file_path)
    
    print(f"{filename}: {size_bytes:,} bytes ({size_mb:.2f} MB)")