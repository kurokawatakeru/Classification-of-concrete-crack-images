import os
import shutil

# Change to the working directory
os.chdir('/Users/beetlea/aitd/FastAPI/re')

# List of files to delete
files_to_delete = [
    'vercel.json',
    'app_vercel.py',
    'railway.json',
    'convert_to_onnx.py'
]

# Delete individual files
for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted: {file}")
    else:
        print(f"File not found: {file}")

# Delete api directory
if os.path.exists('api'):
    shutil.rmtree('api')
    print("Deleted: api directory")
else:
    print("Directory not found: api")

print("Deletion complete!")