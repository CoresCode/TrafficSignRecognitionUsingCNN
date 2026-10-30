from PIL import Image
from pathlib import Path

# --------------------------------------------------
# Classes to extract
# --------------------------------------------------

classes = {
    "00001": "Speed_Limit_30",
    "00011": "Right_of_Way",
    "00013": "Yield",
    "00014": "Stop",
    "00017": "No_Entry",
    "00033": "Turn_Right_Ahead"
}

# --------------------------------------------------
# GTSRB Training Images Folder
# --------------------------------------------------

base = Path(r"D:\Projects 2\External drive\CNN model - Copy\dataset\raw\GTSRB\Final_Training\Images")

# --------------------------------------------------
# Output Folder
# --------------------------------------------------

output = Path("ppt_samples")
output.mkdir(exist_ok=True)

# --------------------------------------------------
# Extract One Sample Image From Each Class
# --------------------------------------------------

for folder, name in classes.items():

    folder_path = base / folder

    ppm_files = sorted(folder_path.glob("*.ppm"))

    if not ppm_files:
        print(f"No images found in {folder}")
        continue

    img = Image.open(ppm_files[0])

    img.save(output / f"{name}.png")

print("\nDone!")
print(f"Images saved in: {output.resolve()}")