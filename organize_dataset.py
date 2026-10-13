import os
import shutil
import pandas as pd


# ==========================================
# 1. FOLDER PATHS
# ==========================================

RAW_FOLDER = "raw_dataset"
OUTPUT_FOLDER = "dataset"

METADATA_FILE = os.path.join(
    RAW_FOLDER,
    "HAM10000_metadata"
)


# ==========================================
# 2. READ METADATA
# ==========================================

print("Reading metadata...")

df = pd.read_csv(
    METADATA_FILE,
    sep=","
)

print("Metadata loaded successfully!")

print("Total records:", len(df))

print("\nColumns found:")
print(df.columns.tolist())


# ==========================================
# 3. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

print("\nCleaned columns:")
print(df.columns.tolist())


# ==========================================
# 4. DISEASE CLASSES
# ==========================================

classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ==========================================
# 5. CREATE DATASET FOLDERS
# ==========================================

print("\nCreating disease folders...")

for class_name in classes:

    folder_path = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )


print("Disease folders created!")


# ==========================================
# 6. FIND IMAGE
# ==========================================

def find_image(image_id):

    filename = str(image_id) + ".jpg"

    path1 = os.path.join(
        RAW_FOLDER,
        "HAM10000_images_part_1",
        filename
    )

    path2 = os.path.join(
        RAW_FOLDER,
        "HAM10000_images_part_2",
        filename
    )

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


# ==========================================
# 7. ORGANIZE IMAGES
# ==========================================

print("\nStarting image organization...")

count = 0
missing = 0


for _, row in df.iterrows():

    image_id = row["image_id"]

    disease = row["dx"]


    # Remove extra spaces

    image_id = str(image_id).strip()

    disease = str(disease).strip()


    # Check valid class

    if disease not in classes:

        continue


    # Find image

    source = find_image(image_id)


    if source is None:

        missing += 1

        continue


    # Destination

    destination = os.path.join(
        OUTPUT_FOLDER,
        disease,
        image_id + ".jpg"
    )


    # Copy image

    if not os.path.exists(destination):

        shutil.copy2(
            source,
            destination
        )

        count += 1


    # Progress

    if count % 500 == 0 and count != 0:

        print(
            "Images organized:",
            count
        )


# ==========================================
# 8. FINAL RESULT
# ==========================================

print("\n")
print("======================================")
print("DATASET ORGANIZATION COMPLETE")
print("======================================")

print(
    "Images copied:",
    count
)

print(
    "Images missing:",
    missing
)


# ==========================================
# 9. CLASS DISTRIBUTION
# ==========================================

print("\nClass distribution:")

print(
    df["dx"].value_counts()
)


# ==========================================
# 10. FINAL FOLDER CHECK
# ==========================================

print("\nFinal dataset folders:")

for class_name in classes:

    folder = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    number_of_images = len(
        [
            file
            for file in os.listdir(folder)
            if file.lower().endswith(".jpg")
        ]
    )

    print(
        class_name,
        "->",
        number_of_images,
        "images"
    )


print("\nDataset is ready!")