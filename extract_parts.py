import os
from PIL import Image
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cv2
from tqdm import tqdm
import csv
import zipfile, io
from concurrent.futures import as_completed
import traceback

# Current Issues:
# Small parts are hard to select
# I didn't optimize much

# PARAMETERS
# Set the parameters for the extraction process
MIN_PIXEL_COUNT = 5
DENSITY_THRESHOLD = 0.3  # Minimum density of the mask to be considered valid
SAVE_PROGRESS = True # if not test
DELETE_PREVIOUS_OUTPUT = False
MAX_PROCESS = 999999
PROCESS_RANDOM = True  # If True and not TEST, will process images in random order
WORKERS = 4
TEST = False  # If True, will only process the first 10 files


output_dir = r"parts_output"
LOG_CSV = os.path.join(output_dir, r"processed.csv")
# seg_dir = r"D:\PN\chimera-painter-dataset.zip\chimera-art/segmap"
# img_dir = r"D:\PN\chimera-painter-dataset.zip\chimera-art/image"
zip_seg_prefix = "chimera-art/segmap/"
zip_img_prefix = "chimera-art/image/"
dataset_dir = r"D:\PN\chimera-painter-dataset.zip"
logs = False
logs2 = False
logs3 = False
logs4 = False
completed_files = set()

# Mapping of part names to their RGB colors in segmentation maps
PART_COLORS = {
    "Ground": (255, 255, 255),
    "Background": (0, 0, 0),
    "Head": (230, 25, 75),
    "Muzzle": (230, 190, 255),
    "Eyes": (250, 190, 190),
    "Nose": (70, 240, 240),
    "Mouth": (67, 99, 216),
    "Teeth": (128, 128, 0),
    "Ears": (170, 255, 195),
    "Horns": (117, 0, 0),
    "Neck": (60, 180, 75),
    "Wing": (240, 50, 230),
    "Torso Front": (145, 30, 180),
    "Torso Back": (0, 128, 128),
    "Upper Front Legs": (188, 246, 12),
    "Upper Back Legs": (245, 130, 49),
    "Lower Front Legs": (0, 0, 117),
    "Lower Back Legs": (128, 0, 0),
    "Foot Top": (128, 128, 128),
    "Foot Bottom": (255, 225, 25),
    "Claws": (154, 99, 36),
    "Tail": (255, 216, 177),
}

PART_TOPN = {
    "Ground": 0,
    "Background": 0,
    "Head": 1,
    "Muzzle": 2,
    "Eyes": 1,
    "Nose": 1,
    "Mouth": 2,
    "Teeth": 1,
    "Ears": 1,
    "Horns": 1,
    "Neck": 1,
    "Wing": 2,
    "Torso Front": 1,
    "Torso Back": 1,
    "Upper Front Legs": 2,
    "Upper Back Legs": 2,
    "Lower Front Legs": 2,
    "Lower Back Legs": 2,
    "Foot Top": 2,
    "Foot Bottom": 2,
    "Claws": 2,
    "Tail": 1,
}


# Returns parts found and not found in the segmentation map
def extract_parts(seg_path: str, img_path: str, output_dir: str) -> list:

    with zipfile.ZipFile(dataset_dir, "r") as zf:  # Maybe we should read in batches
        seg_data = zf.read(seg_path)
        img_data = zf.read(img_path)

    # Check if the file has already been processed
    file_name = os.path.basename(img_path).split(".")[0]
    if file_name in completed_files:
        if logs:
            print(f"File {file_name} already processed. Skipping.")
        return None

    # Load images
    seg_img = Image.open(io.BytesIO(seg_data)).convert("RGB")
    orig_img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    # Keep track of
    parts_found = []

    # for faster processing, convert to numpy array
    seg_array = np.array(seg_img)
    orig_array = np.array(orig_img)

    for part, _ in PART_COLORS.items():
        # Create a mask for the current part
        if part in parts_found:
            if logs3:
                print(f"{part} already found in {file_name}.")
            continue

        mask, pf = mask_select(seg_array, part, parts_found)

        if pf is None or mask is None:
            continue

        parts_found.extend(pf)

        # Create transparent image for the part
        part_array = np.zeros_like(orig_array)
        part_array[mask] = orig_array[mask]
        part_img = Image.fromarray(part_array, mode="RGBA")

        # Crop
        bbox = part_img.getbbox()
        if bbox:
            part_img = part_img.crop(bbox)

        # Save the part image
        part_dir = os.path.join(output_dir, file_name)
        os.makedirs(part_dir, exist_ok=True)
        part_img.save(os.path.join(part_dir, f"{file_name}_{part.replace(' ', '_')}.png"))

    # save text file with parts found
    parts_found = list(set(parts_found))  # Remove duplicates
    with open(os.path.join(output_dir, file_name, f"{file_name}_parts.txt"), "w") as f:
        for part in parts_found:
            f.write(f"{part}\n")

    # record file processed
    if SAVE_PROGRESS and not TEST:
        completed_files.add(file_name)
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([file_name, ",".join(parts_found)])

    if logs:
        print(f"Processed {file_name}: Found parts: {', '.join(parts_found)}")

    return parts_found


def mask_select(
    seg_array: np.ndarray, part, parts_found: list
) -> tuple:  # it might be better to combine images after they are proccessed
    # Part rules
    parts_to_select = []
    parts_to_select.append(part)

    if part == "Foot Top" or part == "Foot Bottom" or part == "Claws":
        parts_to_select = ["Foot Top", "Foot Bottom", "Claws"]
        parts_found.extend(parts_to_select)
    if part == "Head":
        parts_to_select = ["Head", "Muzzle", "Eyes", "Nose", "Mouth", "Teeth", "Ears"]
    if part == "Torso Front" or part == "Torso Back":
        parts_to_select = [
            "Torso Front",
            "Torso Back",
            "Upper Front Legs",
            "Upper Back Legs",
            "Lower Front Legs",
            "Lower Back Legs",
        ]
        parts_found.append("Torso Front")
        parts_found.append("Torso Back")
    if part == "Neck":
        parts_to_select = ["Neck", "Torso Front", "Torso Back"]
    if part == "Background" or part == "Ground":
        return None, None
    if part == "Muzzle" or part == "Nose" or part == "Mouth" or part == "Teeth":
        parts_to_select = ["Muzzle", "Nose", "Mouth", "Teeth"]
        parts_found.extend(parts_to_select)

    H, W = seg_array.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    valid_colors = [PART_COLORS[p] for p in parts_to_select]
    valid_vals = np.array(
        [(r << 16) | (g << 8) | b for (r, g, b) in valid_colors], dtype=np.uint32
    )

    r = seg_array[:, :, 0].astype(np.uint32)
    g = seg_array[:, :, 1].astype(np.uint32)
    b = seg_array[:, :, 2].astype(np.uint32)
    encoded = (r << 16) | (g << 8) | b

    mask = np.isin(encoded, valid_vals)

    if not mask.any():
        return None, None

    if not check_if_clean(mask):
        mask = clean_mask(mask, top_n=PART_TOPN[part])

    if not mask.any():
        return None, None

    parts_found.append(part) # before we were appending all selected parts
    return mask, parts_found


def check_if_clean(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
    mask_area = np.sum(mask)
    density = mask_area / bbox_area

    if density < DENSITY_THRESHOLD:
        return False
    return True


def clean_mask(mask: np.ndarray, top_n=1) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_clean = mask_open.astype(np.bool_)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_clean.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]

    if areas.size == 0 or areas.max() < MIN_PIXEL_COUNT:
        return np.zeros_like(mask)

    # get inices of the top N largest areas
    top_n_indices = np.argsort(areas)[-top_n:]
    top_n_labels = {i + 1 for i in top_n_indices}

    mask_final = np.zeros_like(mask, dtype=bool)
    for label in top_n_labels:
        mask_final |= labels == label

    return mask_final


def process_all(dataset_dir: str, output_dir: str, workers: int = 4):
    args_list = []
    total = 0

    with zipfile.ZipFile(dataset_dir, "r") as zf:
        all_files = zf.namelist()

        if PROCESS_RANDOM and not TEST:
            np.random.shuffle(all_files)

    path_pairs = {}

    for file in tqdm(all_files, desc="Finding files", unit="file"):
        if not file.endswith(".png"):
            continue

        if file.startswith(zip_seg_prefix):
            name = os.path.basename(file)
            seg_path = f"{zip_seg_prefix}{name}"
            img_path = f"{zip_img_prefix}{name}"
            if img_path in path_pairs:
                path_pairs[img_path] = seg_path
            elif seg_path in path_pairs:
                path_pairs[seg_path] = img_path
            else:
                path_pairs[seg_path] = ""
        elif file.startswith(zip_img_prefix):
            name = os.path.basename(file)
            seg_path = f"{zip_seg_prefix}{name}"
            img_path = f"{zip_img_prefix}{name}"
            if img_path in path_pairs:
                path_pairs[img_path] = seg_path
            elif seg_path in path_pairs:
                path_pairs[seg_path] = img_path
            else:
                path_pairs[img_path] = ""

    for path_1, path_2 in tqdm(
        path_pairs.items(), desc="Building arg list", unit="file"
    ):
        if path_1 and path_2:
            seg_path = ""
            img_path = ""

            if path_1[: len(zip_seg_prefix)] == zip_seg_prefix:
                seg_path = path_1
                img_path = path_2
            else:
                seg_path = path_2
                img_path = path_1

            args_list.append((seg_path, img_path, output_dir))

    if len(args_list) > MAX_PROCESS:
        args_list = args_list[:MAX_PROCESS]

    if TEST:
        args_list = args_list[:10]
        print(f"TEST mode: processing only {len(args_list)} images.")
        if logs3:
            print(args_list)

    total = len(args_list)

    # parallel processing
    print(f"Processing {total} images with {workers} workers.")
    bar = tqdm(total=total, desc="Processing images", unit="img", mininterval=0.5)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if not logs and not logs2 and not logs3:
            futures = []
            for seg, img, out in args_list:
                fut = executor.submit(extract_parts, seg, img, out)
                # whenever a task finishes, bump the bar by 1
                fut.add_done_callback(lambda _f: bar.update(1))
                futures.append(fut)

            for fut in futures:
                try:
                    fut.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] {e}")

    bar.close()


if __name__ == "__main__":
    # If output_dir already exists, delete it and everything inside
    if DELETE_PREVIOUS_OUTPUT:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # create log file if it doesn't exist
    if SAVE_PROGRESS:
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["File Name", "Parts Found"])
        else:
            with open(LOG_CSV, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    completed_files.add(row[0])

    process_all(dataset_dir, output_dir, WORKERS)
