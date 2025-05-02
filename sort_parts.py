import os
from pathlib import Path

SRC = Path("parts_output")
DST = Path("parts_by_part_animal")

for dir in SRC.iterdir():
    if dir.is_dir():
      dir_name = dir.name
      phot_params = dir_name.split("_")
      animal_params = phot_params[0].split("-")
      animal_name = animal_params[0]
      animal_pose = animal_params[2]
      photo_size = phot_params[2]
      animal_texture = phot_params[4]
      photo_angle = phot_params[6]

      for file in dir.rglob("*.png"):
        part_name = file.name.split("_")[-1].removesuffix(".png")
        target_dir = DST/part_name/animal_name
        target_dir.mkdir(parents=True, exist_ok=True)
        link = target_dir/file.name
        if not link.exists():
          link.symlink_to(file.resolve())


