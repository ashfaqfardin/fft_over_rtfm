import numpy as np
from pathlib import Path
from scipy.io import loadmat

TEST_LIST = Path("list/ucf-i3d-test.list")
MAT_DIR = Path("list/Matlab_formate")
OUT_FILE = Path("list/gt-ucf-local.npy")

def main():
    file_list = [line.strip() for line in TEST_LIST.read_text().splitlines() if line.strip()]
    mat_name_set = {p.name for p in MAT_DIR.glob("*.mat")}

    gt = []

    for file in file_list:
        file_path = Path(file)
        features = np.load(file_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        num_frame = features.shape[0] * 16
        name = file_path.name

        if name.startswith("Normal_"):
            gt.extend([0.0] * num_frame)
            continue

        split_file = name.split('_')[0]
        mat_file = split_file + '_x264.mat'

        if mat_file not in mat_name_set:
            raise FileNotFoundError(f"Missing annotation mat file for {name}: {mat_file}")

        annots = loadmat(MAT_DIR / mat_file)
        annots_idx = annots['Annotation_file']['Anno'].tolist()

        start_idx = int(annots_idx[0][0][0][0])
        end_idx = int(annots_idx[0][0][0][1])

        second_event = len(annots_idx[0][0]) == 2

        local_gt = []

        if not second_event:
            local_gt.extend([0.0] * start_idx)

            if (end_idx + 1) <= num_frame:
                local_gt.extend([1.0] * (end_idx - start_idx + 1))
                local_gt.extend([0.0] * (num_frame - (end_idx + 1)))
            else:
                local_gt.extend([1.0] * max(0, num_frame - start_idx))

        else:
            start_idx_2 = int(annots_idx[0][0][1][0])
            end_idx_2 = int(annots_idx[0][0][1][1])

            local_gt.extend([0.0] * start_idx)
            local_gt.extend([1.0] * (end_idx - start_idx + 1))
            local_gt.extend([0.0] * max(0, start_idx_2 - (end_idx + 1)))

            if (end_idx_2 + 1) <= num_frame:
                local_gt.extend([1.0] * (end_idx_2 - start_idx_2 + 1))
                local_gt.extend([0.0] * (num_frame - (end_idx_2 + 1)))
            else:
                local_gt.extend([1.0] * max(0, num_frame - start_idx_2))

        if len(local_gt) != num_frame:
            raise ValueError(
                f"GT length mismatch for {name}: expected {num_frame}, got {len(local_gt)}"
            )

        gt.extend(local_gt)

    gt = np.array(gt, dtype=float)
    np.save(OUT_FILE, gt)
    print("Saved:", OUT_FILE)
    print("GT total length:", len(gt))
    print("Num test videos:", len(file_list))

if __name__ == "__main__":
    main()
