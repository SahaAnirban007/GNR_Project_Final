# GNR Project — MCQ Visual Question Answering

## Directory Structure
The inference.py file is in a single flat directory (no subdirectories):

```
./
├── inference.py        # Main inference script
```

## Setup
Run setup.bash, which has been submitted to Moodle, to set up the environment, clone the repo, and activate it after.
```bash
bash setup.bash
conda activate gnr_project_env
```

## Inference
```bash
python inference.py --test_dir <absolute_path_to_test_dir>
```

The test_dir should contain:
- `images/`    — folder with .png images
- `test.csv`   — with columns: id, image_name

Output: `submission.csv` in the current directory
