def make_dir_and_transfer_h5_data(dir_name: str, clean_dir: bool = True):
    """Makes a new directory and transfers h5 flow data files to the directory"""
    import os

    if clean_dir:
        os.system(f"rm -rf {dir_name}")
    os.system(f"mkdir {dir_name}")
    os.system(f"mv *.xmf *.h5 {dir_name}")
