import os
import zipfile
import gzip
import shutil
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO
import json

INPUT_DIR = "/root/as/case"
OUTPUT_DIR = "/root/as/case/af3_predictions"

def extract_all_archives(input_dir, temp_dir):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith('.zip'):
                print(f"UNZIP: {file}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

            elif file.endswith('.gz'):
                print(f"UNGZ: {file}")
                output_file = os.path.join(temp_dir, file[:-3])
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

def cif_to_pdb(cif_file, pdb_file):
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file)

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)
        return True
    except Exception as e:
        print(f"Conversion failed {cif_file}: {str(e)}")
        return False


def process_af3_results(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temp_dir = os.path.join(input_dir, "temp_extracted")

    print("=" * 60)
    print("unzip...")
    print("=" * 60)
    extract_all_archives(input_dir, temp_dir)

    # 查找所有CIF文件
    print("\n" + "=" * 60)
    print("Convert the CIF file to the PDB format...")
    print("=" * 60)

    cif_files = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(input_dir):
        if 'temp_extracted' in root:
            continue
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(os.path.join(root, file))

    print(f"find {len(cif_files)} CIF")

    success_count = 0
    fail_count = 0

    for i, cif_file in enumerate(cif_files, 1):
        base_name = os.path.basename(cif_file).replace('.cif', '.pdb')
        pdb_file = os.path.join(output_dir, base_name)

        print(f"[{i}/{len(cif_files)}] Convert: {os.path.basename(cif_file)} -> {base_name}")

        if cif_to_pdb(cif_file, pdb_file):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("Clean up temporary files...")
    print("=" * 60)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("done")

    print("\n" + "=" * 60)
    print("finish!")
    print("=" * 60)
    print(f"Successful conversion: {success_count} 个文件")
    print(f"Failure conversion: {fail_count} 个文件")
    print(f"Output: {output_dir}")
    print("=" * 60)

    output_files = sorted(os.listdir(output_dir))
    if output_files:
        print(f"\n Generated PDB files (first 10):")
        for f in output_files[:10]:
            print(f"  - {f}")
        if len(output_files) > 10:
            print(f"  ... remaining {len(output_files) - 10} files")


if __name__ == "__main__":
    print("=" * 60)
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print("=" * 60)
    if not os.path.exists(INPUT_DIR):
        print(f"Error: The input directory does not exist: {INPUT_DIR}")
        exit(1)
    process_af3_results(INPUT_DIR, OUTPUT_DIR)