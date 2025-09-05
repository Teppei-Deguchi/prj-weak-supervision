import os
import re
import csv

#output file
output_file = "../result/ddGf_value.csv"

#get result directory
base_dir = "calculation_result"

results = []

# search sub directories (e.g. L1A)
for subdir in os.listdir(base_dir):
    if re.match(r"^[A-Za-z]+\d+[A-Za-z]$", subdir):
        subdir_path = os.path.join(base_dir, subdir)
        file_path = os.path.join(subdir_path, "ddg_predictions.out") #get result file path (e.g. L1A/ddg_predictions.out)

        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("ddG:") and not line.startswith("ddG: description"):
                        parts = line.strip().split()
                        if len(parts) > 2:
                            try:
                                total_value = float(parts[2]) # read ddG:
                                results.append([subdir, total_value])
                            except Exception as e:
                                print(f"⚠️ Error processing {subdir}: {e}")
                        break

# output to csv
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["mutation", "ddGf_rosetta"])
    writer.writerows(results)

print(f"Result has been saved to '{output_file}' ")

