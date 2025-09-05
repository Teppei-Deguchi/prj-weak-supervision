import os
import csv

directory_path = 'ddg_result'

results = {}

for filename in os.listdir(directory_path):
    if filename.endswith(".out"):
        file_path = os.path.join(directory_path, filename)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            total_ddg = 0
            count = 0
            
            for line in lines:
                columns = line.split()
                if len(columns) >= 6 and columns[4] == "ddG":
                    try:
                        ddg_value = float(columns[5])  # get 6th column value if there is "ddg" at 5th column
                        total_ddg += ddg_value
                        count += 1
                    except (ValueError, IndexError):
                        pass
            
            # average ddG
            if count > 0:
                average_ddg = total_ddg / count
                results[filename[:-4]] = average_ddg

output_file = '../result/ddGb_value.csv'
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['mutation', 'ddGb_rosetta'])
    for xx, average in results.items():
        csv_writer.writerow([xx, average])

print(f'Result has been saved to {output_file} ')

