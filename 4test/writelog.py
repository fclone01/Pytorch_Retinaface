import re
import csv

# Define the input and output file paths
log_file = 'retina_log.txt'  # Replace with the path to your log file
output_csv = 'output.csv'

# Open the log file and create a CSV writer
with open(log_file, 'r') as log, open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header row
    csv_writer.writerow(['epoch', 'Loc', 'Cla', 'Landm'])
    
    # Regular expression to match the required fields
    pattern = re.compile(r'Epoch:(\d+)/\d+.*?Loc: ([\d.]+) Cla: ([\d.]+) Landm: ([\d.]+)')
    
    # Process each line in the log file
    for line in log:
        match = pattern.search(line)
        if match:
            epoch = match.group(1)
            loc = match.group(2)
            cla = match.group(3)
            landm = match.group(4)
            # Write the matched values to the CSV
            csv_writer.writerow([epoch, loc, cla, landm])

print(f"Data has been written to {output_csv}")
