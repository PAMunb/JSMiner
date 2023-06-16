import glob
import csv
import os
import sys


cwd = sys.argv[1]

# Define o nome do arquivo de saída
output_file = 'results.csv'

delimiter = ','

# Lista todos os arquivos que correspondem ao padrão
csv_files = [f for f in os.listdir(cwd) if f.endswith('.csv')]

print(csv_files)


with open(output_file, 'w', newline='') as outfile:
    writer = None

    for csv_file in csv_files:
        with open(cwd+csv_file, 'r') as infile:
            reader = csv.reader(infile, delimiter=delimiter)

            has_header = csv.Sniffer().sniff(infile.readline())
            infile.seek(0)

            if writer is None:
                writer = csv.writer(outfile, delimiter=delimiter)

                if has_header:
                    header = next(reader)
                    header = [col.replace('-', '_') for col in header]
                    writer.writerow(header)

            if has_header:
                next(reader)
            for row in reader:
                if row[-1].endswith(','):
                    row[-1] = row[-1][:-1]
                writer.writerow(row)

print('Merge finished:', output_file)