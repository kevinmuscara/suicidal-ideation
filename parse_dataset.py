import csv
import os
import argparse

def export_to_text_files(csv_file, positive_dir, negative_dir, one_class):
  if not os.path.exists(positive_dir):
    os.makedirs(positive_dir)
  
  if(one_class == "False"):
    if not os.path.exists(negative_dir):
      os.makedirs(negative_dir)

  with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for index, row in enumerate(reader):
      if index == 0:
        continue

      post = row[1]
      label = int(row[2])

      if(one_class == "True"):
        if label == 0:
          filename = f'{positive_dir}/{index}.txt'
      else:
        if label == 0:
          filename = f'{positive_dir}/{index}.txt'
        else:
          filename = f'{negative_dir}/{index}.txt'

      with open(filename, 'w', encoding='utf-8') as textfile:
        textfile.write(post)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export rows from CSV to text files based on labels.")
  parser.add_argument("--i", help="Path to the CSV input file")
  parser.add_argument("--oc", default=False, help="False for binary, True for one-class (default: False)")
  parser.add_argument("--c1", default="./struc/positive", help="Directory for positive label (default: ./struc/positive)")
  parser.add_argument("--c2", default="./struc/negative", help="Directory for negative label (default: ./struc/negative)")
  
  args = parser.parse_args()
  
  export_to_text_files(args.i, args.c1, args.c2, args.oc)