import csv
import os


def save(list_, filename, path_up=''):
    path = os.path.abspath(os.path.join(path_up, 'data', filename))
    with open(path, 'w', newline='') as my_csv:
        csv_writer = csv.writer(my_csv)
        csv_writer.writerows(list_)
