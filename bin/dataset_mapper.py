import csv

class DatasetMapper:
    def read(self, file_path, offset = 0, rows = -1):
        x = []
        y = []

        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            for row in csv_reader:
                if line_count - offset > rows and rows >= 0:
                    break
                if line_count > offset:
                    y.append(row[0])
                    x.append(row[1:])
                line_count += 1
        return x, y
