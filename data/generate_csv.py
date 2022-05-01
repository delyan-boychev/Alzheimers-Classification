import csv
import os
import enumClasses

directories = os.listdir("./Dataset")

f = open('./dataset.csv', 'w')
w = csv.writer(f)
for dir in directories:
    for file in os.listdir("./Dataset/"+dir):
        w.writerow([file, enumClasses.Classes[dir].value])
print("Dataset generated!")
