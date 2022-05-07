import csv
import os
import enumClasses as enumClasses
import zipfile
import shutil

dataset_archive = "./raw_dataset.zip"
folder_extracted = "./Dataset"
folder_images = "./Images"

with zipfile.ZipFile(dataset_archive, 'r') as zip_ref:
    zip_ref.extractall("./")

directories = os.listdir(folder_extracted)
l = [[], [], [], []]

f = open('./train.csv', 'w')
f2 = open('./test.csv', 'w')
w = csv.writer(f)
w2 = csv.writer(f2)
w.writerow(['label', 'file_name'])
w2.writerow(['label', 'file_name'])
k = 0
for dir in directories:
    for file in os.listdir(os.path.join(folder_extracted, dir)):
        l[k].append([enumClasses.Classes[dir].value, file])
    k += 1
train = []
test = []
for i in range(k):
    train = train + l[i][:round(len(l[i])*0.8)]
    test = test + l[i][round(len(l[i])*0.8):]
for tr in train:
    w.writerow(tr)
for ts in test:
    w2.writerow(ts)
f.close()
f2.close()
if not os.path.exists(folder_images):
    os.mkdir(folder_images)
for root, dirs, files in os.walk(folder_extracted, topdown=False):
    for name in files:
        if os.path.isfile(os.path.join(root, name)):
            os.rename(os.path.join(root, name),
                      os.path.join(folder_images, name))
    for name in dirs:
        if os.path.isfile(os.path.join(root, name)):
            os.rename(os.path.join(root, name),
                      os.path.join(folder_images, name))
shutil.rmtree(folder_extracted)
print("Dataset generated!")
