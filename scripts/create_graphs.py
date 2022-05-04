import csv
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

csv_path = 'bar_results.csv'
with open(csv_path, 'w+') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["ID", "Type", "Correctly detected and clasify (all)", "Correctly detected and clasify (real)"])

    csv_file.close()

with open(csv_path, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([1, "SSD MobileNet", 70.991, 68.226])
    writer.writerow([2, "SSD MobileNet", 66.412, 60.861])
    writer.writerow([3, "SSD ResNet-50", 70.934, 67.123])
    writer.writerow([4, "SSD ResNet-50", 63.8, 55.604])
    writer.writerow([5, "Faster R-CNN ResNet-50", 80.263, 76.875])
    writer.writerow([6, "Faster R-CNN ResNet-50", 77.851, 73.971])
    writer.writerow([7, "Faster R-CNN ResNet-50", 78.268, 73.808])
    writer.writerow([8, "Faster R-CNN ResNet-101", 78.043, 73.241])
    writer.writerow([9, "Faster R-CNN ResNet-101", 76.898, 72.616])
    writer.writerow([10, "Faster R-CNN ResNet-101", 76.348, 70.175])
    writer.writerow([11, "EfficientDet D0", 71.893, 67.633])
    writer.writerow([12, "EfficientDet D0", 72.788, 70.448])
    writer.writerow([13, "EfficientDet D1", 71.23, 68.106])
    writer.writerow([14, "EfficientDet D1", 76.027, 75.329])
    writer.writerow([15, "CenterNet Hourglass", 72.682, 66.736])
    writer.writerow([16, "CenterNet Hourglass", 76.551, 72.019])
    writer.writerow([17, "CenterNet ResNet-101", 76.608, 73.682])
    writer.writerow([18, "CenterNet ResNet-101", 77.193, 74.624])

    csv_file.close()

#col_list = ["ID", "Type", "Correctly detected and clasify (all)"]
#df = pd.read_csv(csv_path, usecols=col_list)

data = pd.read_csv(r"bar_results.csv")
data.head()
df = pd.DataFrame(data)

#name = df['car'].head(12)
#price = df['price'].head(12)

ids = df['ID'].astype(str)
vals = df['Correctly detected and clasify (real)']
print(ids)
print(vals)

c = ['#59A4F2',
    '#59A4F2',
    '#0076F1',
    '#0076F1',
    '#50BE9F',
    '#50BE9F',
    '#50BE9F',
    '#00A073',
    '#00A073',
    '#00A073',
    '#BCA79C',
    '#BCA79C',
    '#54433A',
    '#54433A',
    '#F0AC82',
    '#F0AC82',
    '#E17A3B',
    '#E17A3B']

NA = mpatches.Patch(color='#59A4F2', label='SSD MobileNet')
EU = mpatches.Patch(color='#0076F1', label='SSD ResNet-50')
AP = mpatches.Patch(color='#50BE9F', label='Faster R-CNN ResNet-50')
SA = mpatches.Patch(color='#00A073', label='Faster R-CNN ResNet-101')
NN = mpatches.Patch(color='#BCA79C', label='EfficientDet D0')
EE = mpatches.Patch(color='#54433A', label='EfficientDet D1')
AA = mpatches.Patch(color='#F0AC82', label='CenterNet Hourglass')
SS = mpatches.Patch(color='#E17A3B', label='CenterNet ResNet-101')


fig = plt.figure(figsize =(10, 7))

# Horizontal Bar Plot
plt.barh(ids, vals, color = c)
#plt.title('Normalized IoU Distributions')
plt.xlim(0,100)
plt.ylabel('ID trénování')
plt.xlabel('Správně detekovaná a klasifikovaná plocha [%]')
plt.legend(handles=[NA,EU,AP,SA,NN,EE,AA,SS], loc=2)
plt.savefig('graph_to_doc.pdf', bbox_inches='tight')
