from lib.data import DataMaker

maker = DataMaker()

maker.batch("/Users/gwonchan-u/Downloads/Infant ABR 1-50/", "./data/train.txt", 660)
maker.batch("/Users/gwonchan-u/Downloads/Infant ABR 51-110/", "./data/train2.txt", 660)
