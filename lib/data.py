from os import listdir
from os.path import isfile, join
from imagelib.extractor import Extractor

import cv2


class DataMaker:
    def __init__(self):
        self.extractor = Extractor()

    def batch(self, base_path, save_path, x_limit = -1):
        files = ["{}{}".format(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f))]
        graph_list = []

        for file in files:
            try:
                image = cv2.imread(file)
                cropped_image = self.crop(image)
                left_graph, right_graph = self.get_graph(cropped_image)

                graph_list.extend(self.to_list(file, left_graph, x_limit))
                graph_list.extend(self.to_list(file, right_graph, x_limit))
            except Exception as e:
                graph_list.append("{}\t-1\t{}".format(file, str(e)))

        with open(save_path, "w") as f:
            f.write("\n".join(graph_list))

        print("batch done.")

    def crop(self, image):  # 26, 110, 705, 920 (Left) | 740, 110, 1426, 920 (Right)
        left = image[115:920, 26:705]
        right = image[115:920, 740:1426]
        cv2.imwrite("./data/crop.png", left)
        return [left, right]

    def get_graph(self, cropped_image):
        return self.extractor.extract_image_with_peak(cropped_image[0], True), \
               self.extractor.extract_image_with_peak(cropped_image[1], False)

    def to_list(self, file_path, graph_list, x_limit=-1):
        text_list = []
        for i in range(len(graph_list)):
            graph_item = graph_list[i]
            graph = [str(g) for g in graph_item["graph"]]
            peak_list = graph_item["peak"]
            peak_list = [str(x) for (x, y) in peak_list]

            if x_limit > 0:
                if len(graph) > x_limit:
                    graph = graph[:x_limit]
                else:
                    graph = graph + ["0" for x in range(x_limit - len(graph))]

            text_list.append("{}\t{}\t{}\t{}".format(file_path, i, ",".join(graph), ",".join(peak_list)))

        return text_list
