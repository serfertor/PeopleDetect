from ultralytics import YOLO
import cv2 as cv
import time


class Model:
    coord = [{'A': [77, 426, 282, 714], 'B': [229, 574, 562, 838], 'C': [508, 545, 774, 729]},
             {'D': [1285, 355, 1443, 418], 'E': [1468, 355, 1605, 456], 'F': [1608, 357, 1771, 502],
              'G': [1145, 450, 1335, 595], 'H': [1230, 519, 1580, 762], 'I': [1383, 445, 1568, 589]}]

    def __init__(self):
        self.model = YOLO("best.pt")

    def calculate_people(self, img1, img2):
        log = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
               'F': 0, 'G': 0, 'H': 0, 'I': 0}
        sum_people = 0
        for position in self.coord:
            result = self.model.predict(img1)[0]
            for _ in result.boxes.xyxy:
                for i in position.keys():
                    if (position[i][0] <= (_[0] + _[2]) / 2 <= position[i][2] and position[i][1] <= (_[1] + _[3]) / 2 \
                            <= position[i][3]):
                        log[i] += 1
                        sum_people += 1
                        break
        return sum_people, log


'''    def use_stream(self, stream_address):
        cap = cv.VideoCapture(stream_address)
        interval = 30  # интервал в секундах
        prev_time = time.time()

        while cap.isOpened():
            log = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
                   'F': 0, 'G': 0, 'H': 0, 'I': 0}
            sum_people = 0
            ret, frame = cap.read()
            if ret:
                current_time = time.time()
                if current_time - prev_time >= interval:
                    result = model.predict(frame)[0]
                    for _ in result.boxes.xyxy:
                        for i in coord.keys():
                            if coord[i][0] <= (_[0] + _[2]) / 2 <= coord[i][2] and coord[i][1] <= (_[1] + _[3]) / 2 <= \
                                    coord[i][3]:
                                log[i] += 1
                                sum_people += 1
                                break

                    for i in log.keys():
                        print("Количество человек в области {} - {}".format(i, log[i]))
                    print("Всего человек - " + str(sum_people))
                    prev_time = current_time
            else:
                break

        cap.release() '''
