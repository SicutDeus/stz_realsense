import math
import numpy as np
import cv2
__all__ = [cv2]
from utils.utils import Utils
from cfg.config import Config


class FrameDetection:
    @staticmethod
    def detect_and_label(image, net, depth_frame):
        """
        Полная обработка входного изображения.
        :param image: входное изображение
        :param net: модель нейронной сети
        :return: размеченное изображение
        """
        yolo5_img = Utils.convert_frame_to_yolov5_format(image)
        output_data = FrameDetection.__detect_on_frame(yolo5_img, net)
        output_img = FrameDetection.__label_on_frame(yolo5_img, output_data,depth_frame)
        result = output_img[0:480, 0:640]
        cv2.imwrite('2.jpg',depth_frame)
        cv2.imwrite('1.jpg',image)
        return result

    @staticmethod
    def __detect_on_frame(image, net):
        """
        Получает сырые данные детекции.
        :param image: входное изображения
        :param net: модель нейронной сети
        :return: сырые данные детекции
        """
        H,W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0,
                                     (W, H), swapRB=True, crop=False)
        print(blob.shape)
        net.setInput(blob)
        try:
            predictions = net.forward()
        except Exception as e:
            print(e.args)
        return predictions

    @staticmethod
    def __wrap_detection(input_image, output_data):
        """
        Обёртывает данные после детекции.
        :param input_image: входное изображение
        :param output_data: данные, полученные после детекции о нахождении объекта на кадре
        :return: id класса, уверенность в определении объекта, локация объекта
        """
        class_ids = []
        confidences = []
        boxes = []
        rows = output_data.shape[0]
        image_width, image_height, _ = input_image.shape
        H, W = input_image.shape[:2]
        x_factor = image_width / W
        y_factor = image_height / H
        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:
                classes_scores = row[5:]
                _, _, _, max_index = cv2.minMaxLoc(classes_scores)
                class_id = max_index[1]
                if classes_scores[class_id] > .25:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
        return result_class_ids, result_confidences, result_boxes

    @staticmethod
    def __label_on_frame(frame, outs, depth_frame):
        """
        Отмечает на входном изображении расположение объекта.
        :param frame: входное изображение
        :param outs: данные, полученные после детекции о нахождении объекта на кадре
        :return: размеченное изображение в cv2 формате
        """
        color = (255,0,0)
        depth_frame = cv2.resize(depth_frame,(640,640))
        class_ids, confidences, boxes = FrameDetection.__wrap_detection(frame, outs[0])
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            if confidence >= .5 :
                try:
                    color = Config.colors[int(classid) % len(Config.colors)]
                    cv2.rectangle(frame, box, color, 2)
                    cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                    cv2.putText(frame, Config.class_list[classid] + ' ' + str(math.floor(confidence * 100)) + '% ' +
                                str(float(depth_frame[(int(box[0] + box[2]/2), int(box[1]+box[3]/2))]/10)) + 'sm',
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), thickness=1,)
                    #cv2.circle(frame,((int(box[0] + box[2]/2), int(box[1]+box[3]/2))),6,(0,0,255))
                except Exception as ex:
                    print (ex.args)

        return frame