import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    
    #RTSP를 불러오는 곳
    video_capture = cv2.VideoCapture('./result.m4v')
    
    # 웹캠 설정
    video_capture.set(3, 1920)  # 영상 가로길이 설정
    video_capture.set(4, 1080)  # 영상 세로길이 설정
    fps = 20
    # 가로 길이 가져오기
    streaming_window_width = int(video_capture.get(3))
    # 세로 길이 가져오기
    streaming_window_height = int(video_capture.get(4))  
    

    #파일 저장하기 위한 변수 선언
    path = "/home/user/Downloads/rtsp/tensorflow-yolov4-tflite/result.avi"
    
    # DIVX 코덱 적용 # 코덱 종류 # DIVX, XVID, MJPG, X264, WMV1, WMV2
    # 무료 라이선스의 이점이 있는 XVID를 사용
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    
    # 비디오 저장
    # cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
    out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    count = 1

    while True:
        ret, frame = video_capture.read()
        # # 촬영되는 영상보여준다. 프로그램 상태바 이름은 'streaming video' 로 뜬다.
        # cv2.imshow('streaming video', frame)
        
        # # 영상을 저장한다.
        # out.write(frame)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size
        image_path = FLAGS.image

        original_image = frame
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)

        

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(original_image, pred_bbox)
        # image = utils.draw_bbox(image_data*255, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imshow('streaming video', image)
        out.write(image)
        print("frame : {0}".format(count))
        count +=1
        # 1ms뒤에 뒤에 코드 실행해준다.
        k = cv2.waitKey(1) & 0xff
        #키보드 esc 누르면 종료된다.
        if k == 27:
            break
    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
