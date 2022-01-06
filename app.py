import time
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
from flask import Flask, render_template, Response

app = Flask(__name__)
camera = cv2.VideoCapture("rtsp://admin:Haesung!!@hscam1.iptime.org:558/Livechannel/0/Live4NVR.smp")
def get_frames():
    framework='tf'
    weights='./checkpoints/yolov4-416'
    size=416
    tiny=False
    model='yolov4'
    video='rtsp://admin:Haesung!!@hscam1.iptime.org:558/Livechannel/0/Live4NVR.smp'
    iou=0.45
    score=0.25
    output=None
    output_format='XVID'
    dis_cv2_window = False

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = size
    video_path = video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)

    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    # vid.set(3, 320)
    # vid.set(4, 240)
    fps = 20
    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(3))
        height = int(vid.get(3))  # 영상 세로길이 설정
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_id = 0
    begin = time.time()
    ################################################
    while True:
        return_value, frame = vid.read()
        frame_id += 1
        end = time.time()
        if int(end - begin)%7 == 0:
            continue 
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
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
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not dis_cv2_window:
            ret, buffer = cv2.imencode('.jpg', result)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if output:
            out.write(result)
        # cv2.imshow("result", frame)
        # cv2.waitKey(1)
        
    vid.release()
    cv2.destroyAllWindows()
    
# def gen_frames():  
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except SystemExit:
        pass