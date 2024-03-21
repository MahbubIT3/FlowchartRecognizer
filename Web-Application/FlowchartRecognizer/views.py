# object_detection/views.py
import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

def homepage(request):
    return render(request, 'index.html')

def detect_object(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        img = Image.open(image_file)
        image_np = np.array(img)

        # Perform object detection on the uploaded image
        model = tf.saved_model.load(os.path.join(settings.BASE_DIR, 'media/required_files/saved_model'))
        category_index = label_map_util.create_category_index_from_labelmap(os.path.join(settings.BASE_DIR, 'media/required_files/label_map.pbtxt'))

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)  # Use tf.uint8
        detections = model.signatures['serving_default'](input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be integers.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=50,
            min_score_thresh=.85,
            agnostic_mode=False
        )

        # Save the image with detections
        image_with_detections_path = os.path.join(settings.STATIC_ROOT, 'images', 'image_with_detections.jpg')
        image_with_detections = Image.fromarray(image_np_with_detections)
        image_with_detections.save(image_with_detections_path)


        num_objects_detected = detections['num_detections']
        detected_objects = []
        for i in range(num_objects_detected):
            class_label = category_index[detections['detection_classes'][i]]['name']
            confidence = detections['detection_scores'][i]
            if confidence > 0.85:  # Filter objects with confidence > 0.8
                detected_objects.append({'class': class_label, 'confidence': confidence})

                class_frequencies = {}
                for obj in detected_objects:
                    class_label = obj['class']
                    if class_label in class_frequencies:
                        class_frequencies[class_label] += 1
                    else:
                        class_frequencies[class_label] = 1

                # Sort class frequencies by class label
                class_frequencies = sorted(class_frequencies.items(), key=lambda x: x[0])
                

        context = {'image_with_detections': image_with_detections_path, 'num_objects_detected': num_objects_detected, 'detected_objects': detected_objects, 'class_frequencies': class_frequencies}

        return render(request, 'detection.html', context)

    return render(request, 'detection.html')
