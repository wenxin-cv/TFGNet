import numpy as np
# pip install insightface==0.7.3
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class FaceAnalysis2(FaceAnalysis):
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)

def analyze_faces(face_analysis: FaceAnalysis, img_data: np.ndarray, det_size=(640, 640)):
    detection_sizes = [None] + [(size, size) for size in range(640, 256, -64)] + [(256, 256)]

    for size in detection_sizes:
        faces = face_analysis.get(img_data, det_size=size)
        if len(faces) > 0:
            return faces

    return []
