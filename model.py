import cv2
import json
import torch
import numpy as np
from PIL import Image
from collections import deque
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity

def frame_num_to_timestamp(frame_num, fps):
    frame_sec = frame_num / fps
    f_hour = int(frame_sec // 3600)
    f_min = int((frame_sec % 3600) // 60)
    f_sec = frame_sec % 60
    return f"{f_hour:02d}h:{f_min:02d}m:{f_sec:06.3f}s"

def write_video_clip_from_buffer(filename, frame_buffer, fourcc, fps, frameSize, isColor=True):
    writer = cv2.VideoWriter(f"results/{filename}", fourcc, fps, frameSize, isColor)
    for frame in frame_buffer:
        writer.write(frame)
    writer.release()

def face_crop(frame_rgb, bbox, output_size):
    x, y, w, h = bbox
    x2, y2 = x+w, y+h
    cropped = frame_rgb[y:y2, x:x2]
    crop_h,crop_w = cropped.shape[:2]
    if crop_h != crop_w:
        max_side = max(crop_h, crop_w)
        square_crop = np.zeros((max_side, max_side, 3), dtype=cropped.dtype)
        x_offset = (max_side - crop_w) // 2
        y_offset = (max_side - crop_h) // 2
        square_crop[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped
        cropped = square_crop
    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    return cv2.resize(cropped_bgr, output_size)


class FaceDetection:
    def __init__(self, mtcnn, resnet, embeddings_refs, threshold = 0.63):
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.embeddings_refs = embeddings_refs
        self.threshold = threshold

    def detector(self, pil_image):
        face_tensor = self.mtcnn(pil_image)
        bboxes, detection_probs = self.mtcnn.detect(pil_image)
        best_face_index = None
        best_embedding = None

        if face_tensor is not None and bboxes is not None:
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
            embeddings = self.resnet(face_tensor.to(torch.float32))
            for i, embedding in enumerate(iterable=embeddings):
                similarities = [cosine_similarity(embedding.unsqueeze(0), ref.unsqueeze(0)).item() for ref in self.embeddings_refs]
                max_similarity = max(similarities)
                if max_similarity > self.threshold:
                    best_face_index = i
                    best_embedding = embedding
        return best_face_index, bboxes, detection_probs, best_embedding


class Redetection:
    def __init__(self, mtcnn, resnet, embeddings_refs, margin, threshold = 0.8):
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.embeddings_refs = embeddings_refs
        self.threshold = threshold
        self.margin = margin

    def redetector(self, frame_rgb, last_bbox):
        x, y, w, h = last_bbox
        x2, y2 = x + w, y + h
        roi_x1 = max(0, x - self.margin)
        roi_y1 = max(0, y - self.margin)
        roi_x2 = min(frame_rgb.shape[1], x2 + self.margin)
        roi_y2 = min(frame_rgb.shape[0], y2 + self.margin)
        roi_frame = frame_rgb[roi_y1 : roi_y2, roi_x1 : roi_x2]
        roi_image = Image.fromarray(roi_frame)
        roi_tensor = self.mtcnn(roi_image)
        roi_bboxes, roi_detection_probs = self.mtcnn.detect(roi_image)
        best_face_index = None
        best_embedding = None

        if roi_tensor is not None and roi_bboxes is not None:
            if roi_tensor.ndim == 3:
                roi_tensor = roi_tensor.unsqueeze(0)
            embeddings = self.resnet(roi_tensor.to(torch.float32))
            for j, embedding in enumerate(embeddings):
                similarities = [cosine_similarity(embedding.unsqueeze(0), ref.unsqueeze(0)).item() for ref in self.embeddings_refs]
                max_similarity = max(similarities)
                if max_similarity > self.threshold:
                    best_face_index = j
                    best_embedding = embedding
        
        if best_face_index is not None:
            rx1, ry1, rx2, ry2 = map(int, roi_bboxes[best_face_index])
            rx1, rx2, ry1, ry2 = rx1 + roi_x1, rx2 + roi_x1, ry1 + roi_y1, ry2 + roi_y1
            rw, rh = rx2 - rx1, ry2 - ry1
            return [rx1, ry1, rw, rh], roi_detection_probs[best_face_index], best_embedding
        else:
            return None, None, None
        
class FaceClipExtractor:
    def __init__(self, video_path, ref_image_path, output_frame_size, margin, fps):
        self.video_path = video_path
        self.ref_image_path = ref_image_path
        self.output_frame_size = output_frame_size
        self.margin = margin
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.metadata = []
        self.active_detection = False
        self.clip_index = 0
        self.frame_num = 0
        self.last_bbox = None
        self.frame_buffer = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.mtcnn = MTCNN(min_face_size=8, device=self.device).eval()
        self.resnet = InceptionResnetV1(pretrained="vggface2", num_classes=1).eval().to(self.device)
        self.embeddings_refs = deque(maxlen=10)
        self.load_ref_image()
        self.detector = FaceDetection(self.mtcnn, self.resnet, self.embeddings_refs, threshold=0.53)
        self.redetector = Redetection(self.mtcnn, self.resnet, self.embeddings_refs, margin=self.margin, threshold=0.1)
    
    def load_ref_image(self):
        ref_img = cv2.imread(self.ref_image_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        pil_ref = Image.fromarray(ref_img)
        face_tensor = self.mtcnn(pil_ref).to(self.device)
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        ref_embeddings = self.resnet(face_tensor.to(torch.float32))
        for emb in ref_embeddings:
            self.embeddings_refs.append(emb)

    def finalize_clip(self, frame_num):
        if self.frame_buffer and self.current_clip is not None:
            filename = self.current_clip["filename"]
            write_video_clip_from_buffer(filename, self.frame_buffer, self.fourcc, self.fps, self.output_frame_size)
            self.current_clip["end_time"] = frame_num_to_timestamp(frame_num - 1, self.fps)
            self.metadata.append(self.current_clip)
        self.frame_buffer = []
        self.current_clip = None
        self.active_detection = False
        self.last_bbox = None

    def process_frame(self, frame_rgb, frame_num):
        pil_frame = Image.fromarray(frame_rgb)
        face_detected = False
        best_face_index, bboxes, detection_probs, best_embedding = self.detector.detector(pil_frame)
        if best_face_index is not None and bboxes is not None:
            x1, y1, x2, y2 = map(int, bboxes[best_face_index])
            bbox = [x1, y1, x2 - x1, y2 - y1]
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if not self.active_detection:
                self.active_detection = True
                self.frame_buffer = []
                self.current_clip = {"filename":f"clip_{self.clip_index}.mp4", "start_time":frame_num_to_timestamp(frame_num, self.fps), "frames":[]}
                self.clip_index += 1
            crop_img = face_crop(frame_rgb, bbox, self.output_frame_size)
            if crop_img is not None:
                self.frame_buffer.append(crop_img)
                self.current_clip["frames"].append({"bounding box":bbox})
            self.last_bbox = bbox
            face_detected =  True 
            if best_embedding is not None:
                self.embeddings_refs.append(best_embedding)   
        
        if not face_detected and self.active_detection and self.last_bbox is not None:
            redetected_bbox, redetected_prob, redetected_embedding = self.redetector.redetector(frame_rgb, self.last_bbox)
            if redetected_bbox is not None:
                face_detected = True
                rx, ry, rw, rh = redetected_bbox
                cv2.rectangle(frame_rgb, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                crop_img = face_crop(frame_rgb, redetected_bbox, self.output_frame_size)
                if crop_img is not None:
                    self.frame_buffer.append(crop_img)
                    self.current_clip["frames"].append({"bounding box": redetected_bbox})
                self.last_bbox = redetected_bbox
                

        
        if not face_detected and self.active_detection:
            self.finalize_clip(frame_num)

    def run(self):
        vid = cv2.VideoCapture(self.video_path)
        if not vid.isOpened():
            raise ValueError("cannot open video")
        frame_num = 0
        while vid.isOpened():
            ret, frame_bgr = vid.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.process_frame(frame_rgb, frame_num)
            frame_bgr_disp = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", frame_bgr_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_num += 1
        if self.active_detection:
            self.finalize_clip(frame_num)
        with open("results/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        vid.release()
        cv2.destroyAllWindows()


def main():
    ref_image_path = "resources/Ref.jpg"
    video_path = "resources/Ref.mp4"
    extractor = FaceClipExtractor(video_path, ref_image_path, output_frame_size = (300, 300), margin = 10, fps=24)
    extractor.run()

if __name__ == "__main__":
    main()


