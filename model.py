from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from deep_sort_realtime.deepsort_tracker import DeepSort
import json

if torch.backends.mps.is_available():
    device = torch.device("mps")

mtcnn = MTCNN(min_face_size=8).eval()
resnet = InceptionResnetV1(pretrained="vggface2", num_classes=1).eval()
deepsort = DeepSort(max_age=60, n_init=2, max_iou_distance=0.5)

vid = cv2.VideoCapture("resources/will.mp4")
ref_image = cv2.imread("resources/will.jpg")
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(ref_image)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
metadata = []

face_tensor = mtcnn(image, save_path = "face.jpg")
face_tensor = face_tensor.unsqueeze(0)
ref_tensor = face_tensor.to(torch.float32)
embeddings_ref = resnet(ref_tensor)
embeddings_refs = list(embeddings_ref)
new_detection = False
clip_index = 0
frame_num = 0
last_bbox = None
fps = 30
margin = 30

def frame_num_to_timestamp(frame_num, fps = fps):
    frame_sec = frame_num / fps
    f_hour = int(frame_sec // 3600)
    f_min = int((frame_sec % 3600) // 60)
    f_sec = frame_sec % 60
    return f"{f_hour:02d}h:{f_min:02d}m:{f_sec:06.3f}s"


while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    face_tensor = mtcnn(image)
    boxes, detection_probs = mtcnn.detect(image)
    matched_in_this_frame = False
    if face_tensor is not None:
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        threshold = 0.33
        embedding_tars = resnet(face_tensor.to(torch.float32))
        for i, embedding_tar in enumerate(embedding_tars):
            match = False
            for embedding_ref in embeddings_refs:
                similarity = cosine_similarity(embeddings_ref, embedding_tar.unsqueeze(0))
                if similarity > threshold:
                    match = True
                    embeddings_refs.append(embedding_tars[i])
                    break
            if match:
                matched_in_this_frame = True
                x1, y1, x2, y2 = map(int, boxes[i])
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
                if not new_detection:
                    new_detection = True
                    out = cv2.VideoWriter(f"results/clip_{clip_index}.mp4", fourcc, fps, (160, 160),isColor=True)
                    current_clip = {
                        "file name" : f"clip_{clip_index}.mp4",
                        "start_time" : frame_num_to_timestamp(frame_num),
                        "frames" : []
                    }
                    clip_index += 1
                cropped_frame_rgb = frame[y1:y2, x1:x2]
                if cropped_frame_rgb.size == 0:
                    pass
                cropped_frame_bgr = cv2.cvtColor(cropped_frame_rgb, cv2.COLOR_BGR2RGB)
                cropped_frame_bgr = cv2.resize(cropped_frame_bgr, (160, 160))
                out.write(cropped_frame_bgr)
                current_clip["frames"].append({
                    "bounding box" : [x1, y1, w, h]
                })
                last_bbox = [x1, y1, w, h]
                tracks = deepsort.update_tracks([([x1, y1, w, h], detection_probs[i])], [embedding_tars[i].detach().numpy()])   
                for track in tracks:
                    track_id = track.track_id
                    tentative = track.is_tentative()
                    confidence = track.get_det_conf()
                    status = "Tentative" if tentative else "Confirmed"
                    text = f"track_id: {track_id}, status: {status}"

                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                    cv2.putText(frame, str(confidence), (x1, y2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                break

    if not matched_in_this_frame and new_detection:
        if last_bbox is not None:
            x1, y1, w, h = last_bbox
            x2, y2 = x1 + w, y1 + h
            roi_x1 = max(0, x1 - margin)
            roi_x2 = min(frame.shape[1], x2 + margin)
            roi_y1 = max(0, y1 - margin)
            roi_y2 = min(frame.shape[0], y2 + margin)

            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_image = Image.fromarray(roi_frame)
            roi_tensor = mtcnn(roi_image)
            roi_boxes, roi_detection_probs = mtcnn.detect(roi_image)
            if roi_tensor is not None:
                if roi_tensor.ndim == 3:
                    roi_tensor = roi_tensor.unsqueeze(0)
                roi_threshold = 0.09
                embedding_rois = resnet(roi_tensor.to(torch.float32))
                for j, embedding_roi in enumerate(embedding_rois):
                    roi_match = False
                    for embedding_ref in embeddings_refs:
                        roi_similarity = cosine_similarity(embeddings_ref, embedding_roi.unsqueeze(0))
                        if roi_similarity > roi_threshold:
                            roi_match = True
                            embeddings_refs.append(embedding_rois[j])
                            break
                    if roi_match:
                        matched_in_this_frame = True

                        rx1, ry1, rx2, ry2 = map(int, roi_boxes[j])

                        rx1 += roi_x1
                        rx2 += roi_x1
                        ry1 += roi_y1
                        ry2 += roi_y1

                        rw, rh = rx2 - rx1, ry2 - ry1
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0))
                        cropped_roi_rgb = frame[ry1:ry2, rx1:rx2]
                        if cropped_roi_rgb.size == 0:
                            pass
                        cropped_roi_bgr = cv2.cvtColor(cropped_roi_rgb, cv2.COLOR_BGR2RGB)
                        cropped_roi_bgr = cv2.resize(cropped_roi_bgr, (160, 160))
                        out.write(cropped_roi_bgr)
                        current_clip["frames"].append({
                            "bounding box" : [rx1, ry1, rw, rh]
                        })
                        last_bbox = [rx1, ry1, rw, rh]
                        tracks = deepsort.update_tracks([([rx1, ry1, w, h], roi_detection_probs[j])], [embedding_rois[j].detach().numpy()])   
                        for track in tracks:
                            track_id = track.track_id
                            tentative = track.is_tentative()
                            confidence = track.get_det_conf()
                            status = "Tentative" if tentative else "Confirmed"
                            text = f"track_id: {track_id}, status: {status}"

                            cv2.putText(frame, text, (rx1, ry1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                            cv2.putText(frame, str(confidence), (rx1, ry2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                        break    

        if not matched_in_this_frame:
            new_detection = False
            out.release()
            out = None
            current_clip["end_time"] = frame_num_to_timestamp(frame_num - 1)
            metadata.append(current_clip)
            current_clip = None
            last_bbox = None

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1

if new_detection and current_clip is not None:
    new_detection = False
    out.release()
    out = None
    current_clip["end_time"] = frame_num_to_timestamp(frame_num - 1)
    metadata.append(current_clip)

with open("results/metadata.json", "w") as f:
    json.dump(metadata, f, indent= 2)


vid.release()
cv2.destroyAllWindows()
