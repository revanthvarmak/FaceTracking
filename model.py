from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# resent = InceptionResnetV1()
# help(resent)

mtcnn = MTCNN().eval()
resnet = InceptionResnetV1(pretrained="vggface2", num_classes=1).eval()
deepsort = DeepSort()

vid = cv2.VideoCapture("resources/Ref.mp4")
ref_image = cv2.imread("resources/Ref.jpg")
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

def frame_num_to_timestamp(frame_num, fps = 30):
    frame_sec = frame_num / fps
    f_hour = int(frame_sec // 3600)
    f_min = int((frame_sec % 3600) // 60)
    f_sec = frame_sec % 60
    return f"{f_hour:02d}h:{f_min:02d}m:{f_sec:06.3f}s"


while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        face_tensor = mtcnn(image)
        boxes, detection_probs = mtcnn.detect(image)
        if face_tensor is not None:
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
            target_tensor = face_tensor.to(torch.float32)
            embedding_tars = resnet(target_tensor)
            for i, embedding_tar in enumerate(embedding_tars):
                match = False
                for embedding_ref in embeddings_refs:
                    similarity = cosine_similarity(embeddings_ref, embedding_tar.unsqueeze(0))
                    if similarity > 0.33:
                        match = True
                        embeddings_refs.append(embedding_tars[i])
                        break
                if match:
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
                    if not new_detection:
                        new_detection = True
                        out = cv2.VideoWriter(f"results/clip_{clip_index}.mp4", fourcc, 30.0, (160, 160),isColor=True)
                        current_clip = {
                            "file name" : f"clip_{clip_index}.mp4",
                            "start_time" : frame_num_to_timestamp(frame_num),
                            "frames" : []
                        }
                        clip_index += 1
                    cropped_frame_rgb = frame[y1:y2, x1:x2]
                    if cropped_frame_rgb.size == 0:
                        pass
                    else:
                        cropped_frame_bgr = cv2.cvtColor(cropped_frame_rgb, cv2.COLOR_BGR2RGB)
                        cropped_frame_bgr = cv2.resize(cropped_frame_bgr, (160, 160))
                        out.write(cropped_frame_bgr)
                    w = x2 - x1
                    h = y2 - y1
                    current_clip["frames"].append({
                        "bounding box" : [x1, y1, w, h]
                    })
                    tracks = deepsort.update_tracks([([x1, y1, w, h], detection_probs[i])], [embedding_tars[i].detach().numpy()])   
                    for track in tracks:
                        track_id = track.track_id
                        tentative = track.is_tentative()
                        confidence = track.get_det_conf()
                        status = "Tentative" if tentative else "Confirmed"
                        text = f"track_id: {track_id}, status: {status}"

                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                        cv2.putText(frame, str(confidence), (x1, y2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0, 0), 4)
                else:
                    if new_detection:
                        new_detection = False
                        out.release()
                        out = None
                        current_clip["end_time"] = frame_num_to_timestamp(frame_num - 1)
                        metadata.append(current_clip)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)            
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_num += 1
    else:
        break

if new_detection:
    new_detection = False
    out.release()
    out = None
    current_clip["end_time"] = frame_num_to_timestamp(frame_num - 1)
    metadata.append(current_clip)

with open("results/metadata.json", "w") as f:
    json.dump(metadata, f, indent= 2)


vid.release()
cv2.destroyAllWindows()
