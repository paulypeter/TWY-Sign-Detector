from wpod_net import sign_detection, detect_sign_areas
from yolov5 import detect as detect_arrows

def run(input_dir):
    sign_img = sign_detection.detect(input_dir)
    segments = detect_sign_areas.segmenting(sign_img)
    for segment in segments:
        if segments[0] == "yellow":
            detect_arrows.main(segment[1])
            # remove arrows from crop image
        # perform OCR here

if __name__ == "__main__":
    run('/home/peter/IFF/Masterarbeit/DeepLearning/alpr-unconstrained/samples/test')