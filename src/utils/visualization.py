import cv2
import time


async def save_predictions_image(image, predictions):
    draw_img = image.copy()
    info = ""
    for box in predictions:
        draw_img = cv2.rectangle(
            draw_img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            3,
        )
        draw_img = cv2.putText(
            draw_img,
            f"{box[5]:.2f}",
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )
        info += f"{box[5]:.2f} "
    # saving file by time
    file_name = f"{int(time.time())}_{info}.jpg"
    cv2.imwrite(file_name, draw_img)
    print(f"Saved image: {file_name}")
