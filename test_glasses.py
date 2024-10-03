import cv2
import numpy as np

# Hàm xoay ảnh mà không cắt góc và giữ nguyên sự trong suốt
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Tính toán kích thước khung mới để chứa toàn bộ ảnh sau khi xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Tính kích thước mới của khung sau khi xoay
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Điều chỉnh ma trận xoay để căn giữa ảnh
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Xoay ảnh, bao gồm cả kênh alpha để giữ sự trong suốt
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return rotated

# Đọc ảnh PNG (bao gồm kênh alpha)
image1 = cv2.imread('data/glasses/glasses_01.png', cv2.IMREAD_UNCHANGED)  # Sử dụng IMREAD_UNCHANGED để giữ kênh alpha
image2 = cv2.imread('c:/Users/four/Downloads/Untitled.png', cv2.IMREAD_UNCHANGED)

# Xoay ảnh 1 10 độ
rotated_image1 = rotate_image(image1, 10)

# Đảm bảo ảnh nền (image2) có cùng kênh alpha
if image2.shape[2] == 3:
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)

# Vị trí để chèn (có thể điều chỉnh)
(x_offset, y_offset) = (50, 50)

# Giới hạn kích thước của vùng chèn sao cho không vượt quá kích thước của ảnh nền
y1, y2 = max(0, y_offset), min(image2.shape[0], y_offset + rotated_image1.shape[0])
x1, x2 = max(0, x_offset), min(image2.shape[1], x_offset + rotated_image1.shape[1])

# Giới hạn kích thước ảnh đã xoay để phù hợp với vùng chèn
rotated_crop = rotated_image1[0:(y2 - y1), 0:(x2 - x1)]

# Tạo vùng chứa để giữ sự trong suốt
alpha_rotated = rotated_crop[:, :, 3] / 255.0  # Lấy kênh alpha của ảnh đã xoay
alpha_background = 1.0 - alpha_rotated

# Chèn ảnh vào nền với alpha
for c in range(0, 3):  # Chèn các kênh RGB với alpha
    image2[y1:y2, x1:x2, c] = (alpha_rotated * rotated_crop[:, :, c] +
                               alpha_background * image2[y1:y2, x1:x2, c])

# Lưu kết quả với định dạng PNG để giữ sự trong suốt
cv2.imwrite('output.png', image2)

# Hiển thị ảnh kết quả
cv2.imshow('Output Image', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
