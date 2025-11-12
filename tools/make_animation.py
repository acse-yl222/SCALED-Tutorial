import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
import sys
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(project_root)
# os.chdir(project_root)

# sys.path.append(project_root)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]

def images_to_gif(folder_path, output_path="output.gif", duration=100, resize_factor=1):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [f for f in os.listdir('result') if f.lower().endswith(valid_exts)]
    image_files.sort(key=natural_sort_key)  # 按文件名排序
    image_files = [image_files[i] for i in range(len(image_files))]
    if not image_files:
        print("❌ 没有找到可用的图片文件。")
        return

    # 读取并缩放所有图片
    images = []
    for file in image_files:
        img = Image.open(os.path.join(folder_path, file))
        if resize_factor != 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size)
        images.append(img)
    images = images

    # 将第一张作为起始图，其余作为追加帧
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,  # 每帧持续时间（毫秒）
        loop=0              # 循环次数，0 表示无限循环
    )

    print(f"✅ GIF 已生成：{output_path}")

# images_to_gif('result_u',output_path='u.gif',duration=5)
# images_to_gif('result',output_path='xy.gif',duration=5)
# images_to_gif('result_h',output_path='xz.gif',duration=5)

images_to_gif('result',output_path='result.gif',duration=5)


# images_to_gif('south_kensington','south_kensington.gif',duration=5)
# images_to_gif('LondonAirpot','LondonAirpot.gif',duration=5)
# images_to_gif('Canary Wharf','CanaryWharf.gif',duration=5)
# images_to_gif('LondonBridge','LondonBridge.gif',duration=5)