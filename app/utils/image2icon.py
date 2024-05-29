from PIL import Image

# 打开PNG图像
img = Image.open('icon.jpg')

# 保存为ICO文件
img.save('icon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (256, 256)])