from PIL import Image, ImageDraw

# 定义要可视化的颜色字符串列表
colors = [
    'black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
    'gray', 'lightgray', 'darkgray', 'silver',
    'lightcoral', 'lightpink', 'lightgreen', 'lightblue', 'lightyellow',
    'darkred', 'darkgreen', 'darkblue', 'darkorange',
    'orange', 'purple', 'brown', 'violet', 'indigo'
]

# 每个颜色方块的大小
square_size = 100
# 计算图像的宽度和高度
num_colors_per_row = 5
num_rows = len(colors) // num_colors_per_row + (1 if len(colors) % num_colors_per_row != 0 else 0)
image_width = num_colors_per_row * square_size
image_height = num_rows * square_size

# 创建一个新的白色图像
image = Image.new('RGB', (image_width, image_height), color='white')
draw = ImageDraw.Draw(image)

# 绘制颜色方块
x = 0
y = 0
for color in colors:
    draw.rectangle((x, y, x + square_size, y + square_size), fill=color)
    # 在方块上添加颜色名称文本
    draw.text((x + 10, y + 10), color, fill='white' if color != 'white' else 'black')
    x += square_size
    if x >= image_width:
        x = 0
        y += square_size

# 保存图像
image.save('color_visualization.png')
# 显示图像
image.show()