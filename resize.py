from PIL import Image

# Open the image file
original_image = Image.open("images/crosswalk.jpg")

# Resize the image
width = 300  # specify the width
height = 300  # specify the height
resized_image = original_image.resize((width, height))

# Show the resized image
resized_image.show()