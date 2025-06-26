from PIL import Image, ImageDraw
import os

def create_favicon():
    # Create a 32x32 image with a white background
    size = 32
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple blue circle
    draw.ellipse([2, 2, size-2, size-2], fill='#007bff')
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Save as ICO file
    favicon_path = os.path.join('static', 'favicon.ico')
    image.save(favicon_path, format='ICO')
    print(f"Favicon created at: {favicon_path}")

if __name__ == "__main__":
    create_favicon() 