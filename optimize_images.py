import os
import sys
from PIL import Image

def optimize_png(input_path, output_path, quality=85):
    """Optimize a PNG image by converting it to a more efficient format."""
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if it has an alpha channel (transparency)
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                # Paste the image on the background
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Try both PNG and JPEG optimization to see which is better
            # Save as optimized PNG
            png_output = output_path.replace('.jpg', '.png')
            img.save(png_output, 'PNG', optimize=True)
            
            # Save as optimized JPEG
            jpg_output = output_path
            img.save(jpg_output, 'JPEG', quality=quality, optimize=True)
            
            # Get file sizes
            original_size = os.path.getsize(input_path)
            png_size = os.path.getsize(png_output)
            jpg_size = os.path.getsize(jpg_output)
            
            # Determine which format is better
            if png_size <= jpg_size and png_size < original_size:
                # PNG is better, remove the JPEG
                os.remove(jpg_output)
                final_output = png_output
                final_size = png_size
                format_used = 'PNG'
            elif jpg_size < original_size:
                # JPEG is better, remove the PNG
                os.remove(png_output)
                final_output = jpg_output
                final_size = jpg_size
                format_used = 'JPEG'
            else:
                # Original is better, remove both optimized versions
                os.remove(png_output)
                os.remove(jpg_output)
                print(f"Original {input_path} is already optimized, skipping.")
                return False
            
            # Calculate size reduction
            reduction = (1 - final_size / original_size) * 100
            
            print(f"Optimized {input_path}:")
            print(f"  Original size: {original_size / 1024:.2f} KB")
            print(f"  New size ({format_used}): {final_size / 1024:.2f} KB")
            print(f"  Reduction: {reduction:.2f}%")
            
            return True
    except Exception as e:
        print(f"Error optimizing {input_path}: {e}")
        return False

def main():
    # Create output directory if it doesn't exist
    output_dir = 'optimized_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in the current directory
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    
    if not png_files:
        print("No PNG files found in the current directory.")
        return
    
    print(f"Found {len(png_files)} PNG files to optimize.")
    
    # Optimize each PNG file
    successful = 0
    total_original_size = 0
    total_new_size = 0
    optimized_files = []
    
    for png_file in png_files:
        input_path = png_file
        output_path = os.path.join(output_dir, os.path.splitext(png_file)[0] + '.jpg')
        
        if optimize_png(input_path, output_path):
            successful += 1
            total_original_size += os.path.getsize(input_path)
            
            # Check if the output is a PNG or JPG (based on our optimization function)
            jpg_path = output_path
            png_path = output_path.replace('.jpg', '.png')
            
            if os.path.exists(jpg_path):
                total_new_size += os.path.getsize(jpg_path)
                optimized_files.append(jpg_path)
            elif os.path.exists(png_path):
                total_new_size += os.path.getsize(png_path)
                optimized_files.append(png_path)
    
    # Print summary
    if successful > 0:
        total_reduction = (1 - total_new_size / total_original_size) * 100
        print(f"\nSummary:")
        print(f"  Successfully optimized {successful} of {len(png_files)} files")
        print(f"  Total original size: {total_original_size / 1024:.2f} KB")
        print(f"  Total new size: {total_new_size / 1024:.2f} KB")
        print(f"  Total reduction: {total_reduction:.2f}%")
        print(f"\nOptimized images saved to the '{output_dir}' directory.")
        print("You can replace the original PNG files with these optimized files to reduce project size.")
        
        # List optimized files
        print("\nOptimized files:")
        for file in optimized_files:
            print(f"  {file}")
    else:
        print("No files were successfully optimized.")
        print("Your images may already be well-optimized.")


if __name__ == "__main__":
    main()