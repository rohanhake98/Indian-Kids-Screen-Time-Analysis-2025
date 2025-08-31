import os
import shutil
import sys

def get_size(path):
    """Get the size of a file or directory in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    return total_size

def format_size(size_bytes):
    """Format size in bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def main():
    # Directories and files to consider for removal
    items_to_check = [
        'venv',                # Virtual environment (large)
        '__pycache__',         # Python cache files
        '.ipynb_checkpoints',  # Jupyter notebook checkpoints
        'build',               # Build directories
        'dist',                # Distribution directories
        '.huggingface',        # Hugging Face cache
    ]
    
    # Check if each item exists and get its size
    found_items = []
    total_size = 0
    
    for item in items_to_check:
        if os.path.exists(item):
            item_size = get_size(item)
            found_items.append((item, item_size))
            total_size += item_size
    
    # Sort by size (largest first)
    found_items.sort(key=lambda x: x[1], reverse=True)
    
    # Print found items
    if found_items:
        print("Found the following items that can be removed to reduce project size:")
        print("\nItem\tSize\tDescription")
        print("-" * 60)
        
        for item, size in found_items:
            description = ""
            if item == 'venv':
                description = "Virtual environment (can be recreated with 'python -m venv venv')"
            elif item == '__pycache__':
                description = "Python cache files (automatically generated)"
            elif item == '.ipynb_checkpoints':
                description = "Jupyter notebook checkpoints"
            elif item == 'build':
                description = "Build directory"
            elif item == 'dist':
                description = "Distribution directory"
            elif item == '.huggingface':
                description = "Hugging Face cache"
            
            print(f"{item}\t{format_size(size)}\t{description}")
        
        print("\nTotal potential space savings: ", format_size(total_size))
        
        # Ask for confirmation
        print("\nWould you like to remove these items? This action cannot be undone.")
        print("1. Remove all items")
        print("2. Select specific items to remove")
        print("3. Cancel (don't remove anything)")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Remove all items
            for item, _ in found_items:
                try:
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                        print(f"Removed directory: {item}")
                    else:
                        os.remove(item)
                        print(f"Removed file: {item}")
                except Exception as e:
                    print(f"Error removing {item}: {e}")
            
            print("\nCleanup complete!")
            
        elif choice == '2':
            # Select specific items
            print("\nEnter the numbers of items you want to remove, separated by spaces:")
            for i, (item, size) in enumerate(found_items, 1):
                print(f"{i}. {item} ({format_size(size)})")
            
            selections = input("Your selections: ").split()
            
            try:
                selected_indices = [int(s) - 1 for s in selections]
                for idx in selected_indices:
                    if 0 <= idx < len(found_items):
                        item = found_items[idx][0]
                        try:
                            if os.path.isdir(item):
                                shutil.rmtree(item)
                                print(f"Removed directory: {item}")
                            else:
                                os.remove(item)
                                print(f"Removed file: {item}")
                        except Exception as e:
                            print(f"Error removing {item}: {e}")
                
                print("\nCleanup complete!")
            except ValueError:
                print("Invalid input. Cleanup cancelled.")
        
        else:
            print("Cleanup cancelled.")
    
    else:
        print("No items found that can be safely removed.")

if __name__ == "__main__":
    main()