import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import os
import time

IMAGE_PATH = "latest_training_viz.png"

fig, ax = plt.subplots(figsize=(12, 3))
fig.canvas.manager.set_window_title('Training Progress Monitor')
ax.axis('off')

img_display = None
last_mtime = 0

def update_gui(frame):
    global last_mtime, img_display
    
    if not os.path.exists(IMAGE_PATH):
        ax.set_title("Waiting for training script to output image...")
        return
    
    current_mtime = os.path.getmtime(IMAGE_PATH)
    if current_mtime != last_mtime:
        try:
            img = mpimg.imread(IMAGE_PATH)
            
            if img_display is None:
                img_display = ax.imshow(img)
            else:
                img_display.set_data(img)
            
            last_mtime = current_mtime
            current_time_str = time.strftime('%H:%M:%S')
            ax.set_title(f"Last updated: {current_time_str}")
            
        except Exception:
            pass
            
    return img_display,

ani = animation.FuncAnimation(fig, update_gui, interval=1000, cache_frame_data=False)

plt.show()
