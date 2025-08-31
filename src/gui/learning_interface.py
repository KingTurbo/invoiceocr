import logging
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import imagehash # <-- NEW: Import the hashing library

logger = logging.getLogger(__name__)

class LearningInterface:
    def __init__(self, root, image_path, suggested_vendor_name, fields_config):
        self.root = root
        self.image_path = image_path
        self.vendor_name = suggested_vendor_name
        
        self.defined_areas = {}
        self.mandatory_fields = {f['name'] for f in fields_config if f.get('mandatory') is True}
        self.all_fields = [f['name'] for f in fields_config]

        # --- MODIFIED: We now store the hash, not the text ---
        self.identifier_hash = None

        self.start_x = None
        self.start_y = None
        self.selection_rect = None

        self.zoom_level = 1.0
        self.pil_image = Image.open(self.image_path)
        self.tk_image = None
        self.final_template = None

        self.setup_ui()
        self._on_listbox_select(None)
        
        # ... (bindings are the same) ...
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)

    # ... (setup_ui, _update_image_on_canvas, _on_zoom, _on_listbox_select are the same) ...
    def setup_ui(self):
        self.root.title(f"Template Creation for: {self.vendor_name}")
        self.root.geometry("1400x900")
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_panel = ttk.Frame(main_frame, width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(control_panel, text="Fields to Define (Mandatory in Blue):", font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
        self.fields_listbox = tk.Listbox(control_panel, height=15, exportselection=False)
        for field in self.all_fields:
            self.fields_listbox.insert(tk.END, field)
            if field == "**VENDOR IDENTIFIER**":
                self.fields_listbox.itemconfig(tk.END, {'bg': 'dark red', 'fg': 'white', 'selectbackground': 'red'})
            elif field in self.mandatory_fields:
                self.fields_listbox.itemconfig(tk.END, {'selectbackground': '#336699'})
        self.fields_listbox.pack(fill=tk.X, pady=5)
        self.fields_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        self.save_button = ttk.Button(button_frame, text="Save Template", command=self._on_save, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=2)
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(fill=tk.X, pady=2)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.instruction_label = ttk.Label(right_panel, text="Instruction:", font=("Helvetica", 12), wraplength=1000)
        self.instruction_label.pack(anchor=tk.NW, pady=(0, 5))
        canvas_frame = ttk.Frame(right_panel, relief="sunken", borderwidth=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="gray")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._update_image_on_canvas()

    def _update_image_on_canvas(self):
        new_width = int(self.pil_image.width * self.zoom_level)
        new_height = int(self.pil_image.height * self.zoom_level)
        resized_image = self.pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
    def _on_zoom(self, event):
        if event.num == 4 or event.delta > 0: self.zoom_level *= 1.1
        elif event.num == 5 or event.delta < 0: self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        self._update_image_on_canvas()

    def _on_listbox_select(self, event):
        selection_indices = self.fields_listbox.curselection()
        if not selection_indices:
            self.instruction_label.config(text="Select a field from the list to define its area. (Ctrl+Scroll to Zoom)")
            return
        selected_field = self.fields_listbox.get(selection_indices[0])
        if selected_field == "**VENDOR IDENTIFIER**":
            self.instruction_label.config(text=f"Now defining: '{selected_field}'. Select a unique, permanent visual feature like a logo.")
        else:
            self.instruction_label.config(text=f"Now defining: '{selected_field}'. Click and drag on the image to select its area.")

    def _validate_state(self):
        defined_fields = set(self.defined_areas.keys())
        if self.mandatory_fields.issubset(defined_fields):
            self.save_button.config(state=tk.NORMAL)
        else:
            self.save_button.config(state=tk.DISABLED)

    def _on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.selection_rect: self.canvas.delete(self.selection_rect)
        self.selection_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='lime green', width=2, dash=(4, 4))

    def _on_drag(self, event):
        if self.start_x is None: return
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.selection_rect, self.start_x, self.start_y, cur_x, cur_y)

    def _on_release(self, event):
        if self.start_x is None: return
        selection_indices = self.fields_listbox.curselection()
        if not selection_indices:
            logger.warning("Attempted to define an area without selecting a field first.")
            self.canvas.delete(self.selection_rect)
            self.start_x = None
            return

        selected_field = self.fields_listbox.get(selection_indices[0])
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x = int(min(self.start_x, end_x) / self.zoom_level)
        y = int(min(self.start_y, end_y) / self.zoom_level)
        w = int(abs(self.start_x - end_x) / self.zoom_level)
        h = int(abs(self.start_y - end_y) / self.zoom_level)

        if w > 0 and h > 0:
            final_coords = {"x": x, "y": y, "width": w, "height": h}
            
            # --- MODIFIED: This is the core change to use hashing ---
            if selected_field == "**VENDOR IDENTIFIER**":
                # Crop the original image to the selected area
                box = (x, y, x + w, y + h)
                identifier_image_crop = self.pil_image.crop(box)
                
                # Calculate the perceptual hash
                self.identifier_hash = str(imagehash.phash(identifier_image_crop))
                self.defined_areas[selected_field] = final_coords
                self.fields_listbox.itemconfig(selection_indices[0], {'fg': 'white', 'bg': 'dark green'})
                logger.info(f"Captured '{selected_field}' with hash '{self.identifier_hash}' at {final_coords}")

                # Prompt for name *after* successful hash
                new_name = simpledialog.askstring("Vendor Name", "Please enter a name for this vendor:", initialvalue=self.vendor_name, parent=self.root)
                if new_name and new_name.strip():
                    self.vendor_name = new_name.strip()
                    self.root.title(f"Template Creation for: {self.vendor_name}")
                    logger.info(f"Vendor name set to '{self.vendor_name}'.")
                else:
                    logger.warning("User cancelled or entered an empty vendor name.")
            else:
                self.defined_areas[selected_field] = final_coords
                self.fields_listbox.itemconfig(selection_indices[0], {'fg': 'white', 'bg': 'green'})
                logger.info(f"Captured '{selected_field}': {final_coords}")
            
            self._validate_state()
            
        self.start_x = None

    def _on_save(self):
        identifier_coords = self.defined_areas.pop("**VENDOR IDENTIFIER**")
        # --- MODIFIED: Save the hash, not the text ---
        self.final_template = {
            "vendor_name": self.vendor_name,
            "identifier_area": identifier_coords,
            "identifier_hash": self.identifier_hash,
            "fields": []
        }
        for name, coords in self.defined_areas.items():
            self.final_template["fields"].append({"field_name": name, "coordinates": coords})

        logger.info(f"Final template data captured: {self.final_template}")
        self.root.destroy()

    # ... (_on_cancel and start_learning_gui are the same) ...
    def _on_cancel(self):
        logger.warning("Template creation cancelled by user.")
        self.final_template = None
        self.root.destroy()


def start_learning_gui(image_path: str, suggested_vendor_name: str, fields_config: list) -> dict | None:
    logger.info(f"Starting GUI for vendor '{suggested_vendor_name}' with image: {image_path}")
    root = tk.Tk()
    app = LearningInterface(root, image_path, suggested_vendor_name, fields_config)
    root.mainloop()
    return app.final_template

#### for direct testing run this script

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.ocr import engine
    from src.config import FIELDS_CONFIG 
    
    test_image_path = project_root / "tests" / "test_invoice.png"

    print("Running GUI in standalone test mode...")
    
    if not test_image_path.exists():
        print(f"CRITICAL: Test image not found at expected location: {test_image_path}")
    else:
        logging.basicConfig(level=logging.INFO)
        engine.initialize_reader()
        created_template = start_learning_gui(str(test_image_path), "Test Vendor Inc.", FIELDS_CONFIG)
        if created_template:
            print("\n--- GUI Returned Template Data ---")
            import json
            print(json.dumps(created_template, indent=2))
            print("---------------------------------")
        else:
            print("\nGUI was cancelled. No template data returned.")