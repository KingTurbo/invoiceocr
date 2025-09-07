
import logging
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import base64

logger = logging.getLogger(__name__)

class LearningInterface:
    def __init__(self, root, image_path, suggested_vendor_name, fields_config, image_obj):
        self.root = root
        self.image_path = image_path
        self.vendor_name = suggested_vendor_name
        self.fields_config = fields_config
        self.pil_image = image_obj
        
        self.defined_areas = {}
        self.data_fields = [f['name'] for f in fields_config if not f['name'].startswith('**')]
        self.mandatory_data_fields = {f['name'] for f in fields_config if f.get('mandatory') and not f['name'].startswith('**')}

        self.primary_anchor = None
        self.secondary_anchor = None
        self.final_template = None
        
        self.current_step = "**PRIMARY ANCHOR**"

        self.start_x = None
        self.start_y = None
        self.selection_rect = None
        self.zoom_level = 1.0
        self.tk_image = None

        self.setup_ui()
        self._update_workflow_state()

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)

    def setup_ui(self):
        self.root.title(f"Template Creation for: {self.vendor_name}")
        self.root.geometry("1400x900")
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_panel = ttk.Frame(main_frame, width=350)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.workflow_label = ttk.Label(control_panel, text="Step 1: Define Primary Anchor", font=("Helvetica", 12, "bold"))
        self.workflow_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(control_panel, text="Data Fields to Define:", font=("Helvetica", 10)).pack(anchor=tk.W)
        self.fields_listbox = tk.Listbox(control_panel, height=15, exportselection=False)
        for field in self.data_fields:
            self.fields_listbox.insert(tk.END, field)
            if field in self.mandatory_data_fields:
                self.fields_listbox.itemconfig(tk.END, {'selectbackground': '#336699'})
        self.fields_listbox.pack(fill=tk.X, pady=5)
        self.fields_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)

        button_frame = ttk.Frame(control_panel)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        self.save_button = ttk.Button(button_frame, text="Save Template", command=self._on_save_press, state=tk.DISABLED)
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
        if self.current_step in ["**PRIMARY ANCHOR**", "**SECONDARY ANCHOR**"]:
            # Ignore listbox selections while defining anchors
            self.fields_listbox.selection_clear(0, tk.END)
        self._update_workflow_state()

    def _update_workflow_state(self):
        if self.primary_anchor is None:
            self.current_step = "**PRIMARY ANCHOR**"
            self.workflow_label.config(text="Step 1: Define Primary Anchor")
            self.instruction_label.config(text="Select a unique, permanent visual feature (e.g., a logo) as the PRIMARY anchor.")
            self.fields_listbox.config(state=tk.DISABLED)
        elif self.secondary_anchor is None:
            self.current_step = "**SECONDARY ANCHOR**"
            self.workflow_label.config(text="Step 2: Define Secondary Anchor")
            self.instruction_label.config(text="SUCCESS! Now select another unique feature, FAR FROM the first, as the SECONDARY anchor.")
            self.fields_listbox.config(state=tk.DISABLED)
        else:
            self.current_step = "DATA_FIELDS"
            self.workflow_label.config(text="Step 3: Define Data Fields")
            self.fields_listbox.config(state=tk.NORMAL)
            selection_indices = self.fields_listbox.curselection()
            if not selection_indices:
                self.instruction_label.config(text="Anchors defined. Select a field from the list and draw a box around its value.")
            else:
                selected_field = self.fields_listbox.get(selection_indices[0])
                self.instruction_label.config(text=f"Now defining: '{selected_field}'. Click and drag on the image to select its area.")
        
        self._validate_state()

    def _validate_state(self):
        defined_data_fields = set(self.defined_areas.keys())
        if self.primary_anchor and self.secondary_anchor and self.mandatory_data_fields.issubset(defined_data_fields):
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
        
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x0 = int(min(self.start_x, end_x) / self.zoom_level)
        y0 = int(min(self.start_y, end_y) / self.zoom_level)
        w = int(abs(self.start_x - end_x) / self.zoom_level)
        h = int(abs(self.start_y - end_y) / self.zoom_level)

        if w < 10 or h < 10:
            self.canvas.delete(self.selection_rect)
            self.start_x = None
            return
            
        final_coords = {"x": x0, "y": y0, "width": w, "height": h}

        if self.current_step in ["**PRIMARY ANCHOR**", "**SECONDARY ANCHOR**"]:
            box = (x0, y0, x0 + w, y0 + h)
            anchor_image_crop = self.pil_image.crop(box)
            cv_crop = np.array(anchor_image_crop)
            
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(cv_crop, None)
            
            if descriptors is None or len(descriptors) < 20:
                messagebox.showerror("Anchor Too Weak", f"The selected area is not suitable. It produced only {len(descriptors) if descriptors is not None else 0} features. Please select a detailed, high-contrast logo.", parent=self.root)
                self.canvas.delete(self.selection_rect)
            else:
                # CRITICAL FIX: Convert keypoint coordinates to the GLOBAL coordinate system
                global_keypoints_pts = [(kp.pt[0] + x0, kp.pt[1] + y0) for kp in keypoints]

                anchor_data = {
                    "bounding_box": final_coords,
                    "descriptors_b64": base64.b64encode(descriptors).decode('utf-8'),
                    "keypoints_pts": global_keypoints_pts,
                    "source_shape": {'width': w, 'height': h}
                }
                
                if self.current_step == "**PRIMARY ANCHOR**":
                    self.primary_anchor = anchor_data
                    logger.info(f"Captured Primary Anchor with {len(descriptors)} ORB features.")
                    # First time an anchor is set, ask for vendor name
                    new_name = simpledialog.askstring("Vendor Name", "Please enter a name for this vendor:", initialvalue=self.vendor_name, parent=self.root)
                    if new_name and new_name.strip():
                        self.vendor_name = new_name.strip()
                        self.root.title(f"Template Creation for: {self.vendor_name}")
                else: # Secondary anchor
                    self.secondary_anchor = anchor_data
                    logger.info(f"Captured Secondary Anchor with {len(descriptors)} ORB features.")

                self.canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline='cyan', width=3)
                self.selection_rect = None

        else: # Defining a data field
            selection_indices = self.fields_listbox.curselection()
            if not selection_indices:
                logger.warning("Attempted to define a data area without selecting a field first.")
            else:
                selected_field = self.fields_listbox.get(selection_indices[0])
                self.defined_areas[selected_field] = final_coords
                self.fields_listbox.itemconfig(selection_indices[0], {'fg': 'white', 'bg': 'green'})
                logger.info(f"Defined area for data field '{selected_field}'.")

        self.start_x = None
        self._update_workflow_state()

    def _on_save_press(self):
        proposed_template = {
            "vendor_name": self.vendor_name,
            "primary_anchor": self.primary_anchor,
            "secondary_anchor": self.secondary_anchor,
            "fields": []
        }
        for name, coords in self.defined_areas.items():
            proposed_template["fields"].append({"field_name": name, "coordinates": coords})

        # The confirmation step is removed for brevity but can be added back if needed
        self.final_template = proposed_template
        messagebox.showinfo("Success", f"Template for '{self.vendor_name}' has been created successfully.", parent=self.root)
        self.root.destroy()

    def _on_cancel(self):
        self.final_template = None
        self.root.destroy()


def start_learning_gui(image_path: str, suggested_vendor_name: str, fields_config: list, image_obj: Image.Image) -> dict | None:
    root = tk.Tk()
    app = LearningInterface(root, image_path, suggested_vendor_name, fields_config, image_obj)
    root.mainloop()
    return app.final_template