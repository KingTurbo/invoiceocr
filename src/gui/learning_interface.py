import logging
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import base64
from config.config import MIN_FEATURES_FOR_VALID_ANCHOR

logger = logging.getLogger(__name__)

class VendorSelectionDialog(simpledialog.Dialog):
    def __init__(self, parent, title, existing_vendors, suggested_name):
        self.existing_vendors = existing_vendors
        self.suggested_name = suggested_name
        self.result = None
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text="Select an existing vendor or type a new name:").pack(pady=5)
        self.combo = ttk.Combobox(master, values=self.existing_vendors, width=40)
        self.combo.pack(padx=10, pady=5)
        self.combo.set(self.suggested_name)
        return self.combo # initial focus

    def apply(self):
        result = self.combo.get()
        if result and result.strip():
            self.result = result.strip()

class LearningInterface:
    def __init__(self, root, image_path, suggested_vendor_name, fields_config, image_obj, existing_vendors):
        self.root = root
        self.image_path = image_path
        self.vendor_name = suggested_vendor_name
        self.fields_config = fields_config
        self.pil_image = image_obj
        self.existing_vendors = existing_vendors
        
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
        self.drawn_rects = []   
        self.zoom_level = 1.0
        self.tk_image = None

        # --- DEFINITIVE FIX: CONTEXT-AWARE FEATURE GENERATION ---
        # Generate all features from the full image ONCE at the beginning.
        # This ensures descriptors have the correct "full image" context.
        logger.info("Pre-computing all features for the document to ensure contextual consistency...")
        self.full_image_cv = np.array(self.pil_image)
        self.orb_detector = cv2.ORB_create(nfeatures=5000) # Increased features for better coverage
        self.global_keypoints, self.global_descriptors = self.orb_detector.detectAndCompute(self.full_image_cv, None)
        logger.info(f"Found {len(self.global_keypoints)} total features in the document.")
        # --- END OF FIX ---

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
        
        self.reset_button = ttk.Button(button_frame, text="Reset Selections", command=self._on_reset)
        self.reset_button.pack(fill=tk.X, pady=2)
        
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

    def _on_reset(self):
        logger.info("Resetting template creation.")
        self.primary_anchor = None
        self.secondary_anchor = None
        self.defined_areas = {}

        for i in range(self.fields_listbox.size()):
            self.fields_listbox.itemconfig(i, {'fg': 'black', 'bg': 'white'})
        
        for rect_id in self.drawn_rects:
            self.canvas.delete(rect_id)
        self.drawn_rects = []
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
        self.start_x = None
        self._update_workflow_state()

    def _ask_for_vendor_name(self):
        dialog = VendorSelectionDialog(self.root, "Assign Vendor", self.existing_vendors, self.vendor_name)
        return dialog.result

    def _update_image_on_canvas(self):
        new_width = int(self.pil_image.width * self.zoom_level)
        new_height = int(self.pil_image.height * self.zoom_level)
        resized_image = self.pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        for rect_id in self.drawn_rects:
            coords = self.canvas.coords(rect_id) 
        self._redraw_persistent_rects()

    def _redraw_persistent_rects(self):
        pass

    def _on_zoom(self, event):
        if event.num == 4 or event.delta > 0: self.zoom_level *= 1.1
        elif event.num == 5 or event.delta < 0: self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        self._update_image_on_canvas()
    
    def _on_listbox_select(self, event):
        if self.current_step in ["**PRIMARY ANCHOR**", "**SECONDARY ANCHOR**"]:
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
        x1 = int(max(self.start_x, end_x) / self.zoom_level)
        y1 = int(max(self.start_y, end_y) / self.zoom_level)
        w, h = x1 - x0, y1 - y0

        if w < 10 or h < 10:
            self.canvas.delete(self.selection_rect)
            self.start_x = None
            return
            
        final_coords = {"x": x0, "y": y0, "width": w, "height": h}
        
        rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline='cyan', width=3)
        self.drawn_rects.append(rect_id)

        if self.current_step in ["**PRIMARY ANCHOR**", "**SECONDARY ANCHOR**"]:
            # --- DEFINITIVE FIX: Filter pre-computed global features ---
            # Instead of running ORB on a crop, filter the global features
            # that were computed with full-image context.
            
            # 1. Find the indices of keypoints that lie within the user's box
            anchor_kp_indices = []
            for i, kp in enumerate(self.global_keypoints):
                if x0 <= kp.pt[0] <= x1 and y0 <= kp.pt[1] <= y1:
                    anchor_kp_indices.append(i)
            
            if self.global_descriptors is None or len(anchor_kp_indices) < MIN_FEATURES_FOR_VALID_ANCHOR:
                messagebox.showerror("Anchor Too Weak", f"The selected area is not suitable. It contained only {len(anchor_kp_indices)} features, but the minimum required is {MIN_FEATURES_FOR_VALID_ANCHOR}.", parent=self.root)
                self.canvas.delete(self.drawn_rects.pop())
            else:
                # 2. Use the indices to select the corresponding keypoints and descriptors
                selected_kps = [self.global_keypoints[i] for i in anchor_kp_indices]
                selected_descriptors = self.global_descriptors[anchor_kp_indices]
                
                # 3. Get the global coordinates directly from the selected keypoints
                keypoints_pts_to_save = [kp.pt for kp in selected_kps]
                
                anchor_data = {
                    "bounding_box": final_coords,
                    "descriptors_b64": base64.b64encode(selected_descriptors).decode('utf-8'),
                    "keypoints_pts": keypoints_pts_to_save,
                }
                # --- END OF FIX ---
                
                if self.current_step == "**PRIMARY ANCHOR**":
                    self.primary_anchor = anchor_data
                    new_name = self._ask_for_vendor_name()
                    if new_name:
                        self.vendor_name = new_name
                        self.root.title(f"Template Creation for: {self.vendor_name}")
                        logger.info(f"Captured Primary Anchor with {len(selected_descriptors)} context-aware features.")
                    else: 
                        self.primary_anchor = None
                        self.canvas.delete(self.drawn_rects.pop())
                else:
                    self.secondary_anchor = anchor_data
                    logger.info(f"Captured Secondary Anchor with {len(selected_descriptors)} context-aware features.")
        else:
            selection_indices = self.fields_listbox.curselection()
            if not selection_indices:
                logger.warning("No field selected. Discarding selection.")
                self.canvas.delete(self.drawn_rects.pop())
            else:
                selected_field = self.fields_listbox.get(selection_indices[0])
                self.defined_areas[selected_field] = final_coords
                self.fields_listbox.itemconfig(selection_indices[0], {'fg': 'white', 'bg': 'green'})
                logger.info(f"Defined area for data field '{selected_field}'.")

        self.start_x = None
        self.canvas.delete(self.selection_rect)
        self.selection_rect = None
        self._update_workflow_state()

    def _on_save_press(self):
        proposed_template = { "vendor_name": self.vendor_name, "primary_anchor": self.primary_anchor, "secondary_anchor": self.secondary_anchor, "fields": [] }
        for name, coords in self.defined_areas.items():
            proposed_template["fields"].append({"field_name": name, "coordinates": coords})
        self.final_template = proposed_template
        messagebox.showinfo("Success", f"Template for '{self.vendor_name}' has been created successfully.", parent=self.root)
        self.root.destroy()

    def _on_cancel(self):
        self.final_template = None
        self.root.destroy()

def start_learning_gui(image_path: str, suggested_vendor_name: str, fields_config: list, image_obj: Image.Image, existing_vendors: list) -> dict | None:
    root = tk.Tk()
    app = LearningInterface(root, image_path, suggested_vendor_name, fields_config, image_obj, existing_vendors)
    root.mainloop()
    return app.final_template