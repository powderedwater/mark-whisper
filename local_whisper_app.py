import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import datetime
import time
import warnings
import shutil

# --- FIX FOR "NoneType object has no attribute write" ---
# When running as a windowed GUI (no console), sys.stderr is None.
# Whisper/tqdm tries to write progress bars to it, causing a crash.
class NullWriter:
    def write(self, data):
        pass
    def flush(self):
        pass

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()
# -------------------------------------------------------

# Try to import windnd for Drag and Drop (Windows only)
try:
    import windnd
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False

# --- PATH FIX FOR BUNDLED FFMPEG ---
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.environ["PATH"] += os.pathsep + application_path
# -----------------------------------

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

class WhisperQueueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Whisper Transcriber")
        self.root.geometry("950x750")
        
        # State variables
        self.queue = []
        self.is_processing = False
        self.stop_event = threading.Event()
        self.model = None
        self.current_model_name = None
        
        # Device detection
        self.has_gpu = torch and torch.cuda.is_available()
        self.device_map = {"GPU (CUDA)": "cuda", "CPU": "cpu"}
        
        self.setup_ui()
        self.setup_drag_drop()
        
        self.log(f"System ready. Application Path: {application_path}")
        
        if self.has_gpu:
            self.log("NVIDIA GPU detected. Defaulting to CUDA.")
        else:
            self.log("No GPU detected. Defaulting to CPU.")

        # Check FFmpeg
        self.check_ffmpeg()

    def check_ffmpeg(self):
        ffmpeg_path = os.path.join(application_path, "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            self.log("FFmpeg status: Bundled (OK)")
        elif shutil.which("ffmpeg"):
            self.log("FFmpeg status: System PATH (OK)")
        else:
            self.log("CRITICAL ERROR: FFmpeg not found.")
            self.log("Transcriptions will fail. Please ensure ffmpeg.exe is in the app folder.")

    def setup_drag_drop(self):
        if DRAG_DROP_AVAILABLE:
            try:
                windnd.hook_dropfiles(self.root, func=self.on_files_dropped)
                self.log("Drag and Drop enabled. Drop files anywhere on this window.")
            except Exception as e:
                self.log(f"Drag and Drop initialization failed: {e}")
        else:
            self.log("Drag and Drop library (windnd) not found.")

    def on_files_dropped(self, filenames):
        count = 0
        for name in filenames:
            if isinstance(name, bytes):
                name = name.decode('mbcs')
            
            if os.path.isdir(name):
                self.add_folder_path(name)
            elif os.path.isfile(name):
                if self.is_valid_file(name):
                    self.add_to_queue(name)
                    count += 1
        
        if count > 0:
            self.log(f"Added {count} files via Drag & Drop.")

    def is_valid_file(self, filepath):
        valid_exts = ('.mp4', '.mkv', '.mp3', '.wav', '.m4a', '.flac', '.avi', '.mov', '.webm')
        return filepath.lower().endswith(valid_exts)

    def setup_ui(self):
        # --- Configuration ---
        control_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Row 1
        ttk.Label(control_frame, text="Model Size:").grid(row=0, column=0, padx=5, sticky="w")
        self.model_var = tk.StringVar(value="base")
        models = ["tiny", "base", "small", "medium", "large", "large-v3"]
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=models, state="readonly", width=10)
        self.model_combo.grid(row=0, column=1, padx=5, sticky="w")

        ttk.Label(control_frame, text="Device:").grid(row=0, column=2, padx=5, sticky="w")
        self.device_var = tk.StringVar(value="GPU (CUDA)" if self.has_gpu else "CPU")
        device_options = ["GPU (CUDA)", "CPU"] if self.has_gpu else ["CPU"]
        self.device_combo = ttk.Combobox(control_frame, textvariable=self.device_var, values=device_options, state="readonly", width=12)
        self.device_combo.grid(row=0, column=3, padx=5, sticky="w")

        ttk.Label(control_frame, text="Language:").grid(row=0, column=4, padx=5, sticky="w")
        self.lang_var = tk.StringVar(value="Auto-Detect")
        langs = ["Auto-Detect", "English", "Spanish", "French", "German", "Italian", "Japanese", "Chinese"]
        self.lang_combo = ttk.Combobox(control_frame, textvariable=self.lang_var, values=langs, state="readonly", width=15)
        self.lang_combo.grid(row=0, column=5, padx=5, sticky="w")

        self.task_var = tk.StringVar(value="transcribe")
        self.task_combo = ttk.Combobox(control_frame, textvariable=self.task_var, values=["transcribe", "translate"], state="readonly", width=10)
        self.task_combo.grid(row=0, column=6, padx=5, sticky="w")

        # --- Output Settings ---
        out_frame = ttk.LabelFrame(self.root, text="Output Settings", padding=10)
        out_frame.pack(fill="x", padx=10, pady=5)

        self.use_source_folder = tk.BooleanVar(value=True)
        self.check_source = ttk.Checkbutton(out_frame, text="Save to Source Folder", variable=self.use_source_folder, command=self.toggle_output_entry)
        self.check_source.grid(row=0, column=0, padx=5, sticky="w")
        
        self.output_path_var = tk.StringVar()
        self.entry_output = ttk.Entry(out_frame, textvariable=self.output_path_var, width=50, state="disabled")
        self.entry_output.grid(row=0, column=1, padx=5, sticky="w")
        
        self.btn_browse = ttk.Button(out_frame, text="Browse...", command=self.browse_output_folder, state="disabled")
        self.btn_browse.grid(row=0, column=2, padx=5, sticky="w")

        # Formats
        format_frame = ttk.Frame(out_frame)
        format_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10,0))
        
        self.export_srt = tk.BooleanVar(value=True)
        self.export_vtt = tk.BooleanVar(value=False)
        self.export_txt = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(format_frame, text="Export .srt", variable=self.export_srt).pack(side="left", padx=10)
        ttk.Checkbutton(format_frame, text="Export .vtt", variable=self.export_vtt).pack(side="left", padx=10)
        ttk.Checkbutton(format_frame, text="Export .txt", variable=self.export_txt).pack(side="left", padx=10)

        # --- Queue ---
        queue_frame = ttk.LabelFrame(self.root, text="File Queue (Drag & Drop Supported)", padding=10)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=5)

        btn_frame = ttk.Frame(queue_frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Add Folder", command=self.add_folder).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear Queue", command=self.clear_queue).pack(side="left", padx=5)
        self.lbl_count = ttk.Label(btn_frame, text="Files in queue: 0")
        self.lbl_count.pack(side="right", padx=10)

        columns = ("status", "filename", "path")
        self.tree = ttk.Treeview(queue_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("status", text="Status")
        self.tree.heading("filename", text="Filename")
        self.tree.heading("path", text="Full Path")
        
        self.tree.column("status", width=100, anchor="center")
        self.tree.column("filename", width=200)
        self.tree.column("path", width=500)
        
        scrollbar = ttk.Scrollbar(queue_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Execution ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x")

        self.btn_start = ttk.Button(action_frame, text="START PROCESSING", command=self.start_processing_thread)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_stop = ttk.Button(action_frame, text="STOP", command=self.stop_processing, state="disabled")
        self.btn_stop.pack(side="right", padx=5)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=5)

        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)

    # --- UI Logic ---
    
    def toggle_output_entry(self):
        if self.use_source_folder.get():
            self.entry_output.config(state="disabled")
            self.btn_browse.config(state="disabled")
        else:
            self.entry_output.config(state="normal")
            self.btn_browse.config(state="normal")

    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_path_var.set(folder)

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def add_files(self):
        filetypes = [("Media Files", "*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac *.avi *.mov *.webm"), ("All Files", "*.*")]
        files = filedialog.askopenfilenames(title="Select Media Files", filetypes=filetypes)
        for f in files:
            self.add_to_queue(f)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            self.add_folder_path(folder)

    def add_folder_path(self, folder):
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if self.is_valid_file(file):
                    self.add_to_queue(os.path.join(root_dir, file))

    def add_to_queue(self, path):
        for item in self.queue:
            if item['path'] == path:
                return
        filename = os.path.basename(path)
        item_id = self.tree.insert("", "end", values=("Pending", filename, path))
        self.queue.append({"id": item_id, "path": path, "status": "Pending"})
        self.lbl_count.config(text=f"Files in queue: {len(self.queue)}")

    def clear_queue(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot clear queue while processing.")
            return
        self.tree.delete(*self.tree.get_children())
        self.queue = []
        self.lbl_count.config(text="Files in queue: 0")

    def start_processing_thread(self):
        if not self.queue:
            messagebox.showinfo("Empty Queue", "Please add files to the queue first.")
            return
        if not whisper:
            messagebox.showerror("Missing Libraries", "Whisper library not loaded.")
            return
            
        self.is_processing = True
        self.stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.model_combo.config(state="disabled")
        self.device_combo.config(state="disabled")
        
        thread = threading.Thread(target=self.process_queue)
        thread.daemon = True
        thread.start()

    def stop_processing(self):
        if messagebox.askyesno("Stop", "Stop processing after current file?"):
            self.stop_event.set()
            self.log("Stopping after current file...")

    def process_queue(self):
        model_size = self.model_var.get()
        device_selection = self.device_combo.get()
        target_device = self.device_map.get(device_selection, "cpu")
        
        if self.current_model_name != model_size or self.model is None:
            self.log(f"Loading '{model_size}' on {target_device}...")
            self.log("Downloading model if not present (this may look stuck, please wait)...")
            
            try:
                self.model = whisper.load_model(model_size, device=target_device)
                self.current_model_name = model_size
                self.log("Model loaded successfully.")
            except Exception as e:
                # Fallback logic
                if "CUDA" in str(e) or target_device == "cuda":
                    self.log(f"GPU Error: {e}. Switching to CPU...")
                    try:
                        self.model = whisper.load_model(model_size, device="cpu")
                        self.current_model_name = model_size
                        self.log("Model loaded on CPU.")
                    except Exception as e2:
                        self.log(f"CPU Load Error: {e2}")
                        self.reset_ui()
                        return
                else:
                    self.log(f"Load Error: {e}")
                    self.reset_ui()
                    return

        total = len(self.queue)
        lang_setting = self.lang_var.get()
        task_setting = self.task_var.get()
        
        lang_code = None
        if lang_setting != "Auto-Detect":
            lang_map = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", 
                        "Italian": "it", "Japanese": "ja", "Chinese": "zh"}
            lang_code = lang_map.get(lang_setting)

        for index, item in enumerate(self.queue):
            if self.stop_event.is_set():
                break
            if item["status"] == "Done":
                continue

            self.update_status(item["id"], "Processing...")
            self.progress["value"] = (index / total) * 100
            file_path = item["path"]
            self.log(f"Transcribing: {os.path.basename(file_path)}")

            try:
                start_time = time.time()
                options = {"task": task_setting}
                if lang_code:
                    options["language"] = lang_code
                
                use_fp16 = (self.model.device.type == "cuda")
                result = self.model.transcribe(file_path, fp16=use_fp16, **options)
                
                duration = time.time() - start_time
                self.log(f"Finished in {duration:.2f}s")

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                if self.use_source_folder.get():
                    output_dir = os.path.dirname(file_path)
                else:
                    output_dir = self.output_path_var.get()
                    if not output_dir:
                        output_dir = os.path.dirname(file_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                output_base = os.path.join(output_dir, base_name)

                if self.export_srt.get():
                    self.write_srt(result["segments"], output_base + ".srt")
                if self.export_vtt.get():
                    self.write_vtt(result["segments"], output_base + ".vtt")
                if self.export_txt.get():
                    with open(output_base + ".txt", "w", encoding="utf-8") as f:
                        f.write(result["text"])

                self.update_status(item["id"], "Done")
                item["status"] = "Done"

            except Exception as e:
                self.log(f"Error: {str(e)}")
                self.update_status(item["id"], "Error")
                item["status"] = "Error"

        self.progress["value"] = 100
        self.log("Processing complete.")
        self.reset_ui()

    def reset_ui(self):
        self.is_processing = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.model_combo.config(state="readonly")
        self.device_combo.config(state="readonly")

    def update_status(self, item_id, status):
        self.tree.set(item_id, "status", status)
        self.tree.see(item_id)

    def format_timestamp(self, seconds, always_include_hours=False, decimal_marker=','):
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else "00:"
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

    def write_srt(self, segments, path):
        with open(path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start = self.format_timestamp(segment["start"], always_include_hours=True, decimal_marker=',')
                end = self.format_timestamp(segment["end"], always_include_hours=True, decimal_marker=',')
                text = segment["text"].strip().replace('-->', '->')
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def write_vtt(self, segments, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start = self.format_timestamp(segment["start"], decimal_marker='.')
                end = self.format_timestamp(segment["end"], decimal_marker='.')
                text = segment["text"].strip().replace('-->', '->')
                f.write(f"{start} --> {end}\n{text}\n\n")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    app = WhisperQueueApp(root)
    root.mainloop()
