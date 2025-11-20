import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import datetime
import time
import warnings

# --- PATH FIX FOR BUNDLED FFMPEG ---
# This block ensures that if you bundle ffmpeg.exe in the same folder as the app,
# the app will find it, regardless of Windows System PATH settings.
if getattr(sys, 'frozen', False):
    # If running as a compiled exe
    application_path = os.path.dirname(sys.executable)
else:
    # If running as a script
    application_path = os.path.dirname(os.path.abspath(__file__))

# Add the application directory to the OS PATH for this process only
os.environ["PATH"] += os.pathsep + application_path
# -----------------------------------

# Try to import whisper
try:
    import whisper
    import torch
except ImportError:
    # Fallback for GUI display if imports fail
    whisper = None
    torch = None

class WhisperQueueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Whisper Transcriber")
        self.root.geometry("900x700")
        
        # State variables
        self.queue = []
        self.is_processing = False
        self.stop_event = threading.Event()
        self.model = None
        self.current_model_name = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        self.setup_ui()
        self.log(f"System ready. Detected device: {self.device.upper()}")
        
        # Check for libraries
        if not whisper:
            self.log("ERROR: Libraries not found.")
        
        # Check for FFmpeg locally
        ffmpeg_path = os.path.join(application_path, "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            self.log("FFmpeg detected locally (Bundled Mode).")
        else:
            # Check if it's in global path
            import shutil
            if shutil.which("ffmpeg"):
                self.log("FFmpeg detected in System PATH.")
            else:
                self.log("WARNING: FFmpeg not found. Transcriptions will fail.")
                self.log(f"Please place 'ffmpeg.exe' in this folder: {application_path}")

    def setup_ui(self):
        # --- Top Control Panel ---
        control_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Model Selection
        ttk.Label(control_frame, text="Model Size:").grid(row=0, column=0, padx=5, sticky="w")
        self.model_var = tk.StringVar(value="base")
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=models, state="readonly", width=10)
        self.model_combo.grid(row=0, column=1, padx=5, sticky="w")

        # Language Selection
        ttk.Label(control_frame, text="Language:").grid(row=0, column=2, padx=5, sticky="w")
        self.lang_var = tk.StringVar(value="Auto-Detect")
        langs = ["Auto-Detect", "English", "Spanish", "French", "German", "Italian", "Japanese", "Chinese"]
        self.lang_combo = ttk.Combobox(control_frame, textvariable=self.lang_var, values=langs, state="readonly", width=15)
        self.lang_combo.grid(row=0, column=3, padx=5, sticky="w")

        # Task
        ttk.Label(control_frame, text="Task:").grid(row=0, column=4, padx=5, sticky="w")
        self.task_var = tk.StringVar(value="transcribe")
        self.task_combo = ttk.Combobox(control_frame, textvariable=self.task_var, values=["transcribe", "translate"], state="readonly", width=10)
        self.task_combo.grid(row=0, column=5, padx=5, sticky="w")

        # --- Queue Management ---
        queue_frame = ttk.LabelFrame(self.root, text="File Queue", padding=10)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Buttons
        btn_frame = ttk.Frame(queue_frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Add Folder", command=self.add_folder).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear Queue", command=self.clear_queue).pack(side="left", padx=5)
        
        self.lbl_count = ttk.Label(btn_frame, text="Files in queue: 0")
        self.lbl_count.pack(side="right", padx=10)

        # Treeview (List)
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

        # --- Output Settings ---
        out_frame = ttk.LabelFrame(self.root, text="Output Settings", padding=10)
        out_frame.pack(fill="x", padx=10, pady=5)

        self.export_srt = tk.BooleanVar(value=True)
        self.export_vtt = tk.BooleanVar(value=False)
        self.export_txt = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(out_frame, text="Export .srt (Subtitles)", variable=self.export_srt).pack(side="left", padx=10)
        ttk.Checkbutton(out_frame, text="Export .vtt (Web Captions)", variable=self.export_vtt).pack(side="left", padx=10)
        ttk.Checkbutton(out_frame, text="Export .txt (Transcription)", variable=self.export_txt).pack(side="left", padx=10)

        # --- Execution ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x")

        self.btn_start = ttk.Button(action_frame, text="START PROCESSING", command=self.start_processing_thread)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=5)
        
        self.btn_stop = ttk.Button(action_frame, text="STOP", command=self.stop_processing, state="disabled")
        self.btn_stop.pack(side="right", padx=5)

        # Progress
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=5)

        # Logs
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)

    # --- Logic ---

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
            valid_exts = ('.mp4', '.mkv', '.mp3', '.wav', '.m4a', '.flac', '.avi', '.mov', '.webm')
            for root_dir, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(valid_exts):
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
        
        thread = threading.Thread(target=self.process_queue)
        thread.daemon = True
        thread.start()

    def stop_processing(self):
        if messagebox.askyesno("Stop", "Are you sure you want to stop processing after the current file?"):
            self.stop_event.set()
            self.log("Stopping after current file...")

    def process_queue(self):
        selected_model = self.model_var.get()
        selected_lang = self.lang_var.get()
        task = self.task_var.get()

        if self.current_model_name != selected_model:
            self.log(f"Loading model '{selected_model}' on {self.device.upper()}... (This may take a moment)")
            try:
                self.model = whisper.load_model(selected_model, device=self.device)
                self.current_model_name = selected_model
                self.log("Model loaded successfully.")
            except Exception as e:
                self.log(f"Error loading model: {e}")
                self.reset_ui()
                return

        total = len(self.queue)
        
        for index, item in enumerate(self.queue):
            if self.stop_event.is_set():
                break
            if item["status"] == "Done":
                continue

            self.update_status(item["id"], "Processing...")
            self.root.update_idletasks()
            self.progress["value"] = (index / total) * 100
            file_path = item["path"]
            self.log(f"Transcribing: {os.path.basename(file_path)}")

            try:
                start_time = time.time()
                options = {"task": task}
                if selected_lang != "Auto-Detect":
                    lang_map = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", 
                                "Italian": "it", "Japanese": "ja", "Chinese": "zh"}
                    if selected_lang in lang_map:
                        options["language"] = lang_map[selected_lang]

                fp16 = (self.device == "cuda")
                result = self.model.transcribe(file_path, fp16=fp16, **options)
                
                duration = time.time() - start_time
                self.log(f"Finished in {duration:.2f}s")

                base_name = os.path.splitext(file_path)[0]
                if self.export_srt.get():
                    self.write_srt(result["segments"], base_name + ".srt")
                if self.export_vtt.get():
                    self.write_vtt(result["segments"], base_name + ".vtt")
                if self.export_txt.get():
                    with open(base_name + ".txt", "w", encoding="utf-8") as f:
                        f.write(result["text"])

                self.update_status(item["id"], "Done")
                item["status"] = "Done"

            except Exception as e:
                self.log(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                self.update_status(item["id"], "Error")
                item["status"] = "Error"

        self.progress["value"] = 100
        self.log("Queue processing complete.")
        self.reset_ui()

    def reset_ui(self):
        self.is_processing = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.model_combo.config(state="readonly")

    def update_status(self, item_id, status):
        self.tree.set(item_id, "status", status)
        self.tree.see(item_id)

    def format_timestamp(self, seconds, always_include_hours=False, decimal_marker=','):
        assert seconds >= 0, "non-negative timestamp expected"
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
